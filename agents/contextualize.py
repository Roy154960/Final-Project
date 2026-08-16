"""
Runs once per turn, spliced into graph.py between input_guard and the
supervisor:

    input_guard --clean--> contextualize --> supervisor (Phase 3 loop)

Why this node exists: every specialist (specialists.py's
_last_human_text) and the supervisor's own routing decision
(supervisor.py's _current_turn_context) deliberately look at ONLY the
current turn's raw HumanMessage -- exactly right WITHIN a turn (a
mid-turn re-route must see the same question every specialist that turn
sees), but it leaves a real gap ACROSS turns: nothing downstream of
input_guard ever sees more than the single latest message, so a bare
follow-up like "which size is best?" right after a question about
brushes has no way to resolve WHAT is being sized. Passed straight to
`retrieve` as the query (exactly what every retrieval-shaped specialist
does with _last_human_text's return value), it has no chance of
surfacing brush-related chunks, because the word "brush" never appears
in it.

This node rewrites that kind of follow-up into a standalone question,
using the prior conversation as the only source of the missing context,
BEFORE routing ever sees it -- so the fix lives in exactly one place
rather than needing every retrieval-shaped specialist edited
individually.

Design choices, same "confirmed live run, not just theorized" spirit as
supervisor.py's four safety nets:

- Skips the LLM call entirely on a turn's first-ever message (no prior
  conversation exists to resolve anything against) -- a plain {} no-op,
  same as output_guard's "nothing found" return. This also means ask()
  / the CLI / the Phase 5 eval script (all single-turn, stateless
  callers -- see graph.py's own docstring) never pay for this node at
  all: `prior` is always empty on their one and only turn.
- Rewrites the HumanMessage IN PLACE: same `.id`, so state.py's
  add_messages reducer replaces it rather than appending a duplicate --
  the exact pattern output_guard.py's redaction already uses (see its
  own module docstring), applied here to the input side instead of the
  output side. This means supervisor.py's _current_turn_context and
  every specialist's _last_human_text need ZERO changes to benefit:
  they already just read "the latest HumanMessage," and by the time
  either one runs, that message's content IS the standalone version.
- A model that ignores "output it exactly unchanged" and rewrites a
  question that didn't need it is a cosmetic problem (the rewritten
  version still means the same thing); a model that instead tries to
  ANSWER the question, or returns something empty, is not -- so both
  are guarded against structurally below, not just prompted against,
  the same "structural guardrail over prompt wording" preference
  specialists.py and guardrails.py both already state. An empty or
  implausibly-longer-than-a-rewrite response falls back to the original
  message, untouched.
- Loyalty to the original wording is ALSO enforced structurally, not
  just by prompt instruction: every content word in the original
  message -- everything except a small closed set of pronouns this node
  exists to resolve, and ordinary function words -- must survive,
  unchanged, into the rewrite (see `_dropped_original_words`'s own
  docstring). A rewrite that resolves the referent but ALSO quietly
  drops or rewords something else falls back to the untouched original,
  the same "prefer an unresolved-but-faithful message over a resolved-
  but-altered one" tradeoff this project applies consistently: it is
  better to occasionally miss a resolvable follow-up than to silently
  change what the person actually said.
"""

import re
import sys
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agents.llm_provider import get_chat_model
from agents.prompts import CONTEXTUALIZE_SYSTEM_PROMPT, CONTEXTUALIZE_USER_TURN_TEMPLATE
from agents.specialists import (
    Specialist,
    classify_request_difficulty,
    _message_carries_attachment,
)
from agents.state import AgentState

# How many prior messages (across all earlier turns, not this one) to
# show the rewrite model. Counted in whole messages, not tokens --
# generous enough that a follow-up several turns deep can still resolve
# against its original topic (e.g. "and what about the beginner ones?"
# three turns after a product_search about brushes), bounded so this
# call's prompt doesn't grow unboundedly over a long conversation the
# way the checkpointer's own full-history state does.
_CONTEXT_WINDOW_MESSAGES = 12

# A rewritten question that comes back drastically longer than the
# original is almost certainly the model answering instead of rewriting
# -- a confirmed failure mode worth guarding structurally, same
# reasoning as supervisor.py's own safety nets. Generous multiplier: a
# genuine rewrite adds a few words of resolved context, not paragraphs.
# The floor keeps this from over-triggering on a very short original
# ("that one?" -> ratio would otherwise reject almost any real rewrite).
_MAX_REWRITE_RATIO = 6
_MIN_REWRITE_FLOOR_CHARS = 200

# Small, closed set of referent/pronoun words this node's WHOLE JOB is
# to resolve away -- e.g. rewriting "which one is better?" into "which
# brush size is better?" is SUPPOSED to replace "one" with "brush size",
# not preserve "one" verbatim the way every other original word must be
# (see `_dropped_original_words` below). Deliberately narrow: only
# closed-class pronouns/demonstratives that stand in for a missing noun
# -- never a content word (a noun, verb, adjective, or number) that
# could itself carry information a rewrite must not lose.
_REFERENT_WORDS = frozenset({
    "it", "its", "this", "that", "these", "those", "one", "ones",
    "they", "them", "their", "he", "him", "his", "she", "her",
})

# Ordinary function words -- articles, a handful of the most common
# prepositions/conjunctions/auxiliaries/question-words -- excluded from
# the "every original word must survive" check below for a different
# reason than `_REFERENT_WORDS`: resolving "which one is best?" into
# "which brush size is best?" sometimes needs an article or preposition
# adjusted around the substitution, and none of these carry a referent a
# routing decision or a retrieval query could actually depend on.
# Deliberately SMALL and closed-class only -- unlike `_REFERENT_WORDS`,
# nothing here stands in for missing information, so excluding these is
# purely about not rejecting harmless grammatical smoothing around a
# substitution, never about giving up on a word that might matter.
_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "as", "by",
    "from", "with", "do", "does", "did", "what", "which", "who", "whom",
})


def _significant_words(text: str) -> set[str]:
    """`text`'s own content words -- lowercased, alphanumeric tokens,
    minus `_REFERENT_WORDS` and `_FUNCTION_WORDS` -- the vocabulary
    `_dropped_original_words` below checks for survival across a
    rewrite."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in _REFERENT_WORDS and w not in _FUNCTION_WORDS}


def _dropped_original_words(original: str, rewritten: str) -> list[str]:
    """
    Every content word from `original` that no longer appears ANYWHERE
    in `rewritten` -- the structural half of "loyal to the original
    wording": CONTEXTUALIZE_SYSTEM_PROMPT already instructs the model to
    change the absolute minimum needed, but a small local model
    paraphrasing "helpfully" instead of literally is a confirmed failure
    mode, not a hypothetical one -- "buy me some brushes" rewritten to a
    referent-resolved but verb-free "brushes for painting" is a fine
    English rewrite that silently drops the word product_search's own
    routing rule keys off of.

    Supersedes an earlier, narrower version of this same check that only
    watched a hand-curated list of routing-relevant words
    (SPECIALIST_DESCRIPTIONS' own cue words). A routing-relevant word is
    just one kind of content word, so protecting EVERY content word
    protects the routing-relevant ones too, without a second list that
    has to be remembered and kept in sync by hand every time a
    specialist is added or its cue words change -- exactly the kind of
    manual-sync drift that separately let RouteDecision's own Literal in
    supervisor.py silently miss `color_palette` for a while (see that
    module's docstring for the confirmed bug that exact failure pattern
    caused there).

    Pronouns/demonstratives (`_REFERENT_WORDS`) and ordinary function
    words (`_FUNCTION_WORDS`) are excluded from what must survive --
    replacing exactly the referent words is this node's entire job, and
    grammatical smoothing around that substitution is expected. Every
    OTHER word -- every noun, verb, adjective, and number the person
    actually used -- must appear in the rewrite unchanged, or the
    rewrite is rejected and the original is kept untouched instead (see
    contextualize_node's own call site).

    Deliberately exact-word matching, not stemmed/fuzzy -- "brush"
    silently becoming "brushes" during a rewrite is exactly the kind of
    small, easy-to-miss reword this check exists to catch, even though
    it means an occasional harmless pluralization also gets rejected.
    That is the accepted tradeoff of being stricter: a slightly higher
    false-reject rate on genuinely harmless rewrites, in exchange for a
    much lower false-accept rate on ones that quietly changed what the
    person actually said.
    """
    rewritten_words = _significant_words(rewritten)
    return [w for w in _significant_words(original) if w not in rewritten_words]


def _split_last_human(messages: list[BaseMessage]) -> tuple[Optional[int], list[BaseMessage]]:
    """
    Mirrors specialists.py's _last_human_text / supervisor.py's
    _current_turn_context scanning, but returns the INDEX of the latest
    HumanMessage (so the caller can rebuild a same-id replacement) rather
    than just its text, plus everything strictly BEFORE it -- the prior
    conversation this node exists to read from. Returns (None, []) if no
    HumanMessage exists at all, mirroring _last_human_text's own
    ValueError case one level up (the caller treats a missing index as
    "nothing to do," not as this node's problem to raise on).
    """
    last_human_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    if last_human_idx is None:
        return None, []
    return last_human_idx, messages[:last_human_idx]


def _format_transcript(prior: list[BaseMessage]) -> str:
    """
    Render the prior conversation as plain "Human: ..." / "Assistant
    (name): ..." lines, oldest-to-newest -- the same "who said what"
    shape supervisor.py's own _format_transcript uses for its routing
    prompt, just spanning earlier TURNS instead of this turn's specialist
    attempts. Each line truncated to 300 chars for the same reason
    _current_turn_context's own attempts preview is capped: enough to
    resolve a referent ("the brushes" / "that painting"), not enough to
    blow up the prompt on a verbose specialist answer. Non-string content
    (shouldn't occur -- every node in this graph only ever appends
    HumanMessage/AIMessage with string content, see state.py) is skipped
    rather than crashing this node.
    """
    lines = []
    for msg in prior[-_CONTEXT_WINDOW_MESSAGES:]:
        if not isinstance(msg.content, str):
            continue
        preview = msg.content[:300]
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {preview}")
        else:
            name = getattr(msg, "name", None) or "assistant"
            lines.append(f"Assistant ({name}): {preview}")
    return "\n".join(lines)


def build_contextualize_node() -> Specialist:
    """
    Build the contextualize node, closing over TWO chat model instances
    now, one per difficulty tier -- see specialists.py's own "Model
    routing by difficulty" section for the full rationale. Still built
    once per run here (same one-per-run granularity every other
    LLM-backed node in this project uses -- build_specialists()'s
    llm_large/llm_small, build_supervisor()'s llm_large/llm_small), not
    instantiated fresh per call. temperature=0 for the same
    reproducibility reason every other node here uses it.

    get_chat_model (agents/llm_provider.py) -- not a raw ChatOllama --
    so this node's follow-up rewrite now tries Groq's hosted free tier
    first and only falls back to the exact same local Ollama model on
    any Groq failure. See llm_provider.py's own module docstring.

    Difficulty is classified on the ORIGINAL follow-up text (`original.
    content`, below) -- most follow-ups this node ever sees ("which one
    is best?", "what about the cheaper one") are short referent-
    resolution rewrites well within the SMALL tier's ability; a follow-up
    long or compound enough to trip classify_request_difficulty's own
    heuristic escalates to the LARGE tier the same way a genuinely
    multi-hop question does in specialists.py.
    """
    llm_large = get_chat_model("large", node="contextualize")
    llm_small = get_chat_model("small", node="contextualize")

    def _llm_for(text: str):
        return llm_large if classify_request_difficulty(text) == "complex" else llm_small

    async def contextualize_node(state: AgentState) -> dict:
        messages = state["messages"]
        last_human_idx, prior = _split_last_human(messages)
        if last_human_idx is None:
            # No HumanMessage at all -- nothing for this node to do;
            # let the next node's own error handling (every specialist's
            # _last_human_text raises ValueError on this) surface the
            # real problem rather than raising here too.
            return {}

        original = messages[last_human_idx]
        if not prior or not isinstance(original.content, str):
            # First-ever turn (nothing to resolve against), or
            # non-text content this node isn't built to rewrite --
            # no-op, no LLM call spent on a turn that can't benefit.
            return {}

        if _message_carries_attachment(original.content):
            # Never rewrite a message carrying an `<attachment ...>`
            # marker -- that marker is a first-party, system-generated
            # signal (see specialists._message_carries_attachment's own
            # docstring) that supervisor.py's own deterministic routing
            # check depends on seeing VERBATIM to route straight to
            # `personal_docs`. A paraphrasing rewrite has every reason to
            # drop or reword something that looks like machine-readable
            # noise rather than meaningful prose -- the `_dropped_
            # original_words` check below would likely catch that and
            # reject the rewrite anyway in most cases, but "likely" isn't
            # good enough for a marker routing depends on: skip the LLM
            # call (and its cost/latency) entirely rather than gamble on
            # that check catching every phrasing the model might produce.
            return {}

        transcript = _format_transcript(prior)
        human_content = CONTEXTUALIZE_USER_TURN_TEMPLATE.format(
            transcript=transcript, question=original.content
        )
        response = await _llm_for(original.content).ainvoke(
            [SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT), HumanMessage(content=human_content)]
        )
        rewritten = (response.content or "").strip().strip('"')

        ceiling = max(_MIN_REWRITE_FLOOR_CHARS, len(original.content) * _MAX_REWRITE_RATIO)
        if not rewritten or len(rewritten) > ceiling:
            print(
                f"[contextualize] model output rejected (empty, or implausibly long for a "
                f"rewrite -- likely answered instead of rewrote) -- keeping original message "
                f"unchanged: {rewritten[:120]!r}",
                file=sys.stderr,
            )
            return {}

        if rewritten == original.content:
            # Model correctly judged the message already standalone --
            # no replacement needed, same "no-op when clean" return as
            # output_guard.py uses when nothing was flagged.
            return {}

        dropped = _dropped_original_words(original.content, rewritten)
        if dropped:
            # The rewrite may well be a perfectly fine English paraphrase
            # -- but it dropped or reworded something the person actually
            # said, which makes it a fidelity regression even when it's a
            # cosmetic improvement. Keeping the original here means this
            # turn's referent (e.g. "which size is best?") goes
            # unresolved for retrieval purposes, but that's a smaller
            # loss than silently changing what the person asked -- same
            # tradeoff the length-ratio check above already makes in
            # favor of "keep original, unchanged" whenever the rewrite
            # looks untrustworthy.
            print(
                f"[contextualize] rewrite dropped or reworded original wording "
                f"{dropped} -- keeping original message unchanged: {rewritten[:120]!r}",
                file=sys.stderr,
            )
            return {}

        print(
            f"[contextualize] rewrote follow-up for routing/retrieval: "
            f"{original.content!r} -> {rewritten!r}",
            file=sys.stderr,
        )
        return {"messages": [HumanMessage(content=rewritten, id=original.id)]}

    return contextualize_node

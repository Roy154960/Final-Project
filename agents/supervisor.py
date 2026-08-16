"""
Phase 3 supervisor.

Builds the one node that decides, on every visit, which specialist should
handle the current question next -- or whether the turn is done. The
supervisor never touches `retrieve`/`generate_answer` itself and never
produces the user-facing answer; it only ever emits a routing decision.
That split is enforced structurally here, not just by convention: the
supervisor's LLM is prompted for nothing but `{"route": "..."}`, so there
is no code path by which its output could stand in as an answer even if
the model tried to add prose around it (any prose breaks the strict JSON
parse below, which is treated as a malformed route, not as an answer).

Before any of that, ONE more check fires first, even before the two
early-stop checks below -- a deterministic net, informally "safety net
0", that can route the turn to `invoice` without ever calling the LLM at
all, not even once:

  - product_search/invoice disambiguation: CONFIRMED failure mode this
    exists to close, the same "same route regardless of a changing
    transcript" symptom this file's docstring already documents for
    llama3.2 (retrieval_qa, then painting_lookup, then color_palette --
    see `_finalize_with_first_attempt`'s own docstring for the
    painting_lookup case). product_search is exactly as vulnerable to
    it, in a way that's worse than a wasted specialist visit: a person
    naming an item they were just shown ("I'll take b3", "buy the fine
    detail brush set") reads, to a small local model, a lot like a NEW
    product mention rather than a reference back to something already
    found, so the model can route it to product_search instead of
    invoice. That doesn't just mis-route -- it re-runs a live internet
    search, which can return a differently-ordered or entirely different
    top-N result set than the batch the person is actually looking at,
    so even a LATER, correctly-routed invoice call can end up building a
    bill from the wrong batch. `_looks_like_invoice_followup`
    (agents/specialists.py) makes this call deterministically instead:
    an explicit "all of them"-style phrase, or an explicit purchase/
    invoice action word (buy/total/how much/...) combined with a real
    id/name match against the MOST RECENT product_search's own items
    (`_latest_product_search_batch` -- the exact same batch and matching
    function invoice_node itself uses to select items, so the router's
    "yes, this is an invoice follow-up" decision and invoice_node's own
    item selection can never disagree about whether a match exists).
    Fires once per turn, only while `invoice` hasn't already been tried
    THIS turn (so it never fights the repeat-route guard below). If it
    doesn't fire -- no prior product_search this conversation, or the
    message doesn't clearly qualify -- the turn proceeds exactly as
    before, entirely up to the model's own routing judgment.

Below that, TWO early-stop checks can FINISH the turn without even
calling the LLM a second time, the moment exactly one specialist has
answered and that answer doesn't look like a refusal:

  - Always on: for a specialist that structurally cannot produce a
    hedging "partial or unclear" answer in the first place (see
    `_DETERMINISTIC_NEVER_HEDGES`'s own comment for exactly which ones
    and why), skipping the second look costs nothing, so this one is
    unconditional.
  - Opt-in (`skip_reroute_if_answered`, default OFF -- see
    `DEFAULT_SKIP_REROUTE_IF_ANSWERED`'s own docstring for why): the
    same idea extended to every OTHER specialist too, including ones
    that genuinely can hedge (retrieval_qa, multi_hop, painting_lookup).
    Off by default because this project's own test suite encodes a
    scenario where a hedging first answer specifically SHOULD get a
    second routing decision rather than being treated as final.

Both are a distinct concern from the four nets below: those four are
about validating/correcting what the model outputs when it IS asked to
route; the early-stop nets are about not asking it to route again at all
when the transcript already makes the answer obvious. They exist because
a repeat-route-guard walk (net 3) is a *correct* response to a stuck
model, but still means every specialist it walks through -- including
ones with nothing relevant to say -- gets appended to the transcript
before the turn ends, and (CONFIRMED live-run failure, not
hypothetical) if that redirect happens to land on a specialist that
genuinely finds something to say about a DIFFERENT aspect of the
question, its later answer silently becomes "the" answer api.py shows
(ChatResponse.answer is always the last specialist-named message),
burying a first answer that was already correct and complete. This is
exactly the "found the right answer with the right tool first, then
kept replying with other tools that don't help" pattern both nets exist
to cut off before it starts.

Four independent safety nets sit between "the model said something" and
"the graph routes there":

  1. Schema-level: the raw JSON is validated against a RouteDecision
     model built fresh in build_supervisor() (see
     _build_route_decision_model's own docstring), whose `route` field
     is a Literal built directly from THIS build's own known_routes. A
     value outside that Literal fails Pydantic validation before it's
     ever inspected further.
  2. Explicit membership check: the validated value is *also* checked
     against `known_routes` directly. Now provably redundant with (1) in
     every case, since both are built from the identical known_routes
     value inside the same closure -- kept anyway as cheap defense-in-
     depth against a future regression, not because it currently catches
     anything (1) doesn't. It used to exist for a real drift scenario --
     a hand-maintained Literal silently falling out of sync with the
     live specialists dict -- that a confirmed bug showed was possible
     (color_palette added to specialists.py without the old hardcoded
     Literal being updated to match, so it could never be chosen at all
     under the default schema-constrained format); building the schema
     from known_routes itself is what closed that gap, not this check.
  3. Repeat-route guard: even a schema-valid, known route is rejected if
     it names a specialist that has *already answered this turn* (per
     `_current_turn_context`'s `attempts` list). This exists because a
     confirmed live-run failure showed the first two nets aren't enough
     on their own: SUPERVISOR_SYSTEM_PROMPT already tells the model
     never to re-route to a specialist that already answered, and a
     small local model (llama3.2) was observed ignoring that instruction
     and re-picking the same already-answered specialist on every
     subsequent visit, running the iteration cap all the way out on a
     single-topic question that retrieval_qa most likely already
     answered correctly on its first call. That's a prompt failure the
     first two nets structurally cannot catch (the route was valid and
     known) -- so this net enforces the rule in code instead of trusting
     the model to follow it, the same "structural guardrail over prompt
     wording" fix already applied once in specialists.py's
     _extract_grounded_answer.
  4. Premature-FINISH guard: a schema-valid, known "FINISH" is rejected
     if `attempts` is still EMPTY -- i.e. the model is trying to end the
     turn before a single specialist has ever answered. Confirmed live in
     a Phase 5 eval run: for an out-of-scope query ("What's a good
     recipe for chocolate chip cookies?"), the supervisor's very first
     raw output was `{"route": "FINISH"}`, which nets 1 and 2 both
     accepted (FINISH is a valid, known route) and net 3 has no opinion
     on (it only fires on non-FINISH repeats). With no specialist ever
     invoked, no AIMessage was ever appended, and the turn ended with
     literally no answer for the user to read -- silently, not as an
     error. SUPERVISOR_SYSTEM_PROMPT already tells the model FINISH means
     "a specialist has answered and its answer is sufficient," which is
     exactly the instruction this failure ignores -- another prompt
     failure the first three nets structurally cannot catch, fixed the
     same way net 3 was: enforce the rule in code. See
     test_supervisor_smoke.py's "premature FINISH before any specialist
     has answered is overridden" test for the direct regression case.

Any of the four failing -- unparsable JSON, a ValidationError, a name
that fails the membership check, a name that's already been tried this
turn, or FINISH before anything has been tried at all -- routes to the
next untried specialist (deterministic order: the order `specialists`
was built in), or to `fallback_route` if nothing untried remains and
re-routing is even possible, or to a forced `FINISH` if every specialist
has already been tried. None of these paths ever raises.

The iteration cap is enforced here too, once per supervisor visit, before
any LLM call is made for that visit -- see `DEFAULT_ITERATION_CAP`'s
docstring for the reasoning behind the chosen number.
"""

import json
import sys
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, create_model

from agents.llm_provider import get_chat_model
from agents.prompts import (
    SPECIALIST_DESCRIPTIONS,
    SPECIALIST_ROUTING_EXAMPLES,
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_USER_TURN_TEMPLATE,
)
from agents.specialists import (
    Specialist,
    classify_request_difficulty,
    _latest_product_search_batch,
    _looks_like_invoice_followup,
    _message_carries_attachment,
)
from agents.state import AgentState

# One supervisor visit = one routing decision (an initial pick, a
# re-route, or a FINISH). Originally chosen at 4 for the Phase 3 spec's
# own reasoning with three specialists: one initial pick, up to two
# re-routes, one buffer call to land on FINISH.
#
# Raised to 8 when four more specialists (image_qa, painting_lookup,
# product_search, invoice) were added on top of the original three, then
# to 9 when an eighth (color_palette) was added on top of THAT. The
# reasoning is the SAME formula each time, just against a bigger
# known_routes set: safety net 3 (the repeat-route guard, see this
# module's own docstring) can, in the worst case, walk every untried
# specialist once before forcing FINISH -- with eight specialists now,
# that worst case alone is eight visits, plus one buffer call to land on
# FINISH normally = 9. This is a mechanical consequence of adding more
# routes, not a new judgment call: the fallback logic below
# (_next_untried_route) already tries every specialist that hasn't
# answered yet before giving up, so the cap has to be large enough to
# let that fallback actually reach all of them in a genuinely worst-case
# run, or a legitimate "try the next untried specialist" fallback would
# get cut off by the cap before ever reaching, say, the 8th of 8
# specialists. Tune this after seeing a real eval table on your own
# corpus -- if most turns finish in 1-2 visits (the common case for a
# correctly-routed question), a high cap costs nothing on those turns;
# it only matters on the rarer turn that's actually struggling.
DEFAULT_ITERATION_CAP = 9

# Never defaults to FINISH: a forced FINISH before any specialist has run
# would silently return a non-answer. retrieval_qa is the safest fallback
# specifically because it's grounded (it must retrieve before it can
# answer, and says so plainly when retrieval comes up empty) rather than
# a free-text guess.
DEFAULT_FALLBACK_ROUTE = "retrieval_qa"

# How the supervisor's Ollama call is asked to produce JSON:
#   "json_schema" -- format=RouteDecision.model_json_schema(). Ollama's
#       structured-output mode, constrained to RouteDecision's exact
#       shape and Literal enum. The default.
#   "json"        -- format="json". Ollama's looser JSON mode (any valid
#       JSON, no schema/enum constraint) -- the same "ask for bare JSON,
#       validate after the fact" approach specialists.py's multi-hop
#       decompose step already uses successfully.
# Exists as a build-time toggle, not a hardcoded choice, because of a
# confirmed live-run observation worth testing directly rather than just
# asserting: with "json_schema", a live supervisor returned the exact
# same route ("retrieval_qa") on every one of four consecutive visits,
# even though the transcript fed to it changed each time (a growing list
# of already-tried specialists) -- i.e. the decision showed no visible
# sensitivity to its own prompt. The repeat-route guard (below) makes
# that safe rather than catastrophic, but doesn't explain it. Since
# multi_hop's un-schema-constrained JSON call *does* visibly condition on
# its input (its decomposition varies with the actual question), it's
# worth A/B testing whether RouteDecision's schema constraint itself is
# what's suppressing llama3.2's context-sensitivity here, versus this
# just being a small model failing to follow the FINISH criterion either
# way. Swap DEFAULT_ROUTE_FORMAT (or pass route_format="json" to
# build_supervisor()) and compare the "[supervisor] raw model output"
# lines across repeated live runs of the same question to find out.
DEFAULT_ROUTE_FORMAT: Literal["json_schema", "json"] = "json_schema"

# Substring markers drawn directly from this project's own hardcoded
# "I can't help with this" specialist replies (see specialists.py's
# product_search_node, invoice_node, corpus_meta_node, and
# color_palette_node -- the last one's markers actually come from
# mcp_server/color_tools.py's generate_palette() error strings, since
# that specialist's own refusal text is entirely pass-through from the
# tool). Used ONLY by `_looks_like_refusal` below, which itself is used
# ONLY by the early-stop net in `supervisor_node` -- not safety-critical,
# so a marker going stale just costs one extra, otherwise-harmless
# supervisor visit rather than breaking anything. Keep this list in sync
# with the actual refusal wording if that wording changes -- a
# maintenance burden this module now deliberately keeps only where
# staleness is cheap (a missed early-stop opportunity); see
# _build_route_decision_model's own docstring for the different,
# NOT-cheap version of this same "hand-maintained list drifts" problem
# that function exists to close structurally instead.
_REFUSAL_MARKERS = (
    "i don't have access",
    "not a specialist",
    "couldn't find any",
    "couldn't tell which",
    "nothing to invoice",
    "isn't available right now",
    "i don't see any product search",
    "i don't have a working internet-search tool",
    "couldn't recognize",
    "couldn't connect",
    "i need either a color",
)

# Specialists whose answer is ALWAYS either a complete, confident result
# or an explicit refusal recognized by `_REFUSAL_MARKERS` -- NEVER a
# hedging "partial or unclear" middle ground -- because none of them
# generate open-ended, free-form prose from an LLM's own judgment about
# whether it found enough to say something:
#   - image_qa: zero LLM calls at all (see that node's own comment) --
#     either formats real retrieved images, or returns the one fixed
#     IMAGE_QA_NO_RESULTS_MESSAGE constant.
#   - product_search / invoice: any LLM use is explicitly CONSTRAINED to
#     describing/tagging real tool-returned data (see product_search_node's
#     own comment: "the LLM's single call is confined to writing
#     tier-scoped comparison paragraphs"), never to judging whether the
#     answer itself is complete -- that's decided entirely by whether the
#     underlying tool call found real data.
#   - color_palette: zero LLM calls (see that node's own comment) -- pure
#     deterministic color math, either a real palette or one of
#     color_tools.py's own fixed error strings.
#
# painting_lookup is deliberately NOT included, even though it also
# calls a deterministic tool first: its FINAL step is a real, free-form
# LLM synthesis call (PAINTING_LOOKUP_SYNTHESIZE_PROMPT_TEMPLATE) that
# CAN produce a thin or hedging answer when both its corpus and web
# sources come up empty, the same way retrieval_qa/multi_hop can -- so
# it keeps getting the normal, model-driven second look, same as those.
#
# Used to let the always-on early-stop net below skip a second
# supervisor visit specifically for these specialists, regardless of
# `skip_reroute_if_answered` -- see that net's own comment for the
# CONFIRMED live-run failure (color_palette's correct first answer
# silently buried under a later, off-topic retrieval_qa answer) this
# closes.
_DETERMINISTIC_NEVER_HEDGES = frozenset({"image_qa", "product_search", "invoice", "color_palette"})

# Whether `supervisor_node` skips its second-visit LLM call entirely and
# FINISHes as soon as exactly one specialist has answered and that
# answer doesn't look like a refusal (`_looks_like_refusal`). Exposed as
# a build-time toggle (like `route_format`) rather than hardcoded, for
# the same reason: it changes observable routing behavior, so a person
# tuning this project against their own model should be able to turn it
# off and compare rather than have it silently baked in.
#
# Why this exists: with `route_format="json_schema"`'s own documented
# "same route regardless of a changing transcript" failure mode (see
# that constant's docstring), letting the model make every re-route
# decision means a question the FIRST specialist already answered
# correctly can still visit several more specialists before the
# repeat-route guard (safety net 3, below) or the iteration cap forces a
# stop -- each of those extra specialists' own near-universal refusal
# message (e.g. invoice's "nothing to invoice yet" on a question that
# was never about invoicing) gets appended to the transcript along the
# way, which is exactly the "found the right answer first, then kept
# replying with other tools that don't help" pattern this net exists to
# cut off BEFORE it starts, rather than clean up after the fact the way
# `_finalize_with_first_attempt` already does for the forced-FINISH
# paths. Same "structural guardrail over prompt wording" preference this
# project applies everywhere else (corpus_meta's missing tools,
# _extract_grounded_answer, the four safety nets below) -- a fast,
# deterministic check that doesn't depend on the model reliably knowing
# when to say FINISH, since that's precisely the judgment call
# `DEFAULT_ROUTE_FORMAT`'s docstring already shows this model doesn't
# reliably make.
#
# Deliberately narrow: only fires on exactly ONE prior attempt (the
# second supervisor visit). A question that genuinely needs more than
# one specialist (multi_hop calling out to retrieval_qa internally
# doesn't count -- that's inside one specialist's own turn, not a
# supervisor re-route) still gets the model's own routing judgment from
# the third visit onward, since this net only ever fires once, right
# after the very first specialist answers.
#
# Defaults to OFF (False), NOT on, despite the rest of this project's own
# "structural guardrail over prompt wording" preference -- and that's a
# deliberate, stated exception, not an oversight. `_looks_like_refusal`
# can only recognize a HARD refusal (one of the fixed marker strings);
# it cannot tell a confident answer apart from a genuinely partial or
# unclear one, and this project's OWN existing test suite already
# encodes a scenario where that distinction matters:
# test_repeat_route_guard_redirects_to_untried_specialist (see
# test_supervisor_smoke.py) has retrieval_qa answer with "Partial or
# unclear answer." -- not a refusal by `_REFUSAL_MARKERS`' definition --
# and specifically expects the model to get a chance to notice that and
# re-route, which is exactly the visit this net would otherwise skip.
# test_transcript_lives_in_human_turn_not_system_prompt similarly expects
# a real second LLM call to happen and inspects its message list. Turning
# this on by default would silently break both without a test failure
# anywhere near the actual change (the tests would just never reach the
# code path they're checking). Flip this to True only after deciding
# those two scenarios' tradeoff is acceptable for your own corpus/model
# -- e.g. by updating those two tests' fixture content to something
# `_looks_like_refusal` would actually catch, or by widening
# `_REFUSAL_MARKERS` to also catch hedge-y "partial/unclear" language if
# your own live runs show that's a distinguishable pattern for your
# model's actual wording.
DEFAULT_SKIP_REROUTE_IF_ANSWERED = False


def _build_route_decision_model(known_routes: frozenset[str]) -> type[BaseModel]:
    """
    Build a fresh, `RouteDecision`-shaped Pydantic model whose `route`
    field is a `Literal` over EXACTLY `known_routes` -- constructed here,
    at `build_supervisor()` time, from the live specialists dict's own
    keys (plus "FINISH"), never hand-typed and left to drift.

    CONFIRMED BUG this replaces: `RouteDecision` used to be a single,
    hardcoded module-level class whose `Literal` was manually kept in
    sync with `specialists.py`'s specialist set by hand. When
    `color_palette` was added as an eighth specialist, that Literal was
    NOT updated to match -- and because `route_format="json_schema"`
    (the default; see `DEFAULT_ROUTE_FORMAT`'s docstring) asks Ollama's
    own structured-output support to constrain decoding to
    `RouteDecision.model_json_schema()`, the model could never generate
    `{"route": "color_palette"}` AT ALL, no matter how obviously a
    question called for it -- confirmed directly:
    `RouteDecision.model_validate_json('{"route": "color_palette"}')`
    raised a `ValidationError` even though `color_palette` was a real,
    working, fully wired specialist the whole time. The ONLY way that
    route was ever reached was via safety net 3's fallback walk
    (`_next_untried_route`), after every other specialist had already
    been tried and failed -- never as the model's own first, obviously
    correct choice. Safety net 2 (the membership check right below, in
    `supervisor_node`) is explicitly documented as a backstop against
    exactly this kind of specialist-list drift -- but it only runs AFTER
    net 1's Pydantic validation already succeeded, so a route missing
    from net 1's OWN schema never even reached net 2 to be caught. This
    bug could not have been caught by net 2 no matter how correct net 2
    itself was.

    The fix: stop hand-maintaining a second, separately-typed list of
    route names anywhere. `known_routes` (built in `build_supervisor()`
    from `specialists.keys()`, the exact same source
    `specialist_descriptions` and `ordered_specialist_names` already use)
    is now the SINGLE source of truth for the schema too -- every call to
    `build_supervisor()` builds its own model fresh, so the set of routes
    Ollama is ever asked to choose from is, by construction, always
    exactly the set of specialists this particular graph was actually
    built with. This makes the specific bug above structurally
    impossible to reintroduce, the same "structural guardrail over
    prompt wording" preference this project applies everywhere else --
    here applied to schema CONSTRUCTION itself, not just to validating
    what came out of it.

    A secondary, now-provable consequence: safety net 2's own membership
    check (`raw_route in known_routes`) can never disagree with net 1
    anymore, since both are built from the identical `known_routes` value
    inside the same `build_supervisor()` closure -- net 2 is kept anyway,
    purely as cheap defense-in-depth against a FUTURE regression back
    toward a hand-decoupled schema, not because it currently catches
    anything net 1 doesn't. See its own comment at the call site.
    """
    return create_model(
        "RouteDecision",
        route=(
            Literal[tuple(sorted(known_routes))],
            Field(..., description="Which specialist should handle this question next, or FINISH."),
        ),
    )


def _current_turn_context(state: AgentState) -> tuple[str, list[tuple[str, str]]]:
    """
    Split state["messages"] into "the question this turn is about" and
    "which specialists have already answered it, and roughly what they
    said" -- without adding any new field to AgentState (route and
    iteration_count already cover everything else Phase 3 needs; see
    state.py's docstring on why that file shouldn't need touching again).

    Scoped to only the messages *after* the latest HumanMessage, not the
    whole transcript -- so a follow-up question in a longer conversation
    doesn't drag a previous turn's specialist attempts into this turn's
    routing decision.
    """
    messages = state["messages"]
    last_human_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    if last_human_idx is None:
        raise ValueError("No HumanMessage found in state['messages']")

    question = messages[last_human_idx].content
    attempts = [
        (msg.name, msg.content[:300])
        for msg in messages[last_human_idx + 1 :]
        if isinstance(msg, AIMessage) and getattr(msg, "name", None)
    ]
    return question, attempts


def _format_transcript(attempts: list[tuple[str, str]]) -> str:
    """
    Render `attempts` as the plain-text block SUPERVISOR_USER_TURN_TEMPLATE's
    {transcript} slot expects -- just the "who's already answered, and
    roughly what they said" summary, not the question itself (the
    template carries `question` in its own separate slot, so repeating it
    here would just duplicate it in the human turn). Kept as its own
    function so the "no specialist has answered yet" case is one
    explicit string, not an empty list rendered as blank text that could
    read as a template bug rather than a real first-turn state.
    """
    if not attempts:
        return "No specialist has answered yet."
    return "\n".join(f'- {name} already answered: "{preview}"' for name, preview in attempts)


def _first_attempt_message(state: AgentState) -> Optional[tuple[str, str]]:
    """
    (name, FULL content) of the FIRST specialist to answer this turn --
    deliberately NOT `_current_turn_context`'s `attempts[0]`, which only
    keeps a 300-char preview (enough for the routing prompt's transcript,
    not enough to re-surface as an actual answer).

    Exists for `_finalize_with_first_attempt` below -- see that
    function's docstring for why the FIRST attempt, specifically, is
    what gets reaffirmed as the final answer when the supervisor has to
    force a FINISH without a clean model-driven one.
    """
    messages = state["messages"]
    last_human_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    if last_human_idx is None:
        return None
    for msg in messages[last_human_idx + 1 :]:
        if isinstance(msg, AIMessage) and getattr(msg, "name", None):
            return msg.name, msg.content
    return None


def _finalize_with_first_attempt(note_text: str, first_attempt: Optional[tuple[str, str]]) -> AIMessage:
    """
    Builds the single AIMessage a forced-FINISH path (iteration cap
    reached, or every specialist exhausted) appends -- and this is the
    fix for a confirmed live-run problem, not just cosmetic wording.

    CONFIRMED LIVE-RUN FAILURE this exists to fix: with seven specialists
    now in play, a live run of `python -m agents.graph "Tell me about the
    Mona Lisa"` showed the supervisor's raw model output stuck on
    `{"route": "painting_lookup"}` on every single visit -- the exact
    "same route regardless of a changing transcript" symptom
    DEFAULT_ROUTE_FORMAT's docstring already documents for llama3.2, now
    confirmed to generalize beyond `retrieval_qa`. The repeat-route guard
    did exactly its job and walked every OTHER untried specialist in
    order (retrieval_qa, corpus_meta, multi_hop, image_qa, product_search,
    invoice) before force-FINISHing -- but `invoice` is last in
    `specialists.py`'s build order (see that file's own comment on why
    it's been moved earlier since), so the LAST specialist-named message
    in the transcript ended up being invoice's near-universal
    "nothing to invoice yet" refusal. Both `agent_mcp_server.py`'s
    `_summarize()` and `eval_phase5.py`'s `_extract_route_info()` pick
    "the last specialist message" as *the* answer (by design -- see
    their own docstrings on skipping a trailing supervisor meta-note) --
    which meant those consumers would have shown a wrong, unrelated
    refusal as the answer to a question that had already been answered
    correctly on the very first routing decision.

    The fix: reaffirm the FIRST specialist to answer this turn as the
    final answer, by appending ONE new message carrying that specialist's
    own name (not "supervisor") and its FULL original content, with
    `note_text` prepended for transparency about why this path fired.
    Because it carries the first specialist's real name, both consumers'
    existing "last specialist message" logic picks it up correctly with
    ZERO changes needed on their end -- the fix lives entirely in the
    data shape supervisor.py produces, the same "structural guardrail"
    preference this project applies everywhere else.

    Deliberately APPENDS rather than reordering or overwriting the
    earlier occurrence of this same content -- state.py's own docstring
    on `messages` is explicit that the transcript stays an honest,
    append-only record of what actually happened; a reader can still see
    exactly how many specialists were walked and in what order, with
    this final message making unambiguous which one is being treated as
    the real answer.

    Known, stated tradeoff (worth saying plainly rather than glossing
    over): this always trusts the FIRST attempt, even in a hypothetical
    case where the model's later re-routes were genuinely well-reasoned
    handoffs rather than repeat-route-guard overrides (e.g. corpus_meta
    correctly punting to retrieval_qa). In practice, on every confirmed
    live run so far, the first pick was the correct one and the walk
    past it was driven by the model repeating itself, not by genuine
    reasoning -- but a more precise fix would specifically distinguish
    "walked because of a repeat-route override" from "walked because the
    model kept choosing new routes on its own," which this does not do.
    Worth measuring in a real Phase 5 eval run before assuming this is
    the final word on it.

    Falls back to a bare `AIMessage(name="supervisor", content=note_text)`
    if there's no first attempt to reaffirm (only possible if the cap was
    hit before any specialist ever ran) -- there's nothing to restore in
    that case.
    """
    if first_attempt is None:
        return AIMessage(content=note_text, name="supervisor")
    first_name, first_content = first_attempt
    return AIMessage(content=f"{note_text}\n\n{first_content}", name=first_name)


def _partial_answer_note(iteration_cap: int, attempts: list[tuple[str, str]]) -> str:
    """
    The graceful, non-raising note produced when the iteration cap is
    hit. Combined with the first specialist's full original answer by
    `_finalize_with_first_attempt` (see that function's docstring for
    why the FIRST attempt, not "whatever's most recent," is what gets
    reaffirmed) into the single message the transcript's eval table and
    any downstream consumer will see as the actual answer.
    """
    if attempts:
        first_name = attempts[0][0]
        tried = ", ".join(name for name, _ in attempts)
        return (
            f"[Partial answer -- iteration cap ({iteration_cap}) reached after "
            f"trying: {tried}. Reaffirming {first_name}'s answer (the first "
            "specialist to answer this turn) as final, below -- later "
            "re-routes past this point are not a reliable signal of a "
            "better answer; see supervisor.py's _finalize_with_first_attempt "
            "docstring.]"
        )
    return (
        f"[Iteration cap ({iteration_cap}) reached before any specialist could "
        "answer. Unable to produce a grounded answer for this question within "
        "the iteration limit.]"
    )


def _next_untried_route(ordered_names: list[str], tried_names: set[str]) -> Optional[str]:
    """
    First specialist name in `ordered_names` (the order `specialists` was
    built in -- retrieval_qa, corpus_meta, multi_hop, per
    build_specialists()'s dict) that isn't in `tried_names` yet. Returns
    None if every specialist has already been tried this turn.

    Used both when the model's raw route fails schema/membership
    validation and when it names an already-tried specialist (the
    repeat-route guard) -- in both cases, "try something not yet tried"
    is a better fallback than blindly retrying `fallback_route`, which
    may itself already be the specialist that was just repeated.
    """
    for name in ordered_names:
        if name not in tried_names:
            return name
    return None


def _all_tried_note(attempts: list[tuple[str, str]]) -> str:
    """
    Note produced when every known specialist has already answered this
    turn and the model still didn't say FINISH (or repeated one of them
    again) -- forced to FINISH here, in code, rather than looping the
    model against a fully-exhausted specialist list until the iteration
    cap eventually catches it anyway. Combined with the first
    specialist's full original answer by `_finalize_with_first_attempt`
    (see that function's docstring for the confirmed live-run failure
    this fixes and why the FIRST attempt, not "whatever's most recent,"
    is what gets reaffirmed).
    """
    first_name = attempts[0][0]
    tried = ", ".join(name for name, _ in attempts)
    return (
        f"[All specialists already tried this turn ({tried}) without the "
        f"supervisor confirming FINISH. Reaffirming {first_name}'s answer "
        "(the first specialist to answer this turn) as final, below.]"
    )
def _looks_like_refusal(content: str) -> bool:
    """
    Best-effort, deliberately narrow check for whether a specialist's
    answer is one of this project's own hardcoded "I can't help with
    this" messages (see `_REFUSAL_MARKERS`'s own comment for where these
    strings come from) rather than a genuine answer.

    Substring match on lowercased content, not real NLU -- see
    `_REFUSAL_MARKERS`'s docstring for why that's an acceptable tradeoff
    here specifically (this check is not safety-critical; see
    `DEFAULT_SKIP_REROUTE_IF_ANSWERED`'s docstring for what it gates).
    """
    lowered = content.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def build_supervisor(
    specialists: dict[str, Specialist],
    iteration_cap: int = DEFAULT_ITERATION_CAP,
    fallback_route: str = DEFAULT_FALLBACK_ROUTE,
    route_format: Literal["json_schema", "json"] = DEFAULT_ROUTE_FORMAT,
    skip_reroute_if_answered: bool = DEFAULT_SKIP_REROUTE_IF_ANSWERED,
) -> Specialist:
    """
    Build the supervisor node function, closing over the known route
    names taken from `specialists`'s own keys -- the same dict
    build_specialists() (agents/specialists.py) returns -- so this
    dict's keys are the single source of truth for "known agent names"
    the spec asks the routing decision be checked against, rather than a
    second, separately-maintained list that could drift out of sync.

    Raises ValueError at build time (not at request time, mid-graph) if
    `fallback_route` isn't itself one of `specialists`'s keys -- a
    fallback that isn't a valid route would defeat the whole point of
    having one.

    `route_format` -- see DEFAULT_ROUTE_FORMAT's docstring for why this
    is exposed as a parameter rather than hardcoded: it's the A/B knob
    for testing whether Ollama's schema-constrained structured output is
    itself contributing to the confirmed live-run repeat-routing failure.
    """
    if fallback_route not in specialists:
        raise ValueError(
            f"fallback_route {fallback_route!r} is not among the built "
            f"specialists {sorted(specialists.keys())} -- the fallback must "
            "itself be a valid, callable route."
        )

    known_routes = frozenset(specialists.keys()) | {"FINISH"}
    ordered_specialist_names = list(specialists.keys())

    # See _build_route_decision_model's own docstring for the confirmed
    # bug this fixes: a hand-maintained Literal that silently fell out of
    # sync with the live specialists dict. Built fresh here, from
    # `known_routes` -- the exact same source specialist_descriptions and
    # ordered_specialist_names above already use -- so drift between
    # "what this build's schema allows" and "what this build's dict
    # actually contains" is now structurally impossible.
    route_decision_model = _build_route_decision_model(known_routes)

    specialist_descriptions = "\n".join(
        f"- {name}: {SPECIALIST_DESCRIPTIONS.get(name, '(no description registered)')}"
        for name in specialists
    )
    route_names = ", ".join(sorted(known_routes))
    # Only the worked examples for specialists THIS build actually has --
    # same "only ever describe what's really in the specialists dict"
    # rule specialist_descriptions above already follows, so a reduced or
    # test-only specialist set never shows the model an example pointing
    # at a route that doesn't exist in this build (which route_names /
    # the schema itself would then reject anyway). Small local models
    # follow a handful of concrete labeled examples far more reliably
    # than abstract prose rules alone -- several of these are chosen to
    # make one of the "Specific routing distinctions" bullets above
    # literal rather than abstract (product_search's example is the
    # EXACT "what's a good brush for glazing" case that bullet already
    # calls out by name, so the rule and its worked example match).
    routing_examples = "\n".join(
        f"- {SPECIALIST_ROUTING_EXAMPLES[name]}" for name in specialists if name in SPECIALIST_ROUTING_EXAMPLES
    ) or "(no worked examples registered for this build's specialists)"
    # Fully static now -- built once here, not reformatted per call. The
    # per-turn-state content (the transcript) no longer lives in this
    # prompt at all; see SUPERVISOR_USER_TURN_TEMPLATE's docstring in
    # prompts.py for why it moved to the human turn instead.
    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        specialist_descriptions=specialist_descriptions,
        route_names=route_names,
        routing_examples=routing_examples,
    )

    # format=<json schema> (route_format="json_schema", the default) asks
    # Ollama's own structured-output support to constrain decoding to
    # route_decision_model's shape -- see _build_route_decision_model's
    # own docstring for why this now being built fresh per call, from
    # known_routes, is itself the fix for a confirmed bug. format="json"
    # (route_format="json") is the looser alternative -- see
    # DEFAULT_ROUTE_FORMAT's docstring for why this is worth A/B testing
    # rather than assuming the schema-constrained path is strictly
    # better. temperature=0 either way, for the same reproducibility
    # reason every other node in this project uses it: the eval table's
    # "actual route" column should be stable across repeated runs of the
    # same query.
    ollama_format = route_decision_model.model_json_schema() if route_format == "json_schema" else "json"

    # Two instances now, one per difficulty tier -- see specialists.py's
    # own "Model routing by difficulty" section for the full rationale
    # (the supervisor is already a router; this gives it a model choice
    # to make alongside its agent choice, same as the class's own
    # "Model Routing" guidance suggests). Both built once here, at the
    # same one-per-run granularity this node already used for its single
    # `llm` before this change -- picked between PER VISIT by
    # `_llm_for()` below, never rebuilt mid-run.
    #
    # get_chat_model (agents/llm_provider.py) -- not a raw ChatOllama --
    # so this node's routing decision now tries Groq's hosted free tier
    # FIRST and only falls back to the exact same local Ollama model/
    # `ollama_format` structured-decoding behavior this project already
    # had, on any Groq failure. See llm_provider.py's own module
    # docstring for exactly what changes (and what deliberately doesn't)
    # about `ollama_format` handling on the Groq branch.
    llm_large = get_chat_model("large", node="supervisor", ollama_format=ollama_format)
    llm_small = get_chat_model("small", node="supervisor", ollama_format=ollama_format)

    def _llm_for(text: str):
        """Pick the pre-built chat model for this visit's question --
        see classify_request_difficulty's own docstring (specialists.py)
        for the heuristic. Never builds a new model per call; only ever
        returns one of the two instances already built above."""
        return llm_large if classify_request_difficulty(text) == "complex" else llm_small

    async def supervisor_node(state: AgentState) -> dict:
        iteration_count = state.get("iteration_count", 0) + 1

        if iteration_count > iteration_cap:
            # Cap already exceeded -- no LLM call for this visit at all,
            # just stop gracefully with whatever's already in state. See
            # _finalize_with_first_attempt's docstring for why this is a
            # new appended message (carrying the FIRST specialist's own
            # name and full content, not a bare "supervisor" note) rather
            # than an edit to an existing one.
            _, attempts = _current_turn_context(state)
            note = _partial_answer_note(iteration_cap, attempts)
            final_message = _finalize_with_first_attempt(note, _first_attempt_message(state))
            print(
                f"[supervisor] iteration cap ({iteration_cap}) reached -- forcing FINISH",
                file=sys.stderr,
            )
            return {
                "route": "FINISH",
                "iteration_count": iteration_count,
                "messages": [final_message],
            }

        question, attempts = _current_turn_context(state)

        # --- Safety net 0 (always on, pre-LLM): deterministic
        # product_search/invoice disambiguation -- see this module's own
        # docstring for the confirmed failure mode this exists to head
        # off, and _looks_like_invoice_followup's own docstring
        # (agents/specialists.py) for exactly what has to be true for
        # this to fire. Gated on "invoice" not already tried this turn
        # so it can never fight safety net 3 (the repeat-route guard)
        # below by re-forcing a route that's already been tried and
        # (per that guard's whole point) shouldn't be tried again.
        tried_names_so_far = {name for name, _ in attempts}
        if "invoice" in known_routes and "invoice" not in tried_names_so_far:
            latest_batch = _latest_product_search_batch(state["messages"])
            if _looks_like_invoice_followup(str(question), latest_batch):
                print(
                    "[supervisor] safety net 0: message reads as an invoice "
                    "follow-up to the most recent product_search batch -- "
                    "routing straight to invoice without asking the model to "
                    "choose.",
                    file=sys.stderr,
                )
                return {"route": "invoice", "iteration_count": iteration_count, "messages": []}

        # --- Safety net 0a (always on, pre-LLM): deterministic
        # attachment-follow-up routing straight to `personal_docs`.
        #
        # CONFIRMED live-run failure this closes: a message carrying an
        # `<attachment ...>` marker -- i.e. one that just attached an
        # image, PDF, or text file, per `_message_carries_attachment`'s
        # own docstring (agents/specialists.py) -- is about as
        # unambiguous a "route to personal_docs" signal as this graph
        # ever sees, yet a live run showed the model send "Explain this
        # uploaded image titled ..." straight to `retrieval_qa` instead
        # (which has no access to anything personally uploaded, and
        # answered from the main corpus alone -- unrelated to the
        # image). The repeat-route guard then walked every OTHER
        # specialist before FINISHing, none of them any more able to see
        # the upload than the first one was. Same root cause as safety
        # net 0 above (a small local model's routing judgment isn't
        # reliable enough to trust alone for a case this structurally
        # certain), same fix shape: skip the judgment call entirely
        # rather than hope the prompt wording eventually gets there.
        #
        # Unlike safety net 0, this needs no separate intent-word gate --
        # an attachment marker is a first-party, system-generated token
        # (attachments.ts's own send() builds it), never something a
        # person could coincidentally type themselves the way product
        # names/ids can overlap with an unrelated question. Depends on
        # contextualize.py never rewriting a message carrying this marker
        # (see that module's own check) so the marker's exact text is
        # guaranteed to still be in `question` by the time this runs,
        # regardless of what turn this is within the conversation.
        if "personal_docs" in known_routes and "personal_docs" not in tried_names_so_far:
            if _message_carries_attachment(str(question)):
                print(
                    "[supervisor] safety net 0a: message carries an "
                    "<attachment> marker -- routing straight to "
                    "personal_docs without asking the model to choose.",
                    file=sys.stderr,
                )
                return {"route": "personal_docs", "iteration_count": iteration_count, "messages": []}

        # --- Early-stop net (always on, personal_docs-specific): same
        # shape as the invoice-specific one right below, applied to
        # safety net 0a instead of safety net 0 -- once a message with an
        # attachment marker has been routed to `personal_docs`, ITS
        # answer (refusal or not) is final. There is no "try a different
        # specialist" recovery from personal_docs coming up empty on an
        # attachment this turn genuinely just uploaded -- no other
        # specialist has any access to personal uploads at all
        # (`personal_docs` is the only one that does), so letting the
        # model take a second routing guess here can only ever produce
        # the exact CONFIRMED cascade safety net 0a exists to prevent in
        # the first place (see that net's own comment), just one visit
        # later. `personal_docs` is deliberately NOT in
        # `_DETERMINISTIC_NEVER_HEDGES` below -- a personal_docs search
        # NOT triggered by an attachment marker (e.g. a follow-up
        # question about something uploaded several turns ago) is a
        # genuinely different case where a second look could help, so
        # this stays scoped to the safety-net-0a-routed case specifically
        # rather than reclassifying the specialist's general behavior.
        if len(attempts) == 1 and attempts[0][0] == "personal_docs":
            if _message_carries_attachment(str(question)):
                print(
                    "[supervisor] early-stop (personal_docs, "
                    "safety-net-0a-routed): personal_docs' answer -- "
                    "refusal or not -- is final here; no other specialist "
                    "has any access to personal uploads. FINISHing "
                    "without a second supervisor LLM call.",
                    file=sys.stderr,
                )
                return {"route": "FINISH", "iteration_count": iteration_count, "messages": []}

        # --- Early-stop net (always on, invoice-specific): a refusal
        # from `invoice` specifically does NOT mean "wrong specialist,
        # let the model try again" when safety net 0 above is WHY it was
        # routed here -- that net only ever fires on a real id/name match
        # (or an explicit "all of them") against the most recent
        # product_search batch, so by the time invoice_node actually runs
        # there is no genuine doubt left about WHICH items were meant.
        # Its only possible refusal outcomes at that point are "nothing
        # could be priced" (a data-availability problem) or "isn't
        # available right now" (the tool itself is down) -- neither of
        # which ANY other specialist can fix.
        #
        # CONFIRMED live-run failure this closes: the general
        # _DETERMINISTIC_NEVER_HEDGES net right below deliberately
        # excludes refusals (`not _looks_like_refusal`), on the reasonable
        # premise that a refusal from one of those specialists MIGHT mean
        # "wrong specialist, try something else" -- reasonable in
        # general, but not once safety net 0 has already ruled that out
        # for invoice specifically. A live run asking to buy two named
        # items that both turned out to have no listed price had the
        # model insist on `product_search` on every subsequent visit, and
        # the repeat-route guard was forced to walk EVERY other
        # specialist in turn -- retrieval_qa, personal_docs, corpus_meta,
        # color_palette, multi_hop, image_qa, none of them remotely
        # related to the request -- burning the entire iteration cap
        # before a forced stop. `_finalize_with_first_attempt` still
        # correctly reaffirmed invoice's own original answer as the
        # FINAL one shown to the person, but the wasted visits are still
        # real: several extra LLM calls' worth of latency, and (if a
        # client renders every `turn_messages` entry rather than just
        # `answer`) a wall of unrelated specialist attempts appearing
        # before the real one.
        #
        # Re-derives `_looks_like_invoice_followup` here rather than
        # reading a stored flag, so this can never drift out of sync with
        # safety net 0's own decision -- the same "single source of
        # truth" reason `_looks_like_invoice_followup` and
        # `_latest_product_search_batch` are shared helpers
        # (agents/specialists.py) in the first place.
        if len(attempts) == 1 and attempts[0][0] == "invoice":
            latest_batch = _latest_product_search_batch(state["messages"])
            if _looks_like_invoice_followup(str(question), latest_batch):
                print(
                    "[supervisor] early-stop (invoice, safety-net-0-routed): "
                    "invoice's answer -- refusal or not -- is final here; no "
                    "other specialist can supply a missing price or fix an "
                    "unavailable tool. FINISHing without a second supervisor "
                    "LLM call.",
                    file=sys.stderr,
                )
                return {"route": "FINISH", "iteration_count": iteration_count, "messages": []}

        # --- Early-stop net (always on): a fully-deterministic
        # specialist's clean answer never needs a second opinion -- see
        # _DETERMINISTIC_NEVER_HEDGES's own comment for exactly which
        # specialists this covers and why. CONFIRMED live-run failure
        # this fixes: color_palette (fully deterministic) produced a
        # complete, correct color palette on its first visit; the second
        # supervisor LLM call then re-picked "color_palette" again (the
        # same "same route regardless of a changing transcript" failure
        # this module's docstring already documents for llama3.2), the
        # repeat-route guard correctly caught that and redirected to
        # retrieval_qa -- the next UNTRIED specialist in build order, not
        # the "next after color_palette" one, since _next_untried_route
        # always restarts its walk from the beginning of
        # ordered_specialist_names -- which then genuinely found
        # unrelated-but-plausible corpus content and produced its OWN
        # complete answer. Because api.py's ChatResponse.answer is always
        # the LAST specialist-named message, retrieval_qa's later,
        # off-topic answer silently became "the" answer shown to the
        # person, burying the correct, on-topic palette underneath it,
        # even though the model's own final decision was FINISH (which
        # is not itself wrong -- the turn WAS done -- the wrong part was
        # letting the repeat-driven detour happen at all).
        #
        # Unlike the opt-in net below, this one is NOT gated behind
        # skip_reroute_if_answered: it only ever fires for specialists
        # that cannot produce a hedging "partial or unclear" answer in
        # the first place (see _DETERMINISTIC_NEVER_HEDGES), so it
        # carries none of that toggle's documented tradeoff and is safe
        # to leave on unconditionally.
        if len(attempts) == 1 and attempts[0][0] in _DETERMINISTIC_NEVER_HEDGES and not _looks_like_refusal(
            attempts[0][1]
        ):
            print(
                f"[supervisor] early-stop (deterministic specialist): {attempts[0][0]!r} "
                "cannot produce a hedging answer and this one doesn't look like a refusal "
                "-- FINISHing without a second supervisor LLM call.",
                file=sys.stderr,
            )
            return {"route": "FINISH", "iteration_count": iteration_count, "messages": []}

        # --- Early-stop net (opt-in): skip the LLM call entirely on a
        # clean first answer from ANY specialist, including ones that
        # CAN legitimately hedge -- see DEFAULT_SKIP_REROUTE_IF_ANSWERED's
        # docstring for why this one stays gated behind a toggle rather
        # than always on. Fires only on the second visit (exactly one
        # prior attempt), and only when that attempt doesn't look like a
        # refusal; from the third visit onward this is skipped and the
        # model's own routing judgment (nets 1-4 below) runs as normal.
        if skip_reroute_if_answered and len(attempts) == 1 and not _looks_like_refusal(attempts[0][1]):
            print(
                f"[supervisor] early-stop: {attempts[0][0]!r} answered and doesn't "
                "look like a refusal -- FINISHing without a second supervisor LLM "
                "call. Pass skip_reroute_if_answered=False to build_supervisor() "
                "to disable this and always ask the model.",
                file=sys.stderr,
            )
            return {"route": "FINISH", "iteration_count": iteration_count, "messages": []}

        transcript = _format_transcript(attempts)
        human_content = SUPERVISOR_USER_TURN_TEMPLATE.format(question=question, transcript=transcript)

        response = await _llm_for(question).ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        )

        # Unconditional -- not just on failure -- so a live run's stderr
        # log shows the exact raw JSON for every visit, including ones
        # that turn out valid and non-repeated. This is what let the
        # "same route on every visit despite a changing transcript"
        # observation get confirmed directly from a real run's log
        # rather than inferred indirectly from which specialists ran.
        print(f"[supervisor] raw model output: {response.content!r}", file=sys.stderr)

        # --- Safety net 1: schema-level validation --------------------
        raw_route: Optional[str] = None
        try:
            decision = route_decision_model.model_validate_json(response.content)
            raw_route = decision.route
        except (ValidationError, json.JSONDecodeError, TypeError):
            raw_route = None

        tried_names = {name for name, _ in attempts}

        # --- Safety net 2: explicit membership check -------------------
        # Deliberately re-checks even a schema-valid raw_route against
        # known_routes -- the exact same value route_decision_model's own
        # Literal was built from a few lines up in build_supervisor(), so
        # this can no longer catch anything net 1 doesn't (see
        # _build_route_decision_model's own docstring for the confirmed
        # bug that used to be possible before the schema was built this
        # way). Kept anyway as cheap defense-in-depth against a FUTURE
        # regression -- e.g. someone decoupling the two again by hand --
        # rather than removed just because it is currently redundant by
        # construction.
        schema_and_membership_ok = raw_route is not None and raw_route in known_routes

        if not schema_and_membership_ok:
            print(
                f"[supervisor] routing decision {response.content!r} was invalid or "
                "unrecognized",
                file=sys.stderr,
            )

        # --- Safety net 3: repeat-route guard ---------------------------
        # Only meaningful once safety net 1+2 have passed and the route
        # isn't already FINISH -- see this module's docstring for the
        # confirmed live-run failure this net exists to catch.
        is_repeat = (
            schema_and_membership_ok and raw_route != "FINISH" and raw_route in tried_names
        )
        if is_repeat:
            print(
                f"[supervisor] routing decision {raw_route!r} repeats a specialist "
                f"already tried this turn ({sorted(tried_names)}) -- overriding",
                file=sys.stderr,
            )

        # --- Safety net 4: premature-FINISH guard -----------------------
        # A schema-valid, known "FINISH" is still rejected if no
        # specialist has answered yet this turn -- see this module's
        # docstring for the confirmed live-run failure (an out-of-scope
        # query, supervisor's very first output was FINISH, zero
        # specialists ever ran, turn ended with no answer at all) this
        # net exists to catch. `not attempts` is equivalent to "this is
        # truly the first visit": every path that overrides a route
        # (this one included) hands off to a real specialist immediately,
        # so attempts can only still be empty before that has ever
        # happened once.
        is_premature_finish = (
            schema_and_membership_ok and raw_route == "FINISH" and not attempts
        )
        if is_premature_finish:
            print(
                "[supervisor] routing decision 'FINISH' rejected -- no specialist "
                "has answered this turn yet -- overriding",
                file=sys.stderr,
            )

        extra_messages: list = []

        if schema_and_membership_ok and not is_repeat and not is_premature_finish:
            # Model's own choice survived all four nets untouched.
            validated_route = raw_route
        else:
            # Net 1/2 failed, net 3 caught a repeat, or net 4 caught a
            # premature FINISH -- in every case, prefer a specialist that
            # hasn't been tried yet over blindly reapplying fallback_route
            # (which may itself be the specialist that was just
            # repeated). For a premature FINISH specifically, tried_names
            # is empty, so this always lands on the first specialist in
            # `specialists`'s own build order.
            next_untried = _next_untried_route(ordered_specialist_names, tried_names)
            if next_untried is not None:
                validated_route = next_untried
            elif not schema_and_membership_ok and fallback_route not in tried_names:
                # Genuinely invalid output (not a repeat) and every
                # specialist has technically been tried, but
                # fallback_route was never one of them somehow -- edge
                # case kept for completeness, falls through to FINISH
                # below in the (expected) common case where it has.
                validated_route = fallback_route
            else:
                # Every specialist already tried and there's nothing
                # untried left to route to -- stop here in code rather
                # than spending more iteration-cap budget on a model
                # that either can't produce a valid route or won't try
                # anything new. See _finalize_with_first_attempt's
                # docstring for the confirmed live-run failure this
                # specific case caused (a 7-specialist walk landing on
                # `invoice`'s near-universal refusal as "the answer") and
                # why the fix is to reaffirm the FIRST specialist's
                # answer here, under its own name, rather than leaving
                # whichever specialist happened to run last as the
                # transcript's final specialist-named message.
                validated_route = "FINISH"
                note = _all_tried_note(attempts)
                extra_messages = [_finalize_with_first_attempt(note, _first_attempt_message(state))]

        return {
            "route": validated_route,
            "iteration_count": iteration_count,
            "messages": extra_messages,
        }

    return supervisor_node

"""
agents/eval_language.py

Two independent checks, mirroring this project's existing routing-only
vs. full-end-to-end eval split (eval_routing.py vs. eval_phase5.py),
applied to something this project has never explicitly evaluated
before: does the pipeline behave correctly ACROSS LANGUAGES -- English,
French, Arabic, Spanish -- including a single message that mixes more
than one language in it, the way people in a trilingual context
actually type (Lebanon's own Arabic/French/English code-switching is
the concrete case this question bank is built around).

Part 1 -- ROUTING ACROSS LANGUAGES (fast; only needs Ollama reachable):
Same technique eval_routing.py already uses for its own question bank
-- a real supervisor_node built over a dict of loud, unreachable stub
specialists (see _unreachable_specialist_stub below; this is valid for
the identical reason eval_routing.py's own docstring gives:
build_supervisor() only ever reads the specialists dict's KEYS, never
the functions themselves). The question this part asks isn't "does the
router understand art" -- eval_routing.py's English-only question bank
already covers that -- it's "does TRANSLATING or CODE-SWITCHING the
same underlying question change which specialist it lands on." A
router that handles an English question correctly but misroutes its
French or Arabic translation, or a message that mixes two languages in
one line, has a language-coverage gap eval_routing.py's own question
bank can never surface, because every row in it is English.

Part 2 -- LANGUAGE FIDELITY, END TO END (slow; needs the real live
stack -- Ollama, the MCP server, an ingested corpus, same requirements
as eval_phase5.py): runs real questions through the real compiled graph
and checks what language the FINAL ANSWER actually came back in.

This is only scored pass/fail for retrieval_qa and painting_lookup --
the only two specialists whose own system prompts (prompts.py) actually
promise "answer in the same language as the question" (see
RETRIEVAL_QA_SYSTEM_PROMPT and PAINTING_LOOKUP_SYNTHESIZE_PROMPT_TEMPLATE).
Every other specialist (corpus_meta, product_search, color_palette,
invoice, image_qa, personal_docs) has NO such rule in its own prompt
today, so this script only RECORDS the detected language for those
rows, as diagnostic material -- never marks them correct/incorrect
against a promise their prompt never made. Grading a specialist against
a rule that doesn't exist would be a fake failure, not a real one; the
same reasoning eval_phase5.py's own module docstring already gives for
why it refuses to auto-judge answer *correctness* applies here to
language *fidelity* wherever no explicit contract exists.

Where a contract genuinely doesn't exist yet for a case worth testing
anyway (a single message that mixes two languages itself, which no
specialist prompt addresses -- see prompts.py's language rules, which
only ever discuss the contextualize-appended parenthetical, a different
case), this script still runs the question and records what actually
happens, `scored: False`, with a `note` explaining why it isn't graded
-- so the result is visible and actionable (a candidate for a new
explicit prompt rule) rather than silently skipped.

Language detection here is deliberately NOT a placeholder for a human
judge, the way eval_phase5.py's own answer-quality [EYEBALL] rows are.
"Is this text mostly Arabic script" and "does this text lean on French
function words" are mechanically checkable the same way routing
correctness is mechanically checkable (see eval_routing.py's own
docstring on preferring a mechanical check over a subjective one
wherever one honestly exists) -- so `_detect_language` below is a
small, dependency-free, rule-based checker (Unicode-range script
detection, plus a short function-word list per Latin-script language),
not a probabilistic language-ID model. It's intentionally coarse: good
enough to tell "this answer came back in Arabic instead of French"
apart from "this answer is genuinely bilingual," not good enough to
certify translation quality or grammar. See its own docstring for
exactly what it can and can't tell you, and `_score_fidelity`'s own
"indeterminate" outcome for the case where it honestly can't guess.

Run Part 1 alone (fast, only needs Ollama):
    py -3.12 -m agents.eval_language

Run Part 1 + Part 2 (needs the full live stack, same as eval_phase5.py):
    py -3.12 -m agents.eval_language --full

Writes, next to this file:
    agents/eval_language_results.md
    agents/eval_language_results.json
"""

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

from agents.graph import build_graph
from agents.prompts import SPECIALIST_DESCRIPTIONS
from agents.specialists import Specialist
from agents.state import AgentState
from agents.supervisor import (
    DEFAULT_FALLBACK_ROUTE,
    DEFAULT_ITERATION_CAP,
    DEFAULT_ROUTE_FORMAT,
    build_supervisor,
)
from agents.tracing import new_request_id

_OUT_DIR = Path(__file__).resolve().parent

# Names that show up as a message's `.name` but are never the
# substantive "specialist answered" message -- filtered out when
# looking for the real final answer in a graph result's messages list.
_NON_SPECIALIST_NAMES = {"input_guard", "contextualize", "supervisor", "output_guard", "refuse"}


# ---------------------------------------------------------------------
# Part 1 -- routing across languages (stub specialists, no live corpus)
# ---------------------------------------------------------------------
# Every question below is a translated / code-switched sibling of a
# category prompts.py's own SPECIALIST_ROUTING_EXAMPLES already covers
# in English -- deliberately, so a routing failure here points straight
# at "this category breaks in French/Arabic/mixed input specifically,"
# not at an unrelated category the supervisor was never good at to
# begin with.
#
# `language` tags: "en" / "fr" / "ar" / "es" for a single-language row;
# "en+fr" etc. (opening language first) for a single MESSAGE that
# genuinely mixes two languages in one line -- this project's own
# CONTEXTUALIZE_SYSTEM_PROMPT already anticipates a cross-TURN language
# switch (its own worked Arabic example), but nothing in prompts.py
# addresses a single message mixing languages WITHIN itself, which is
# exactly the gap this question bank targets. "arabizi" tags one
# deliberately hard bonus row (Arabic transliterated into Latin script
# chat-speak, as commonly typed in Lebanon) -- included for visibility,
# not counted in the headline accuracy number (see run_routing_eval).
ROUTING_LANGUAGE_QUESTIONS: list[dict] = [
    {"id": "RL1", "language": "en", "category": "retrieval_qa", "expected_route": "retrieval_qa",
     "query": "How do I mix a good glaze for oil painting?"},
    {"id": "RL2", "language": "fr", "category": "retrieval_qa", "expected_route": "retrieval_qa",
     "query": "Comment mélanger un bon glacis pour la peinture à l'huile ?"},
    {"id": "RL3", "language": "ar", "category": "retrieval_qa", "expected_route": "retrieval_qa",
     "query": "كيف أحضّر طلاء زجاجي جيد للرسم بالزيت؟"},
    {"id": "RL4", "language": "es", "category": "retrieval_qa", "expected_route": "retrieval_qa",
     "query": "¿Cómo mezclo un buen glaseado para pintura al óleo?"},

    {"id": "RL5", "language": "en", "category": "corpus_meta", "expected_route": "corpus_meta",
     "query": "What documents are in your corpus?"},
    {"id": "RL6", "language": "fr", "category": "corpus_meta", "expected_route": "corpus_meta",
     "query": "Quels documents contient votre corpus ?"},
    {"id": "RL7", "language": "ar", "category": "corpus_meta", "expected_route": "corpus_meta",
     "query": "ما هي المستندات الموجودة في مجموعتكم؟"},

    {"id": "RL8", "language": "en", "category": "multi_hop", "expected_route": "multi_hop",
     "query": "How does Cennini's advice on tempera compare to Vasari's on fresco?"},
    {"id": "RL9", "language": "fr", "category": "multi_hop", "expected_route": "multi_hop",
     "query": "Comment les conseils de Cennini sur la détrempe se comparent-ils à ceux de Vasari sur la fresque ?"},

    {"id": "RL10", "language": "en", "category": "image_qa", "expected_route": "image_qa",
     "query": "Show me a picture of an underpainting technique"},
    {"id": "RL11", "language": "ar", "category": "image_qa", "expected_route": "image_qa",
     "query": "أرني صورة لتقنية الرسم التحضيري"},

    {"id": "RL12", "language": "en", "category": "painting_lookup", "expected_route": "painting_lookup",
     "query": "Tell me about the Mona Lisa"},
    {"id": "RL13", "language": "fr", "category": "painting_lookup", "expected_route": "painting_lookup",
     "query": "Parle-moi de la Joconde"},
    {"id": "RL14", "language": "ar", "category": "painting_lookup", "expected_route": "painting_lookup",
     "query": "أخبرني عن لوحة الموناليزا"},

    {"id": "RL15", "language": "en", "category": "product_search", "expected_route": "product_search",
     "query": "What's a good brush for glazing techniques?"},
    {"id": "RL16", "language": "fr", "category": "product_search", "expected_route": "product_search",
     "query": "Quel est un bon pinceau pour les techniques de glacis ?"},

    {"id": "RL17", "language": "en", "category": "invoice", "expected_route": "invoice",
     "query": "How much would the brushes you found cost in total?"},

    {"id": "RL18", "language": "en", "category": "color_palette", "expected_route": "color_palette",
     "query": "Give me a complementary color scheme based on cerulean blue"},
    {"id": "RL19", "language": "fr", "category": "color_palette", "expected_route": "color_palette",
     "query": "Donne-moi un schéma de couleurs complémentaires à partir du bleu cérulé"},

    {"id": "RL20", "language": "en", "category": "personal_docs", "expected_route": "personal_docs",
     "query": "What does the PDF I just uploaded say about this?"},
    {"id": "RL21", "language": "ar", "category": "personal_docs", "expected_route": "personal_docs",
     "query": "ماذا يقول الملف الذي رفعته للتو عن هذا الموضوع؟"},

    # --- single MESSAGE mixing two languages -- the real target of this file ---
    {"id": "RL22", "language": "en+fr", "category": "retrieval_qa", "expected_route": "retrieval_qa",
     "query": "What is glacis, comme technique de peinture à l'huile ?"},
    {"id": "RL23", "language": "en+ar", "category": "painting_lookup", "expected_route": "painting_lookup",
     "query": "Tell me about لوحة الموناليزا"},
    {"id": "RL24", "language": "fr+ar", "category": "product_search", "expected_route": "product_search",
     "query": "Je cherche un bon pinceau للتقنيات الزجاجية"},
    {"id": "RL25", "language": "ar+en", "category": "color_palette", "expected_route": "color_palette",
     "query": "أعطني complementary color scheme لللون الأزرق السماوي"},

    # --- bonus/stretch: Arabic transliterated into Latin chat-speak ("Arabizi") ---
    # Excluded from the headline accuracy number (see run_routing_eval) --
    # this is a genuinely different, much harder problem (no Arabic
    # Unicode at all for the model to lean on) than script-level code-
    # switching, kept visible rather than silently dropped.
    {"id": "RL26", "language": "arabizi", "category": "retrieval_qa", "expected_route": "retrieval_qa",
     "query": "kif ba7ad zeit mnih lal glazing bl oil painting?", "bonus": True},
]


async def _unreachable_specialist_stub(state: AgentState) -> dict:
    """Never actually called -- see eval_routing.py's identical helper for why a
    dict of these is enough to build a real supervisor_node without any of
    build_specialists()'s own MCP/Ollama/corpus setup cost. Raises loudly if it
    ever IS invoked, same "structural guardrail on this script's own scope
    limit" reasoning as that file's version."""
    raise RuntimeError(
        "A specialist stub was actually invoked -- this script only ever tests "
        "the supervisor's routing decision, never a real specialist."
    )


def _build_stub_specialists() -> dict[str, Specialist]:
    return {name: _unreachable_specialist_stub for name in SPECIALIST_DESCRIPTIONS}


async def _route_one(supervisor_node: Specialist, row: dict) -> dict:
    """One supervisor visit on a fresh state, no input_guard step -- unlike
    eval_routing.py's own _route_one, this question bank has no adversarial
    rows (eval_routing.py's own question bank already owns that dimension), so
    skipping the guard keeps this file focused on language, not injection
    detection."""
    state: AgentState = {
        "messages": [HumanMessage(content=row["query"])],
        "route": None,
        "iteration_count": 0,
        "blocked": False,
        "injection_patterns": [],
        "forced_route": None,
        "thread_id": None,
        "request_id": None,
    }
    start = time.monotonic()
    error: Optional[str] = None
    actual_route: Optional[str] = None
    try:
        result = await supervisor_node(state)
        actual_route = result.get("route")
    except Exception as exc:  # noqa: BLE001 -- a crashed row is its own result, not a reason to stop the whole eval
        error = f"{type(exc).__name__}: {exc}"
        print(f"[eval_language]   -> CRASHED: {error}", file=sys.stderr)

    elapsed = round(time.monotonic() - start, 2)
    correct = (not error) and (actual_route == row["expected_route"])
    return {**row, "actual_route": actual_route, "correct": correct,
            "elapsed_seconds": elapsed, "error": error}


async def run_routing_eval(
    questions: Optional[list[dict]] = None,
    supervisor_node: Optional[Specialist] = None,
) -> list[dict]:
    if questions is None:
        questions = ROUTING_LANGUAGE_QUESTIONS
    if supervisor_node is None:
        supervisor_node = build_supervisor(
            _build_stub_specialists(),
            iteration_cap=DEFAULT_ITERATION_CAP,
            fallback_route=DEFAULT_FALLBACK_ROUTE,
            route_format=DEFAULT_ROUTE_FORMAT,
        )

    results = []
    for row in questions:
        print(f"[eval_language] #{row['id']} ({row['language']}/{row['category']}): "
              f"{row['query']!r}", file=sys.stderr)
        result = await _route_one(supervisor_node, row)
        marker = "OK" if result["correct"] else ("CRASH" if result["error"] else "MISROUTE")
        print(f"[eval_language]   -> {marker}: expected={row['expected_route']!r} "
              f"actual={result['actual_route']!r}", file=sys.stderr)
        results.append(result)
    return results


# ---------------------------------------------------------------------
# Language detection -- see this module's own top docstring for the
# honest scope/limits of this approach.
# ---------------------------------------------------------------------
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z\u00C0-\u00FF\u0100-\u017F]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ']+")

_FUNCTION_WORDS = {
    "en": {"the", "is", "and", "of", "to", "in", "what", "how", "this", "are",
           "it", "for", "with", "a", "an", "does", "do", "you", "your", "on"},
    "fr": {"le", "la", "les", "de", "et", "que", "qui", "est", "une", "un",
           "dans", "pour", "avec", "du", "des", "qu", "ce", "cette", "comment",
           "vous", "votre", "sur"},
    "es": {"el", "la", "los", "las", "que", "es", "de", "y", "un", "una",
           "para", "con", "del", "cómo", "qué", "su", "en"},
}


def _detect_language(text: str) -> dict:
    """
    Two combined signals, both deliberately simple and auditable rather
    than a black-box model call (this project's own MCP tools already
    prefer a deterministic check over an LLM judgment wherever one
    exists -- e.g. image_qa/invoice being zero-LLM-call by design, per
    prompts.py's own module comment; this is the same preference
    applied to a language check):

      1. Unicode SCRIPT ranges -- fully reliable for Arabic vs. any
         Latin-script language, since the alphabets never overlap.
         `arabic_ratio` / `latin_ratio` are each (matching letters) /
         (all letters found), ignoring digits, punctuation, and
         markdown syntax so a `[source: ...]` citation or an image tag
         doesn't skew the ratio.
      2. FUNCTION-WORD overlap, only within the text's own Latin-script
         words -- Unicode ranges alone can't distinguish English from
         French or Spanish (same alphabet). Counts how many of the
         text's words appear on each candidate language's short
         function-word list and returns whichever list scored the most
         hits, but ONLY if that best score is at least 2 -- a single
         coincidental hit (e.g. "un" appearing inside an English
         sentence) isn't enough to claim a guess. Below that bar,
         `latin_lang_guess` is honestly None ("indeterminate") rather
         than a confident-looking wrong answer.

    This is NOT a real language-ID model -- see this module's own top
    docstring for what it's good enough for and what it isn't.
    """
    arabic_letters = _ARABIC_RE.findall(text)
    latin_letters = _LATIN_LETTER_RE.findall(text)
    total_letters = len(arabic_letters) + len(latin_letters)

    if total_letters == 0:
        return {"primary_script": "other", "arabic_ratio": 0.0, "latin_ratio": 0.0,
                "latin_lang_guess": None}

    arabic_ratio = len(arabic_letters) / total_letters
    latin_ratio = len(latin_letters) / total_letters

    if arabic_ratio >= 0.85:
        primary_script = "arabic"
    elif latin_ratio >= 0.85:
        primary_script = "latin"
    elif arabic_ratio >= 0.15 and latin_ratio >= 0.15:
        primary_script = "mixed"
    else:
        primary_script = "arabic" if arabic_ratio > latin_ratio else "latin"

    latin_lang_guess = None
    if latin_ratio > 0:
        words = [w.strip("'").lower() for w in _WORD_RE.findall(text)]
        scores = {lang: sum(1 for w in words if w in wordset)
                  for lang, wordset in _FUNCTION_WORDS.items()}
        best_lang, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score >= 2:
            latin_lang_guess = best_lang

    return {
        "primary_script": primary_script,
        "arabic_ratio": round(arabic_ratio, 3),
        "latin_ratio": round(latin_ratio, 3),
        "latin_lang_guess": latin_lang_guess,
    }


def _score_fidelity(expected_lang: str, detected: dict) -> str:
    """
    "match" / "mismatch" / "indeterminate" -- three states, not a bool,
    specifically so a case _detect_language honestly can't resolve
    (latin_lang_guess is None) is never silently counted as either a
    pass or a fail. See eval_phase5.py's own [EYEBALL] convention for
    the same "don't fabricate a verdict a mechanical check can't
    actually support" principle, applied here to language instead of
    content correctness.
    """
    if expected_lang == "ar":
        return "match" if detected["primary_script"] == "arabic" else "mismatch"
    if detected["primary_script"] != "latin":
        return "mismatch"
    if detected["latin_lang_guess"] is None:
        return "indeterminate"
    return "match" if detected["latin_lang_guess"] == expected_lang else "mismatch"


# ---------------------------------------------------------------------
# Part 2 -- language fidelity, end to end (real graph, real corpus)
# ---------------------------------------------------------------------
# `expected_lang`/`scored` are either a single value (applied to every
# turn) or a per-turn list, for the two multi-turn rows below. `None`
# for `expected_lang` always means "record only, never score" --
# whether because no specialist prompt makes a language promise for
# that route (F9), or because the question ITSELF is the untested edge
# case this file exists to surface (F5's single mixed-language message).
FIDELITY_QUESTIONS: list[dict] = [
    {"id": "F1", "expected_route": "retrieval_qa", "expected_lang": "en", "scored": True,
     "turns": ["What is sfumato?"]},
    {"id": "F2", "expected_route": "retrieval_qa", "expected_lang": "fr", "scored": True,
     "turns": ["Qu'est-ce que le sfumato ?"]},
    {"id": "F3", "expected_route": "retrieval_qa", "expected_lang": "ar", "scored": True,
     "turns": ["ما هو السفوماتو؟"]},
    {"id": "F4", "expected_route": "retrieval_qa", "expected_lang": "es", "scored": True,
     "turns": ["¿Qué es el sfumato?"]},
    {"id": "F5", "expected_route": "retrieval_qa", "expected_lang": None, "scored": False,
     "turns": ["What is glazing, و شو هيّ تقنية السفوماتو كمان؟"],
     "note": ("Single message mixing English and Arabic in one line. Neither "
              "RETRIEVAL_QA_SYSTEM_PROMPT nor any other specialist prompt has an "
              "explicit rule for a question that mixes languages ITSELF -- their "
              "existing 'same language' rule is about the question vs. "
              "contextualize's own appended parenthetical, a different case (see "
              "prompts.py). Diagnostic only, on purpose: recorded, never scored "
              "pass/fail, because there is no written contract yet to score it "
              "against. If this keeps coming back e.g. English-only across "
              "several runs, that's a real candidate for a new explicit rule in "
              "RETRIEVAL_QA_SYSTEM_PROMPT -- not evidence of a bug today.")},
    {"id": "F6", "expected_route": "painting_lookup", "expected_lang": "en", "scored": True,
     "turns": ["Tell me about Starry Night"]},
    {"id": "F7", "expected_route": "painting_lookup", "expected_lang": "ar", "scored": True,
     "turns": ["أخبرني عن لوحة الموناليزا"]},
    {"id": "F8", "expected_route": "painting_lookup", "expected_lang": [None, "ar"],
     "scored": [False, True],
     "turns": ["Tell me about the Mona Lisa", "من رسمها؟"],
     "note": ("Multi-turn, mirrors CONTEXTUALIZE_SYSTEM_PROMPT's own worked "
              "example almost exactly (English topic, then an Arabic "
              "pronoun-attached follow-up -- 'رسمها' = 'painted' + attached "
              "'it'). Turn 1 is topic setup only, not scored. Turn 2 is the "
              "real test: does contextualize rewrite 'من رسمها؟' into 'من رسم "
              "الموناليزا' using the Mona Lisa from turn 1, and does "
              "painting_lookup then answer turn 2 in Arabic, not English.")},
    {"id": "F9", "expected_route": "product_search", "expected_lang": [None, None],
     "scored": [False, False],
     "turns": ["Quel est un bon pinceau pour les techniques de glacis ?", "which size is best?"],
     "note": ("Multi-turn, French then English. product_search has NO "
              "documented same-language contract in its own prompt (unlike "
              "retrieval_qa/painting_lookup), so language is recorded for "
              "information only here. What IS scored: routing staying "
              "product_search on both turns despite the language switch "
              "mid-thread -- see each turn's own routing_correct field.")},
]


def _extract_final_answer(result: dict) -> tuple[Optional[str], Optional[str]]:
    """(specialist_name, content) of the last named, non-guard/non-supervisor
    message in a graph result -- the actual substantive answer, same
    "last named message, not the last message" pattern eval_phase5.py's own
    _extract_route_info already uses, for the same reason (a repeat-route-guard
    note from the supervisor can sit after the real answer)."""
    named = [(getattr(m, "name", None), m.content) for m in result.get("messages", [])
             if getattr(m, "name", None)]
    specialist_named = [nm for nm in named if nm[0] not in _NON_SPECIALIST_NAMES]
    if not specialist_named:
        return None, None
    return specialist_named[-1]


async def run_fidelity_eval(questions: Optional[list[dict]] = None) -> list[dict]:
    if questions is None:
        questions = FIDELITY_QUESTIONS

    try:
        from langgraph.checkpoint.memory import InMemorySaver as _Saver
    except ImportError:  # older langgraph pin -- same class under its old name
        from langgraph.checkpoint.memory import MemorySaver as _Saver

    print("[eval_language] building the real graph for Part 2 (needs Ollama, the "
          "MCP server, and an ingested corpus)...", file=sys.stderr)
    # One shared in-memory checkpointer for every row below -- never the
    # real chat_history.sqlite3 the live app uses, so this eval can never
    # leave test rows sitting in a real user's chat history. Each
    # multi-turn row still gets its OWN random thread_id (below), so
    # rows never see each other's conversation state even though they
    # share one checkpointer instance.
    graph = await build_graph(checkpointer=_Saver())

    results = []
    for row in questions:
        turns = row["turns"]
        expected_langs = row["expected_lang"] if isinstance(row["expected_lang"], list) \
            else [row["expected_lang"]] * len(turns)
        scored_flags = row["scored"] if isinstance(row["scored"], list) \
            else [row["scored"]] * len(turns)
        thread_id = f"eval-language-{row['id']}-{uuid.uuid4().hex[:8]}" if len(turns) > 1 else None

        turn_results = []
        for i, (turn_text, expected_lang, scored) in enumerate(
            zip(turns, expected_langs, scored_flags)
        ):
            print(f"[eval_language] {row['id']} turn {i + 1}/{len(turns)}: {turn_text!r}",
                  file=sys.stderr)
            state: AgentState = {
                "messages": [HumanMessage(content=turn_text)],
                "route": None,
                "iteration_count": 0,
                "blocked": False,
                "injection_patterns": [],
                "forced_route": None,
                "thread_id": thread_id,
                "request_id": new_request_id(),
            }
            config = {"recursion_limit": 25}
            if thread_id:
                config["configurable"] = {"thread_id": thread_id}

            error: Optional[str] = None
            actual_route: Optional[str] = None
            answer_text: Optional[str] = None
            try:
                result = await graph.ainvoke(state, config=config)
                actual_route, answer_text = _extract_final_answer(result)
            except Exception as exc:  # noqa: BLE001 -- a crashed turn is its own result row, not a reason to abandon the rest of the eval
                error = f"{type(exc).__name__}: {exc}"
                print(f"[eval_language]   -> CRASHED: {error}", file=sys.stderr)

            detected = _detect_language(answer_text) if answer_text else None
            outcome = None
            if scored and expected_lang and detected:
                outcome = _score_fidelity(expected_lang, detected)

            print(f"[eval_language]   -> route={actual_route!r} "
                  f"detected={detected} outcome={outcome!r}", file=sys.stderr)

            turn_results.append({
                "turn_index": i,
                "turn_text": turn_text,
                "expected_route": row["expected_route"],
                "actual_route": actual_route,
                "routing_correct": (actual_route == row["expected_route"]) if actual_route else None,
                "expected_lang": expected_lang,
                "scored": scored,
                "detected_language": detected,
                "outcome": outcome,
                "answer_text": answer_text,
                "error": error,
            })

        results.append({"id": row["id"], "note": row.get("note"), "turns": turn_results})
    return results


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def _routing_accuracy_by_language(results: list[dict]) -> dict[str, tuple[int, int]]:
    """{language_tag: (correct, total)}, EXCLUDING bonus rows (see ROUTING_LANGUAGE_QUESTIONS'
    own comment on why "arabizi" is tracked separately, never folded into the headline number)."""
    tally: dict[str, list[int]] = {}
    for r in results:
        if r.get("bonus"):
            continue
        tally.setdefault(r["language"], [0, 0])
        tally[r["language"]][1] += 1
        if r["correct"]:
            tally[r["language"]][0] += 1
    return {lang: (c, t) for lang, (c, t) in tally.items()}


def render_markdown_report(
    routing_results: list[dict],
    fidelity_results: Optional[list[dict]],
) -> str:
    scored_routing = [r for r in routing_results if not r.get("bonus")]
    bonus_routing = [r for r in routing_results if r.get("bonus")]
    total = len(scored_routing)
    correct = sum(1 for r in scored_routing if r["correct"])
    overall_accuracy = correct / total if total else 0.0

    lines = [
        "# Cross-language evaluation results",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} by `agents/eval_language.py`.",
        "",
        "## Part 1 -- routing across languages (routing-only, no specialist ever ran)",
        "",
        f"**Overall routing accuracy: {correct}/{total} ({overall_accuracy:.0%})** "
        f"across English, French, Arabic, Spanish, and four single-message "
        f"code-switched (two-language) questions.",
        "",
        "### Accuracy by language",
        "",
        "| Language | Correct / Total | Accuracy |",
        "|---|---|---|",
    ]
    for lang, (c, t) in _routing_accuracy_by_language(scored_routing).items():
        acc = c / t if t else 0.0
        lines.append(f"| `{lang}` | {c} / {t} | {acc:.0%} |")
    lines.append("")

    if bonus_routing:
        lines.append("### Bonus (not counted above): Arabizi / chat-speak transliteration")
        lines.append("")
        lines.append("| # | Query | Expected | Actual | Correct |")
        lines.append("|---|---|---|---|---|")
        for r in bonus_routing:
            lines.append(f"| {r['id']} | {r['query']} | `{r['expected_route']}` | "
                          f"`{r['actual_route'] or '—'}` | {'Y' if r['correct'] else 'N'} |")
        lines.append("")

    lines.append("### Per-question results")
    lines.append("")
    lines.append("| # | Language | Category | Query | Expected | Actual | Correct | Error |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in scored_routing:
        query_display = r["query"].replace("|", "\\|")
        error_display = (r["error"] or "").replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['language']} | {r['category']} | {query_display} | "
            f"`{r['expected_route']}` | `{r['actual_route'] or '—'}` | "
            f"{'Y' if r['correct'] else 'N'} | {error_display} |"
        )
    lines.append("")

    lines.append("## Part 2 -- language fidelity, end to end")
    lines.append("")
    if fidelity_results is None:
        lines.append(
            "Not run this pass -- re-run with `--full` against the real live "
            "stack (Ollama, MCP server, ingested corpus) to include it: "
            "`py -3.12 -m agents.eval_language --full`."
        )
        lines.append("")
        return "\n".join(lines)

    scored_turns = [(row["id"], t) for row in fidelity_results for t in row["turns"] if t["scored"]]
    matches = sum(1 for _, t in scored_turns if t["outcome"] == "match")
    mismatches = sum(1 for _, t in scored_turns if t["outcome"] == "mismatch")
    indeterminate = sum(1 for _, t in scored_turns if t["outcome"] == "indeterminate")
    lines.append(
        f"**Scored turns: {len(scored_turns)} -- {matches} match, {mismatches} mismatch, "
        f"{indeterminate} indeterminate** (only retrieval_qa and painting_lookup turns are "
        "scored; see this file's own module docstring for why every other specialist's "
        "language is recorded for diagnostics only, not graded)."
    )
    lines.append("")
    lines.append("| Row | Turn | Query | Route (exp/actual) | Expected lang | "
                  "Detected | Scored? | Outcome |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in fidelity_results:
        for t in row["turns"]:
            query_display = t["turn_text"].replace("|", "\\|")
            detected = t["detected_language"]
            detected_display = (
                f"{detected['primary_script']}"
                f"{'/' + detected['latin_lang_guess'] if detected.get('latin_lang_guess') else ''}"
            ) if detected else "—"
            lines.append(
                f"| {row['id']} | {t['turn_index'] + 1} | {query_display} | "
                f"`{t['expected_route']}` / `{t['actual_route'] or '—'}` | "
                f"{t['expected_lang'] or '—'} | {detected_display} | "
                f"{'Y' if t['scored'] else 'N'} | {t['outcome'] or '—'} |"
            )
    lines.append("")

    notes = [row for row in fidelity_results if row.get("note")]
    if notes:
        lines.append("### Diagnostic notes (unscored rows worth a human read)")
        lines.append("")
        for row in notes:
            lines.append(f"**{row['id']}** -- {row['note']}")
            lines.append("")
            for t in row["turns"]:
                if t["answer_text"]:
                    snippet = t["answer_text"][:300].replace("\n", " ")
                    lines.append(f"> Turn {t['turn_index'] + 1} answer: {snippet}...")
            lines.append("")

    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate routing and answer-language fidelity across languages."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Also run Part 2 (end-to-end language fidelity) -- needs the full "
             "live stack: Ollama, the MCP server, and an ingested corpus.",
    )
    args = parser.parse_args()

    routing_results = await run_routing_eval()

    fidelity_results: Optional[list[dict]] = None
    if args.full:
        fidelity_results = await run_fidelity_eval()

    report = render_markdown_report(routing_results, fidelity_results)

    md_path = _OUT_DIR / "eval_language_results.md"
    json_path = _OUT_DIR / "eval_language_results.json"
    md_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"routing": routing_results, "fidelity": fidelity_results},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    scored_routing = [r for r in routing_results if not r.get("bonus")]
    total = len(scored_routing)
    correct = sum(1 for r in scored_routing if r["correct"])
    print(f"\n[eval_language] routing accuracy: {correct}/{total} ({correct / total:.0%})",
          file=sys.stderr)
    if fidelity_results is not None:
        scored_turns = [t for row in fidelity_results for t in row["turns"] if t["scored"]]
        matches = sum(1 for t in scored_turns if t["outcome"] == "match")
        print(f"[eval_language] language fidelity: {matches}/{len(scored_turns)} matched "
              f"({matches / len(scored_turns):.0%})" if scored_turns else
              "[eval_language] language fidelity: no scored turns", file=sys.stderr)
    print(f"[eval_language] wrote {md_path}", file=sys.stderr)
    print(f"[eval_language] wrote {json_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

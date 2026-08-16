"""
Routing-only evaluation: routing accuracy and a confusion matrix, over a
large, curated question bank -- WITHOUT ever running a specialist's
actual tools.

This is a deliberately narrower sibling of eval_phase5.py, not a
replacement for it. eval_phase5.py runs 10 designed queries through the
REAL, compiled graph end to end (input_guard -> contextualize ->
supervisor -> the chosen specialist's actual tool calls -> output_guard)
and judges routing AND, where mechanically possible, answer correctness.
This script answers a narrower question over a much larger question set:
"did the supervisor pick the right specialist first?" -- nothing more.
It never calls a specialist's actual generate_answer/retrieve/web-search
tools, and it never needs a running MCP server or an ingested corpus to
do that, because build_supervisor() (agents/supervisor.py) only ever
reads the KEYS of the `specialists` dict it's handed -- to build its
known-routes Literal schema, its system prompt's specialist list, and
its repeat-route-guard walk order (see build_supervisor's own docstring)
-- never the specialist functions themselves. A dict of loud, deliberately-
unreachable stub functions (see `_unreachable_specialist_stub` below) is
enough to build a REAL, fully-wired supervisor_node from this project's
real routing prompt and real safety nets, with none of build_specialists()'s
own setup cost (spawning the MCP server, loading tools, fetching the live
corpus).

What this buys you over eval_phase5.py, specifically for "measure the
router":

  - Fast: no MCP server, no corpus, no specialist tool calls -- only
    Ollama itself (still required, for the supervisor's own routing
    LLM). Dozens of questions run in the time eval_phase5.py's 10
    would, since nothing here waits on retrieve()/generate_answer()/
    web search.
  - A real confusion matrix, not just a Y/N table: with 64 questions
    across 9 specialists plus "blocked", the off-diagonal cells this
    script's confusion matrix produces are exactly the "one prompt fix,
    not a model problem" diagnosis the class's own slides describe --
    see render_confusion_matrix_png's own docstring.
  - Decoupled from corpus content: eval_phase5.py's own docstring
    already warns its 10 queries assume a specific ingested corpus;
    this script's questions test ROUTING ONLY (which specialist should
    handle this, based on the question's own phrasing), so they don't
    go stale if the corpus changes.

What it does NOT give you, on purpose:

  - No answer-quality judgment at all -- there IS no answer, because no
    specialist ever ran. Use eval_phase5.py (or a live run) for that.
  - No trajectory/re-route/iteration-cap behavior -- every question here
    gets exactly ONE supervisor visit, on a fresh state with zero prior
    attempts, matching the supervisor's OWN first-visit behavior on a
    real turn (see build_supervisor()'s docstring: with no attempts yet,
    none of its four early-stop/safety nets have anything to act on
    except the model's own first choice or the schema/premature-FINISH
    override). This deliberately measures the FIRST routing decision
    only, same "first_route, not the repeat-guard's later detour" choice
    eval_phase5.py's own _extract_route_info already makes for the same
    reason.

Run against a running Ollama (no MCP server, no corpus needed):
    py -3.12 -m agents.eval_routing

Writes, next to this file:
    agents/eval_routing_results.md    (report-ready table + summary)
    agents/eval_routing_results.json  (same data, structured)
And, at the project root (one level up from agents/):
    screenshots/routing_confusion_matrix.png
"""

import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from langchain_core.messages import HumanMessage

from agents.guardrails import input_guard_node
from agents.prompts import SPECIALIST_DESCRIPTIONS
from agents.specialists import Specialist
from agents.state import AgentState
from agents.supervisor import (
    DEFAULT_FALLBACK_ROUTE,
    DEFAULT_ITERATION_CAP,
    DEFAULT_ROUTE_FORMAT,
    build_supervisor,
)

# The question bank this script evaluates by default -- see this
# module's own docstring, and QUESTIONS_PATH's own comment below, for
# why this is a checked-in, hand-curated file rather than generated
# fresh on every run. Regenerate deliberately, not implicitly: the
# slides' own "Building Your Test Set" guidance is explicit that
# changing the questions between runs means you can no longer compare
# today's numbers to yesterday's.
QUESTIONS_PATH = Path(__file__).resolve().parent / "routing_eval_questions.json"

# Screenshots directory lives at the PROJECT ROOT (one level up from
# agents/, this file's own parent) -- alongside README.md, local_rag/,
# mcp_server/, frontend/ -- not inside agents/ itself, so it's easy to
# find without knowing this script produced it.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = _PROJECT_ROOT / "screenshots"
CONFUSION_MATRIX_PATH = SCREENSHOTS_DIR / "routing_confusion_matrix.png"

# The label used in both the question bank's own `expected_route` values
# and this script's own results for "input_guard flagged this before the
# supervisor ever ran" -- deliberately the same string eval_phase5.py's
# QUERIES already uses for its two adversarial rows, so a person reading
# both scripts' output doesn't have to learn two different conventions
# for the same concept.
_BLOCKED_LABEL = "blocked"


async def _unreachable_specialist_stub(state: AgentState) -> dict:
    """
    Never actually called. build_supervisor() (agents/supervisor.py)
    only ever reads the KEYS of the `specialists` dict it's handed, not
    the functions themselves -- see this module's own top docstring for
    exactly what those keys get used for. A dict of these stubs is
    therefore enough to build a real, fully-wired supervisor_node
    without any of build_specialists()'s own setup cost.

    Raises loudly rather than quietly returning something if it's ever
    actually invoked -- which would mean this script's own "only test
    routing, never run a specialist" guarantee had silently broken (e.g.
    a future edit here started continuing the graph past the
    supervisor's single routing decision instead of just reading its
    return value). Same "structural guardrail, not just a comment"
    preference this project already applies elsewhere (corpus_meta_node
    having no tools at all; output_guard's redaction not depending on a
    model's cooperation) -- here applied to this script's own scope
    limit instead of the production graph's.
    """
    raise RuntimeError(
        "A specialist stub was actually invoked -- eval_routing.py's whole "
        "point is to test the supervisor's routing decision in isolation, "
        "never to run a real specialist. If you're seeing this, something "
        "here started continuing the graph loop instead of reading "
        "supervisor_node's return value directly and stopping."
    )


def _build_stub_specialists() -> dict[str, Specialist]:
    """
    SPECIALIST_DESCRIPTIONS (agents/prompts.py) is the same source
    build_supervisor() itself renders into the routing prompt's
    {specialist_descriptions} slot (see build_supervisor's own
    `specialist_descriptions = "\\n".join(... for name in specialists)`
    line) -- using its keys here as the known-routes set means this
    script's supervisor is built from the exact same specialist
    descriptions and worked examples (SPECIALIST_ROUTING_EXAMPLES) a
    real graph.py run would use, just without paying build_specialists()'s
    MCP/corpus setup cost to get them.

    Trade-off worth being explicit about: build_specialists()'s own
    returned dict (agents/specialists.py) is the ultimate source of
    truth for which specialists a REAL run has -- if a new specialist is
    ever added there without a matching entry in SPECIALIST_DESCRIPTIONS,
    this script won't know about it (though the real supervisor would
    still route to it fine with a generic placeholder description -- see
    build_supervisor's own comment on that). That's an accepted,
    documented gap in exchange for this script never needing a live MCP
    server or an ingested corpus just to test routing.
    """
    return {name: _unreachable_specialist_stub for name in SPECIALIST_DESCRIPTIONS}


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """
    Load the question bank -- a list of {id, category, query,
    expected_route} dicts, one per line of agents/routing_eval_questions.json.
    Raises FileNotFoundError with a clear message (rather than silently
    generating a fresh, uncomparable set) if the file is missing --
    regenerating the question bank is a deliberate, separate step, not
    an implicit side effect of running the eval.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Routing question bank not found at {path}. This file should be "
            "checked into the repo alongside eval_routing.py -- if it's "
            "genuinely missing, curate a new one following the same "
            "{id, category, query, expected_route} shape as the existing "
            "rows (expected_route must be one of SPECIALIST_DESCRIPTIONS's "
            "keys, or 'blocked' for an adversarial/injection question)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


async def _route_one(supervisor_node: Specialist, row: dict) -> dict:
    """
    Run exactly ONE question through input_guard, then (if not blocked)
    exactly ONE supervisor visit -- never a specialist, never a second
    supervisor visit. Mirrors the real graph's own turn-start order
    (graph.py: START -> input_guard -> ... -> supervisor) for the two
    nodes this script actually exercises, on a fresh AgentState with no
    prior attempts, matching a real turn's own first visit.
    """
    question = row["query"]
    state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "route": None,
        "iteration_count": 0,
        "blocked": False,
        "injection_patterns": [],
        "forced_route": None,
        "thread_id": None,
    }

    start = time.monotonic()
    error: Optional[str] = None
    actual_route: Optional[str] = None
    blocked = False
    injection_patterns: list[str] = []

    try:
        guard_result = await input_guard_node(state)
        state = {**state, **guard_result}
        blocked = bool(state.get("blocked"))
        injection_patterns = state.get("injection_patterns") or []

        if blocked:
            actual_route = _BLOCKED_LABEL
        else:
            # The one and only supervisor call this script ever makes per
            # question -- its return value IS the result; nothing here
            # ever follows `route` onward into the specialist it names.
            sup_result = await supervisor_node(state)
            actual_route = sup_result.get("route")
    except Exception as exc:  # noqa: BLE001 -- a crashed question is its own result row, not a reason to stop the whole eval
        error = f"{type(exc).__name__}: {exc}"
        print(f"[eval_routing]   -> CRASHED: {error}", file=sys.stderr)

    elapsed = round(time.monotonic() - start, 2)
    correct = (not error) and (actual_route == row["expected_route"])
    return {
        **row,
        "actual_route": actual_route,
        "blocked": blocked,
        "injection_patterns": injection_patterns,
        "correct": correct,
        "elapsed_seconds": elapsed,
        "error": error,
    }


async def run_eval(
    questions: Optional[list[dict]] = None,
    supervisor_node: Optional[Specialist] = None,
) -> list[dict]:
    """
    Run every question in `questions` (defaults to load_questions())
    through `supervisor_node` (defaults to a freshly-built one over the
    stub specialists dict -- see _build_stub_specialists()'s own
    docstring). Both are parameters, not hardcoded imports/builds,
    specifically so a smoke test can hand this a fake supervisor_node and
    a tiny fixed question list to exercise the loop/scoring/rendering
    logic below without a running Ollama at all.

    Sequential, not concurrent -- same reasoning eval_phase5.py's own
    run_eval already gives: every call would otherwise hit the same
    local Ollama instance, and sequential timing is the honest number for
    the report.
    """
    if questions is None:
        questions = load_questions()
    if supervisor_node is None:
        supervisor_node = build_supervisor(
            _build_stub_specialists(),
            iteration_cap=DEFAULT_ITERATION_CAP,
            fallback_route=DEFAULT_FALLBACK_ROUTE,
            route_format=DEFAULT_ROUTE_FORMAT,
        )

    results = []
    for row in questions:
        print(f"[eval_routing] #{row['id']} ({row['category']}): {row['query']!r}", file=sys.stderr)
        result = await _route_one(supervisor_node, row)
        marker = "OK" if result["correct"] else ("CRASH" if result["error"] else "MISROUTE")
        print(
            f"[eval_routing]   -> {marker}: expected={row['expected_route']!r} "
            f"actual={result['actual_route']!r}",
            file=sys.stderr,
        )
        results.append(result)
    return results


def build_confusion_matrix(results: list[dict]) -> tuple[list[str], list[list[int]]]:
    """
    Returns (labels, matrix) where matrix[i][j] is the count of rows with
    expected_route == labels[i] and actual_route == labels[j].
    `labels` is every expected_route value (in the question bank's own
    first-seen order, so the 9 specialists-plus-blocked stay grouped the
    same way the question bank itself is organized) plus any ADDITIONAL
    actual_route value that never appears as an expected_route (e.g. a
    crashed row's None, rendered as "(no route / error)") appended at
    the end -- so a genuinely unexpected actual value still gets its own
    row/column instead of being silently dropped or merged into an
    existing label.
    """
    expected_order: list[str] = []
    for r in results:
        if r["expected_route"] not in expected_order:
            expected_order.append(r["expected_route"])

    actual_values = {r["actual_route"] if r["actual_route"] else "(no route / error)" for r in results}
    labels = expected_order + [v for v in sorted(actual_values) if v not in expected_order]

    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for r in results:
        actual = r["actual_route"] if r["actual_route"] else "(no route / error)"
        matrix[index[r["expected_route"]]][index[actual]] += 1
    return labels, matrix


def render_confusion_matrix_png(
    results: list[dict], out_path: Path = CONFUSION_MATRIX_PATH
) -> Path:
    """
    Renders build_confusion_matrix()'s output as a heatmap PNG: rows are
    the expected route, columns are the actual route the supervisor
    picked, each cell annotated with its raw count. The diagonal is
    correct routing; everything off-diagonal is a specific, nameable
    misrouting pattern (e.g. a whole row of "product_search" questions
    landing in the "retrieval_qa" column would show up as one bright
    off-diagonal cell, immediately readable as "one prompt fix, not a
    model problem" -- same diagnostic framing the class's own slides use
    for this exact kind of matrix).

    matplotlib is used with the non-interactive "Agg" backend (set
    BEFORE importing pyplot) since this runs headless, from a script, not
    inside a notebook or a GUI session.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, matrix = build_confusion_matrix(results)
    n = len(labels)
    max_count = max((max(row) for row in matrix), default=0)

    fig_w = max(8.0, n * 0.9)
    fig_h = max(6.5, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Actual route (supervisor's decision)")
    ax.set_ylabel("Expected route")
    ax.set_title(
        "Routing confusion matrix (routing-only, no specialist ever ran)\n"
        "Diagonal = correct routing. Off-diagonal = a specific misroute."
    )

    for i in range(n):
        for j in range(n):
            count = matrix[i][j]
            if count:
                color = "white" if max_count and count > max_count / 2 else "black"
                ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="count", fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _accuracy_by_expected_route(results: list[dict]) -> dict[str, tuple[int, int]]:
    """Per-class recall material: {expected_route: (correct_count, total_count)},
    in the same first-seen order build_confusion_matrix() uses for its labels."""
    tally: dict[str, list[int]] = {}
    for r in results:
        route = r["expected_route"]
        tally.setdefault(route, [0, 0])
        tally[route][1] += 1
        if r["correct"]:
            tally[route][0] += 1
    return {route: (c, t) for route, (c, t) in tally.items()}


def render_markdown_report(results: list[dict], confusion_matrix_path: Optional[Path]) -> str:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    crashed = sum(1 for r in results if r["error"])
    overall_accuracy = correct / total if total else 0.0

    lines = [
        "# Routing-only evaluation results",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} by `agents/eval_routing.py`.",
        "",
        (
            f"**Overall routing accuracy: {correct}/{total} ({overall_accuracy:.0%})**"
            f"{f' -- {crashed} crashed before a route was ever decided.' if crashed else ''}"
        ),
        "",
        (
            "No specialist ever ran for any row below -- every question got exactly one "
            "input_guard check and, if not blocked, exactly one supervisor routing "
            "decision. See this file's own module docstring for what that does and "
            "doesn't measure."
        ),
        "",
    ]

    if confusion_matrix_path is not None:
        lines.append(f"Confusion matrix image: `{confusion_matrix_path}`")
        lines.append("")

    lines.append("## Accuracy by expected route")
    lines.append("")
    lines.append("| Expected route | Correct / Total | Accuracy |")
    lines.append("|---|---|---|")
    for route, (c, t) in _accuracy_by_expected_route(results).items():
        acc = c / t if t else 0.0
        lines.append(f"| `{route}` | {c} / {t} | {acc:.0%} |")
    lines.append("")

    lines.append("## Per-question results")
    lines.append("")
    lines.append("| # | Category | Query | Expected | Actual | Correct | Elapsed (s) | Error |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        query_display = r["query"].replace("|", "\\|")
        error_display = (r["error"] or "").replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['category']} | {query_display} | `{r['expected_route']}` | "
            f"`{r['actual_route'] or '—'}` | {'Y' if r['correct'] else 'N'} | "
            f"{r['elapsed_seconds']} | {error_display} |"
        )

    return "\n".join(lines)


async def main() -> None:
    results = await run_eval()

    confusion_matrix_path: Optional[Path] = None
    try:
        confusion_matrix_path = render_confusion_matrix_png(results)
    except ImportError:
        print(
            "[eval_routing] matplotlib is not installed -- skipping the confusion "
            "matrix PNG (see agents/requirements.txt: `pip install matplotlib`). "
            "The markdown/JSON reports were still written.",
            file=sys.stderr,
        )

    report = render_markdown_report(results, confusion_matrix_path)

    out_dir = Path(__file__).resolve().parent
    md_path = out_dir / "eval_routing_results.md"
    json_path = out_dir / "eval_routing_results.json"
    md_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    misroutes = Counter(
        (r["expected_route"], r["actual_route"])
        for r in results
        if not r["correct"] and not r["error"]
    )

    print(f"\n[eval_routing] routing accuracy: {correct}/{total} ({correct / total:.0%})", file=sys.stderr)
    if misroutes:
        print("[eval_routing] most common misroutes (expected -> actual):", file=sys.stderr)
        for (expected, actual), count in misroutes.most_common(5):
            print(f"[eval_routing]   {count}x  {expected!r} -> {actual!r}", file=sys.stderr)
    print(f"[eval_routing] wrote {md_path}", file=sys.stderr)
    print(f"[eval_routing] wrote {json_path}", file=sys.stderr)
    if confusion_matrix_path is not None:
        print(f"[eval_routing] wrote {confusion_matrix_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

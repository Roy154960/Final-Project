"""
A SECOND MCP server, wrapping the ENTIRE agentic pipeline (Phases 2-4:
input_guard -> supervisor -> specialist(s) -> output_guard) as one tool,
`ask_multi_agent_rag`, rather than exposing the raw retrieve/
generate_answer primitives mcp_server/server.py already does.

Why a second server file rather than a third tool bolted onto
mcp_server/server.py: mcp_server/server.py is the Phase 1 deliverable --
raw retrieval, no routing, no guardrails, already verified and documented
as exactly that. Every specialist in this project talks to it as an MCP
CLIENT (see agents/mcp_client.py). Folding "run the whole agent" into
that same server would make one server spawn a subprocess of itself to
answer its own tool call -- confusing to reason about and to run -- and
it would blur what Phase 1's "two consumers, one server" screenshots are
actually proving. Kept as two separate processes instead, chained:

    Claude Code / OpenCode / Cursor
        -> (stdio) agent_mcp_server.py  [this file]
               -> agents.graph.ask()
                     -> (stdio, spawned fresh per call) mcp_server/server.py

Connecting to THIS server gets you the full guarded, routed system: a
prompt-injection attempt gets refused before routing ever runs, PII gets
redacted before an answer comes back, and the question gets routed to
whichever of retrieval_qa / corpus_meta / multi_hop the supervisor picks
-- not raw chunk retrieval. Connecting to mcp_server/server.py instead
gets you the raw primitives, with no routing or guardrails at all. Both
are legitimate things to expose; they answer different questions ("what's
in the corpus" vs. "what's the fully-guarded answer"), so this project
exposes both rather than picking one.

Known tradeoff, deliberately not engineered around here: `ask_multi_agent_rag`
calls `agents.graph.ask()`, which builds a fresh graph -- and therefore a
fresh MCP client, therefore a fresh subprocess of mcp_server/server.py --
on every single call. That's exactly what `python -m agents.graph`
already does, and it's already covered by five passing test suites
(test_specialists_smoke, test_supervisor_smoke, test_graph_smoke,
test_guardrails_smoke, test_eval_phase5_smoke), so reusing it here rather
than hand-rolling a cached/persistent version means this file has zero
new state-management code to get wrong. It does mean every call to this
tool pays the cost of re-spawning and re-warming the inner server
(embedder load, corpus snapshot) from scratch -- real latency, worth
noting in your report as a Part-2-style "what I'd fix next" rather than
solved here. A cached, module-level compiled graph (built once at server
startup, reused across calls) is the natural next step if that latency
turns out to matter in practice.

Run standalone (sanity check -- same as mcp_server/server.py's own):
    py -3.12 agents/agent_mcp_server.py
    python -m agents.agent_mcp_server

It'll sit waiting on stdin/stdout -- expected, not a hang. Ctrl+C to stop.

Prerequisites: identical to `python -m agents.graph` -- `ollama serve`
running with `llama3.2` pulled, and a corpus already ingested (the inner
mcp_server/server.py, spawned underneath this one, needs both to do
anything useful).
"""

import contextlib
import os
import sys
from pathlib import Path

# Same defensive env-var setup as mcp_server/server.py, for the same
# reason: stdio IS the MCP wire for this process too, and this process
# imports the same transitively-noisy dependency chain (via agents.graph
# -> agents.specialists -> langchain-mcp-adapters -> ... -> the inner
# server's own imports, once it's spawned as a subprocess).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@contextlib.contextmanager
def _stdout_to_stderr():
    """Identical in purpose and content to mcp_server/server.py's own
    copy -- kept as its own copy rather than imported, since importing
    from mcp_server.server would trigger THAT module's own top-level
    pipeline-component build (HFEmbedder, ChromaStore, ...) as a side
    effect of import, which this file has no business doing -- this
    server never touches the pipeline directly, only through a
    subprocess of that other file."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout


# agents/ is a direct child of the project root -- the same root
# agents/mcp_client.py already resolves mcp_server/server.py's absolute
# path against, so this file uses the identical derivation rather than a
# third, potentially-drifting copy of the same logic.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

with _stdout_to_stderr():
    from fastmcp import FastMCP  # noqa: E402

    from agents.graph import ask  # noqa: E402

mcp = FastMCP("multi-agent-rag-server")


# Names no specialist will ever use -- everything else that shows up as a
# named message in the final state must be a real specialist by
# construction of graph.py's node list (input_guard's own message is
# named "input_guard", the supervisor's meta-notes are named
# "supervisor"). Same definition and reasoning as agents/eval_phase5.py's
# own _META_NAMES -- kept as its own copy here rather than imported,
# since eval_phase5.py is a Phase-5 script this server has no other
# reason to depend on.
_META_NAMES = {"supervisor", "input_guard"}


def _summarize(result: dict) -> dict:
    """
    Reduce a full AgentState into the small, client-friendly shape this
    tool actually returns. An MCP client doesn't need LangChain message
    objects -- just the answer and enough routing metadata to be useful.

    Mirrors agents/eval_phase5.py's _extract_route_info, including the
    same fix: the real answer is the LAST SPECIALIST message, not
    necessarily state["messages"][-1]. A trailing supervisor meta-note
    (the iteration-cap partial-answer note, or the all-specialists-tried
    note) would otherwise get returned as "the answer" instead of the
    actual content underneath it -- see guardrails.py's confirmed fix
    and test_guardrails_smoke.py's
    test_full_graph_redacts_pii_when_it_is_not_the_last_message for the
    live run that surfaced this exact shape.
    """
    named = [
        (getattr(m, "name", None), m.content)
        for m in result.get("messages", [])
        if getattr(m, "name", None)
    ]
    specialist_messages = [(name, content) for name, content in named if name not in _META_NAMES]
    specialists_visited = [name for name, _ in specialist_messages]
    blocked = bool(result.get("blocked"))

    if blocked:
        answer = next((content for name, content in named if name == "input_guard"), "")
    elif specialist_messages:
        answer = specialist_messages[-1][1]
    else:
        answer = "(no answer produced)"

    return {
        "answer": answer,
        "blocked": blocked,
        "specialists_visited": specialists_visited,
        "iteration_count": result.get("iteration_count"),
    }


@mcp.tool()
async def ask_multi_agent_rag(question: str) -> dict:
    """
    Answer a question about the ingested art/painting-treatise corpus
    using the full multi-agent pipeline: an input guard screens the
    question for prompt-injection patterns before anything else runs; a
    supervisor routes it to whichever specialist fits (grounded Q&A over
    the corpus, corpus metadata/document listing, or multi-step
    decomposition for compound questions); an output guard redacts any
    structured PII (emails, phone numbers, etc.) before the answer comes
    back.

    Unlike a raw retrieval tool, this one can legitimately REFUSE to
    answer: if `blocked` comes back true, `answer` is a refusal
    explaining why, and no specialist or retrieval ever ran for that
    question.

    Args:
        question: A natural-language question. Works best for questions
            about the ingested corpus's actual content; a question about
            something the corpus doesn't cover will still get an answer
            back (not an error), but expect the specialist to say so
            rather than fabricate one.

    Returns:
        A dict with:
          - "answer": the final answer text (or, if blocked, the input
            guard's refusal text).
          - "blocked": true if the input guard refused this question
            before it ever reached routing.
          - "specialists_visited": ordered list of specialist names that
            ran this turn. Normally exactly one; more than one means the
            supervisor's repeat-route guard had to intervene mid-turn
            (see agents/README.md's documented repeat-route-guard
            limitation for why that happens with the current local model).
          - "iteration_count": how many supervisor routing decisions
            this turn took (capped at DEFAULT_ITERATION_CAP).
    """
    result = await ask(question)
    return _summarize(result)


if __name__ == "__main__":
    mcp.run()

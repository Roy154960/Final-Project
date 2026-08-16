"""
Structured request-id tracing across every LangGraph node (checklist
item: "structured request-id tracing across nodes -- no request_id, no
per-node JSON logging found anywhere in agents/").

traced_node() below is the ONE place this happens -- graph.py's own
build_graph() wraps every node (input_guard, contextualize, the
supervisor, every specialist, output_guard, refuse) with it at graph-
assembly time, rather than each node hand-rolling its own logging the
way the course's own Part 6 "What to Log" slide's `traced()` decorator
does inline per node. Centralizing it here means adding a new node to
graph.py gets tracing for free, and the log's shape can never drift
between nodes the way N independent copy-pasted decorators eventually
would.

Mirrors the course's own Part 6 `traced()` example almost exactly (one
request_id threaded through every node, one JSON line per visit,
latency in ms, what it routed to next) -- the one deliberate difference
is WHERE the line goes: the slide's version calls `log.info(...)` (i.e.
stderr/console), which is fine for a single live debugging session but
disappears the moment that terminal scrolls past it or the process
restarts. This writes to local_rag/logs/request_trace.jsonl instead (via
local_rag/usage_tracker.py's record_node_trace -- see that module's own
top docstring for why the actual file write lives there, not here) so a
session's full trace survives past one terminal window and stays
grep/jq-able afterward. Purely additive: this module never changes what
a node returns or raises, only observes it.
"""

import sys
import time
from pathlib import Path
from typing import Awaitable, Callable

from agents.state import AgentState


def _find_pipeline_root() -> Path:
    """Same duplicated-per-module helper agents/specialists.py,
    agents/guardrails.py, agents/api.py, and agents/llm_provider.py each
    already carry their own copy of -- see any of those modules' own
    docstrings for why this is duplicated rather than imported."""
    here = Path(__file__).resolve().parent  # agents/
    parent = here.parent
    grandparent = parent.parent
    candidates = [
        parent / "config.py",
        parent / "local_rag" / "config.py",
        grandparent / "config.py",
        grandparent / "local_rag" / "config.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.parent
    raise ModuleNotFoundError(
        "Could not find config.py near agents/. Checked:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nEdit _find_pipeline_root() in agents/tracing.py to add your actual path."
    )


_PIPELINE_ROOT = _find_pipeline_root()
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

import usage_tracker  # noqa: E402

NodeFn = Callable[[AgentState], Awaitable[dict]]


def new_request_id() -> str:
    """Re-exported from usage_tracker so agents/graph.py and
    agents/api.py never need their own direct import of a local_rag/
    module just for this one function -- everything agents/ needs from
    the tracing/usage layer is reachable through this module."""
    return usage_tracker.new_request_id()


def traced_node(node_name: str, fn: NodeFn) -> NodeFn:
    """
    Wrap one LangGraph node coroutine so every visit writes one
    structured JSON line to local_rag/logs/request_trace.jsonl -- see
    this module's own top docstring. Never changes the node's own
    behavior: the wrapped function still returns exactly what `fn`
    returned, and still raises exactly what `fn` raised (recorded first,
    then re-raised unchanged) -- graph.py's existing error handling
    (agents/api.py's `_invoke_turn`, specifically) sees the same
    exceptions it always did.
    """

    async def wrapper(state: AgentState) -> dict:
        request_id = state.get("request_id") or "no-request-id"
        thread_id = state.get("thread_id")
        t0 = time.perf_counter()
        try:
            out = await fn(state)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            usage_tracker.record_node_trace(
                request_id, node_name, elapsed_ms, thread_id=thread_id, error=repr(exc)
            )
            raise
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        route = out.get("route") if isinstance(out, dict) else None
        extra = {"thread_id": thread_id}
        if route:
            extra["route"] = route
        usage_tracker.record_node_trace(request_id, node_name, elapsed_ms, **extra)
        return out

    return wrapper

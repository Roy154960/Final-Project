"""
Phase 3 + Phase 4: assembles the supervisor (supervisor.py), the seven
specialists (specialists.py), the two Phase 4 guardrail nodes
(guardrails.py), and the turn-contextualization node (contextualize.py)
into one compiled, runnable LangGraph graph.

Shape:

    START -> input_guard --blocked-------------------------> refuse -> END
                 |
                 +--clean--> contextualize --> supervisor --route=="retrieval_qa"--> retrieval_qa -+
                                                   |         --route=="corpus_meta"--> corpus_meta   +--> back to supervisor
                                                   |         --route=="multi_hop"---> multi_hop     -+
                                                   |         ... (four more specialists, same shape)
                                                   +---------route=="FINISH"---> output_guard -> END

input_guard runs once, before the supervisor ever sees the question --
per the Sub-Project 2 spec's own Phase 4 wording, a flagged input is
routed straight to "refuse" instead of "supervisor", so a prompt-
injection attempt never costs a single routing decision or reaches any
specialist. output_guard runs once, after the supervisor has said
FINISH, immediately before END -- it never participates in the
routing loop itself, it only inspects (and if needed, rewrites) the
final answer on its way out. Neither guard calls an LLM; see
guardrails.py's module docstring for why that matters and for the
"replace in place, don't append" design of output_guard's redaction.

contextualize runs once, between a clean input_guard result and the
supervisor's first routing decision for the turn -- deliberately AFTER
input_guard (a flagged/blocked message is never worth spending a rewrite
call on) and BEFORE supervisor (every specialist's _last_human_text and
the supervisor's own _current_turn_context read "the latest HumanMessage"
completely unchanged; by the time either runs, contextualize has already
made that message standalone if it needed to be). It exists to close a
gap neither guardrail nor the original Phase 2/3 design touches: a
follow-up like "which size is best?" right after a question about
brushes carries no retrievable content on its own once it reaches
`retrieve` as a bare query string. See contextualize.py's own module
docstring for the full reasoning and its no-LLM-call fast path on a
turn's first-ever message.

In the DEFAULT case (no forced_route -- see state.py's docstring for
that field, and every existing caller that never sets it gets exactly
this), every specialist still edges straight back to "supervisor", never
to END directly and never to another specialist directly -- the
supervisor remains the only node that decides the turn is over
(route == "FINISH"), and the only node with a real cycle back into it,
which is what makes the iteration cap in supervisor.py meaningful: it
counts visits to this one node, not some looser notion of "steps." None
of input_guard, refuse, contextualize, or output_guard are part of that
cycle or counted against the cap -- each runs at most once per turn,
unconditionally, by construction of the edges above.

A caller CAN opt out of that whole loop for one turn by setting
`forced_route` to one specialist's name (agents/api.py exposes this as
ChatRequest.tool; graph.py's own ask() exposes it as a keyword arg; the
CLI at the bottom of this file exposes it as a third sys.argv). Two
edges change shape when that key is set, and ONLY when it's set to a
name that is actually one of this run's specialists (an unknown name is
treated exactly like "not set" -- see `_resolve_forced_route` -- the
same "invalid input degrades to the default behavior instead of
erroring" preference supervisor.py's own four safety nets already use):

    contextualize --(forced_route=="image_qa")--> image_qa --> output_guard -> END
    contextualize --(forced_route unset/invalid)--> supervisor (unchanged loop)

That is: the named specialist runs exactly once, on the contextualized
question, and its answer goes straight out through output_guard --
the supervisor's own LLM never makes a routing decision at all for that
turn, so the turn can never expand into a second specialist the way a
normal supervisor-routed turn sometimes does. This is deliberately an
isolation tool (e.g. "does image_qa alone, with no supervisor in the
loop, handle this question correctly?"), not a performance shortcut or a
replacement for normal use -- the supervisor's own multi-specialist
routing, safety nets, and re-route loop are what Sub-Project 2's spec
actually asks the graph to do by default, and remain exactly what every
caller gets unless they explicitly ask to bypass it for one turn.

This module owns none of the routing logic, the guardrail logic, or the
contextualization logic itself -- it only wires already-built nodes
together, exactly as this file's own Phase 3 docstring said Phase 4
would do it (that prediction held again here: contextualize.py's
addition is one new node and one repointed conditional-edge target,
same small-diff shape as input_guard/output_guard's own addition; the
forced_route bypass is likewise two conditional-edge functions added
around the existing wiring, not a rewrite of it).
"""

import asyncio
import sys
from typing import Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.contextualize import build_contextualize_node
from agents.guardrails import input_guard_node, output_guard_node, refuse_node
from agents.specialists import build_specialists
from agents.state import AgentState
from agents.supervisor import (
    DEFAULT_FALLBACK_ROUTE,
    DEFAULT_ITERATION_CAP,
    DEFAULT_ROUTE_FORMAT,
    build_supervisor,
)
from agents.tracing import new_request_id, traced_node


def _resolve_forced_route(state: AgentState, specialist_names: frozenset) -> Optional[str]:
    """
    state.get("forced_route") if -- and only if -- it names one of THIS
    run's actual specialists; None otherwise (key absent, key explicitly
    None, or a name that doesn't match, e.g. a stale value left over from
    a differently-configured run, or "FINISH"/"supervisor" typo'd in by a
    caller). Never raises on a bad value -- same "an invalid override
    degrades to the default behavior" choice supervisor.py's own
    known_routes membership check makes for the model's routing output,
    applied here to a human/API caller's input instead.

    Centralized in one function (used by both conditional edges below)
    so the two places that need this exact same check can't drift out of
    sync with each other the way `path_map` construction elsewhere in
    this file already avoids drifting from `specialists`'s own keys.
    """
    forced = state.get("forced_route")
    return forced if forced in specialist_names else None


async def build_graph(
    iteration_cap: int = DEFAULT_ITERATION_CAP,
    fallback_route: str = DEFAULT_FALLBACK_ROUTE,
    route_format: Literal["json_schema", "json"] = DEFAULT_ROUTE_FORMAT,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """
    Build one shared MCP client + specialist set (build_specialists(), see
    its own docstring for why this is a one-per-run call), wrap it with a
    supervisor, and compile the graph. Call this once per conversation /
    per graph run -- same granularity build_specialists() already
    requires, for the same reason (one live server process, one BM25
    snapshot, one corpus snapshot shared by every specialist AND now the
    supervisor's routing decisions within that run).

    `route_format` is passed straight through to build_supervisor() --
    see DEFAULT_ROUTE_FORMAT's docstring in supervisor.py for what it's
    for and why it's worth A/B testing on a live run.

    `checkpointer` is None by default, which is what ask() below and
    every existing test/CLI caller wants -- a fresh, stateless graph per
    call, no persisted state. Passing a real BaseCheckpointSaver (e.g.
    LangGraph's AsyncSqliteSaver) turns on LangGraph's own thread-scoped
    memory: a caller that then invokes the SAME compiled graph object
    repeatedly with the SAME `config={"configurable": {"thread_id": ...}}`
    gets `messages` accumulated across those calls via state.py's
    add_messages reducer, instead of each call starting from an empty
    list. agents/api.py is that caller -- see its module docstring for
    why a persisted `messages` history across turns matters for more
    than just UX: the `invoice` specialist (specialists.py) reads PAST
    `product_search` messages out of state["messages"] to know what to
    invoice, which without a checkpointer only ever works within one
    ask() call's own single turn.
    """
    specialists = await build_specialists()
    supervisor_node = build_supervisor(
        specialists,
        iteration_cap=iteration_cap,
        fallback_route=fallback_route,
        route_format=route_format,
    )
    contextualize_node = build_contextualize_node()

    # Every node added below is wrapped with traced_node()
    # (agents/tracing.py) before add_node ever sees it -- one structured
    # JSON line per node visit, written to
    # local_rag/logs/request_trace.jsonl, keyed by this turn's
    # state["request_id"] (set once per turn by agents/api.py's
    # `_new_turn_state`/retry/edit, or by this file's own ask() for the
    # CLI/eval-script case below). See tracing.py's own module docstring
    # for why this wrapping happens HERE, at graph-assembly time, rather
    # than inside each individual node function -- it's what makes
    # tracing apply uniformly to every node (guards, contextualize, the
    # supervisor, every specialist, output_guard) without touching any
    # of their own implementations, and what makes a future new node get
    # traced automatically just by going through add_node the normal way.
    builder = StateGraph(AgentState)
    builder.add_node("input_guard", traced_node("input_guard", input_guard_node))
    builder.add_node("refuse", traced_node("refuse", refuse_node))
    builder.add_node("contextualize", traced_node("contextualize", contextualize_node))
    builder.add_node("supervisor", traced_node("supervisor", supervisor_node))
    for name, node_fn in specialists.items():
        builder.add_node(name, traced_node(name, node_fn))
    builder.add_node("output_guard", traced_node("output_guard", output_guard_node))

    # Every turn starts at the guard, never directly at the supervisor --
    # see guardrails.py's module docstring for why a flagged turn must
    # never reach routing at all, not just be routed normally and hoped
    # to fail downstream. A clean turn now goes through contextualize
    # first, not straight to supervisor -- see contextualize.py's own
    # module docstring for why that ordering (after the guard, before
    # routing) is deliberate.
    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges(
        "input_guard",
        lambda state: "refuse" if state.get("blocked") else "contextualize",
        {"refuse": "refuse", "contextualize": "contextualize"},
    )
    builder.add_edge("refuse", END)

    # specialist_name_set is frozen once here and closed over by both
    # conditional-edge functions below, rather than recomputed from
    # `specialists` on every single graph step -- the dict itself never
    # changes after build_specialists() returns it, so there's nothing to
    # gain by re-deriving this on every contextualize/specialist visit.
    # A frozenset (not the ordered tuple stashed on the compiled graph
    # below) specifically for O(1) membership checks on the hot path --
    # this function runs on every contextualize visit and every
    # specialist exit, once per turn each.
    specialist_name_set = frozenset(specialists)

    # Normal case: state.get("forced_route") resolves to None (unset, or
    # invalid -- see _resolve_forced_route), so this lands on "supervisor"
    # exactly as it always has -- the supervisor's own loop picks among
    # every specialist, same as before forced_route existed at all.
    # Override case: a caller-supplied forced_route names a real
    # specialist, so contextualize hands off directly to THAT node,
    # skipping the supervisor's routing decision for this turn entirely.
    builder.add_conditional_edges(
        "contextualize",
        lambda state: _resolve_forced_route(state, specialist_name_set) or "supervisor",
        {**{name: name for name in specialists}, "supervisor": "supervisor"},
    )

    # path_map's specialist keys are exactly known_routes from
    # supervisor.py (the specialists dict's own keys) -- built here from
    # the same `specialists` dict rather than hardcoded a third time, so
    # adding or renaming a specialist can't leave this map out of sync
    # with either build_specialists()'s dict or supervisor.py's
    # known_routes check. "FINISH" now targets "output_guard" instead of
    # END directly -- the one line that actually changed from Phase 3's
    # version of this map.
    path_map: dict[str, str] = {name: name for name in specialists}
    path_map["FINISH"] = "output_guard"
    builder.add_conditional_edges("supervisor", lambda state: state["route"], path_map)

    # Every specialist normally loops back to "supervisor" (unchanged
    # default behavior). The ONE exception: a turn that reached this
    # specialist via a forced_route bypass (contextualize routed here
    # directly, never through the supervisor at all) goes straight to
    # "output_guard" instead -- the whole point of forced_route is to
    # isolate exactly one specialist's answer for the turn, so handing
    # back to the supervisor afterward (which could then route to a
    # SECOND specialist, defeating that isolation) would be wrong here
    # specifically. `_resolve_forced_route` is reused rather than a
    # simpler `bool(state.get("forced_route"))` check so a stale/invalid
    # forced_route value is treated the same way on the way OUT of a
    # specialist as it already is on the way IN via contextualize.
    for name in specialists:
        builder.add_conditional_edges(
            name,
            lambda state, _n=name: (
                "output_guard"
                if _resolve_forced_route(state, specialist_name_set) == _n
                else "supervisor"
            ),
            {"supervisor": "supervisor", "output_guard": "output_guard"},
        )

    builder.add_edge("output_guard", END)

    compiled = builder.compile(checkpointer=checkpointer)
    # Stashed on the compiled graph itself (CompiledStateGraph tolerates
    # arbitrary extra attributes -- confirmed directly rather than
    # assumed, same "check it against the real thing" habit this
    # project's other modules follow) so a caller that only has the
    # compiled graph object -- agents/api.py, specifically, which never
    # sees `specialists` itself -- can validate a caller-supplied
    # forced_route against the SAME name set this graph was actually
    # built with, without importing supervisor.py's hardcoded
    # RouteDecision Literal (which is a second, separately-maintained
    # copy of these names, not the source of truth `specialists` already
    # is everywhere else in this file) or re-running build_specialists()
    # a second time just to get its keys. A tuple, not the frozenset
    # used internally above -- this one is for DISPLAY (GET /tools lists
    # it in order) and `in` on a ~7-element tuple costs nothing measurable
    # at one check per HTTP request, so there's no reason to give up
    # `specialists`'s own build order just to match the hot-path set.
    compiled.known_specialist_names = tuple(specialists)
    return compiled


async def ask(
    question: str,
    iteration_cap: int = DEFAULT_ITERATION_CAP,
    fallback_route: str = DEFAULT_FALLBACK_ROUTE,
    route_format: Literal["json_schema", "json"] = DEFAULT_ROUTE_FORMAT,
    forced_route: Optional[str] = None,
) -> dict:
    """
    Convenience one-shot entry point for manual testing and for Phase 5's
    eval script: build a fresh graph, run exactly one question through
    it, return the final AgentState.

    `forced_route`, when given a valid specialist name, bypasses the
    supervisor's own routing for this one call and runs ONLY that
    specialist -- see graph.py's own module docstring and
    _resolve_forced_route for exactly what that does and doesn't change.
    None (the default) is the normal supervisor-routed behavior this
    function has always had; an unrecognized name is silently treated the
    same as None rather than raising, consistent with every other
    caller-input validation in this project.

    recursion_limit is set generously above what iteration_cap could ever
    need (each supervisor visit is one graph "step", each specialist call
    is one more) so a legitimately high iteration_cap doesn't trip
    LangGraph's own unrelated recursion guard before this graph's own cap
    logic in supervisor.py ever gets a chance to fire.
    """
    graph = await build_graph(
        iteration_cap=iteration_cap, fallback_route=fallback_route, route_format=route_format
    )
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "route": None,
        "iteration_count": 0,
        "blocked": False,
        "injection_patterns": [],
        "forced_route": forced_route,
        "request_id": new_request_id(),
    }
    return await graph.ainvoke(
        initial_state, config={"recursion_limit": max(25, iteration_cap * 4)}
    )


if __name__ == "__main__":
    # Second CLI arg (optional) selects DEFAULT_ROUTE_FORMAT's A/B knob --
    # e.g. `python -m agents.graph "What is glazing?" json` runs the
    # supervisor without Ollama's schema-constrained structured output,
    # to compare against the default `json_schema` mode's behavior on a
    # repeated live run of the same question. See supervisor.py's
    # DEFAULT_ROUTE_FORMAT docstring for what this is testing.
    #
    # Third CLI arg (optional) is forced_route -- e.g.
    # `python -m agents.graph "What does this brush look like?" json_schema image_qa`
    # runs ONLY image_qa, bypassing the supervisor's own routing for this
    # call, for exercising one specialist directly. Omit it (or pass the
    # literal word "auto") for the normal, default, all-specialists
    # supervisor-routed behavior this CLI has always had.
    question = sys.argv[1] if len(sys.argv) > 1 else "What is glazing in oil painting?"
    route_format = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ROUTE_FORMAT
    if route_format not in ("json_schema", "json"):
        raise SystemExit(f"route_format must be 'json_schema' or 'json', got {route_format!r}")
    forced_route_arg = sys.argv[3] if len(sys.argv) > 3 else None
    if forced_route_arg == "auto":
        forced_route_arg = None

    result = asyncio.run(ask(question, route_format=route_format, forced_route=forced_route_arg))

    # Print every named message (each specialist's answer, plus any
    # supervisor notes), not just the last one -- in a partial-answer or
    # repeat-route-guard run, the last message is the supervisor's own
    # short meta-note, and the substantive answer worth reading sits in
    # the specialist message just before it. Printing only the last
    # message hides that answer entirely, which is exactly what happened
    # the first time this was run live: the terminal only showed the
    # cap-reached note, never retrieval_qa's actual (possibly fine)
    # answer underneath it.
    for msg in result["messages"]:
        name = getattr(msg, "name", None)
        if name:
            print(f"--- {name} ---")
            print(msg.content)
            print()

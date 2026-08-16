"""
Shared LangGraph state schema for the whole agentic system.

Defined once here, in Phase 2, so that Phase 3 (the supervisor) doesn't
force a signature change on every specialist node that already exists.
Every specialist in specialists.py reads state["messages"] and returns a
partial update shaped like this schema. Phase 4 (guardrails.py) adds two
more fields, both read/written only by the two guardrail nodes -- no
existing specialist or the supervisor needs to change to accommodate
them, same forward-compatibility intent `route`/`iteration_count` were
added with back in Phase 2.
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    messages: the running conversation, reduced with LangGraph's standard
        add_messages (append-only, dedups by message id on replay). Every
        specialist reads the latest HumanMessage as its input question and
        returns a new AIMessage as its answer. guardrails.py's
        output_guard node relies on the dedup-by-id behavior specifically:
        it replaces a flagged AIMessage in place by returning a new
        message with the same `.id`, rather than appending a second one.

    route: the supervisor's last validated routing decision, OR
        "FINISH" set directly by guardrails.py's refuse_node when
        input_guard blocks a turn before the supervisor ever runs. Not
        read or written by any specialist.

    iteration_count: incremented by the supervisor before each routing
        decision (Phase 3), checked against the iteration cap there.
        Specialists never read or write this field. Stays at its initial
        value (0) on a turn that input_guard blocks, since the supervisor
        never runs on that path -- a useful signal on its own when
        reading a result: iteration_count == 0 with blocked == True means
        the turn never reached routing at all.

    blocked: set by guardrails.py's input_guard node -- True if the
        latest HumanMessage matched a known prompt-injection pattern.
        graph.py's conditional edge out of "input_guard" reads this field
        directly to choose between "refuse" and "supervisor". Not read or
        written by the supervisor or any specialist; defaults to False
        via .get("blocked", False) wherever it's read, so existing code
        that builds an initial state dict without this key (e.g. earlier
        Phase 3 test fixtures) keeps working unchanged.

    injection_patterns: set alongside `blocked` -- the list of matched
        pattern strings from local_rag/safety/prompt_injection.py's
        scan_for_injection, kept for diagnostics/logging (e.g. an eval
        table's "what went wrong" column in Phase 5) rather than only
        printed to stderr and then lost. Empty list when blocked is
        False.

    forced_route: caller-supplied override that skips the supervisor's
        OWN routing decision for this turn and sends the (contextualized)
        question straight to one named specialist -- e.g. for manually
        exercising `image_qa` in isolation without depending on the
        supervisor's LLM picking it. None (the default, and what every
        existing caller that never sets this key gets via .get()) means
        the turn runs exactly as it always has: contextualize -> the
        supervisor's own loop, free to pick among every specialist and
        re-route as many times as the iteration cap allows. Only read by
        graph.py's conditional edges (the branch out of "contextualize",
        and the branch out of each specialist back to either "supervisor"
        or straight to "output_guard"); no specialist and no part of
        supervisor.py's own routing logic reads or writes it. Like
        `route`/`iteration_count`/`blocked`/`injection_patterns`, it has
        no reducer and is expected to be reset every turn by the caller
        (agents/api.py's `_new_turn_state` does this) rather than
        persisted across turns by the checkpointer -- a forced tool
        should apply to the one turn that asked for it, not silently keep
        overriding the supervisor on every later turn of the same thread.

    thread_id: the checkpointer's own thread id for this conversation,
        set every turn by agents/api.py's `_new_turn_state` (and its
        retry/edit siblings) from the SAME thread_id value already used
        to build the LangGraph `config` for that call -- never read from
        anywhere inside the graph itself, and never guessable by a model,
        since it's what scopes the personal_docs specialist's search to
        THIS conversation's own uploads (local_rag/personal_rag.py's
        "temp" collection, filtered by thread_id) and nobody else's. Like
        `forced_route`, has no reducer and needs no state-carrying logic
        beyond "the caller resupplies it every turn" -- it never changes
        within a conversation, so resupplying the same value each turn is
        just consistency with how every other per-turn field here is
        handled, not a real reset. None only for callers that never set
        it at all (graph.py's own ask() / the CLI / the Phase 5 eval
        script, none of which have a persisted thread or any personal
        uploads to scope to) -- personal_docs_node treats a missing
        thread_id as "nothing to search," the same "degrade, don't
        raise" convention every other specialist here follows for a
        missing tool or an empty corpus.

    request_id: a short id generated ONCE per turn (agents/api.py's
        `_new_turn_state` and its retry/edit siblings; agents/graph.py's
        own ask() for the CLI/eval-script case) and resupplied unchanged
        on every call the same way `thread_id` above already is -- same
        "caller resupplies it every turn, no reducer needed" pattern.
        Read only by agents/tracing.py's traced_node() wrapper
        (agents/graph.py wraps every node with it at build time), which
        writes one JSON line per node visit to
        local_rag/logs/request_trace.jsonl carrying this id -- see that
        module's own docstring for why this exists and
        local_rag/usage_tracker.py's own top docstring for why that log
        is dev-only, never surfaced to an HTTP response body. No
        specialist and no part of supervisor.py's own routing logic reads
        or writes it. None only for callers that never set it at all (a
        test fixture predating this field, say) -- traced_node() degrades
        to logging "no-request-id" rather than raising in that case.
    """

    messages: Annotated[list, add_messages]
    route: Optional[str]
    iteration_count: int
    blocked: Optional[bool]
    injection_patterns: Optional[list[str]]
    forced_route: Optional[str]
    thread_id: Optional[str]
    request_id: Optional[str]

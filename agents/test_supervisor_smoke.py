"""
Smoke test for agents/supervisor.py -- fake ChatOllama, real everything
else: real RouteDecision schema validation, real membership-check logic,
real iteration-cap arithmetic, real transcript formatting. Same
philosophy as test_specialists_smoke.py (see its own docstring): no
Ollama server or real corpus needed, fast, catches wiring and
validated-routing bugs before they cost a real model call to discover.

Run with:
    python agents/test_supervisor_smoke.py
    (or, from the project root: python -m agents.test_supervisor_smoke)
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agents import supervisor
from agents.prompts import SPECIALIST_ROUTING_EXAMPLES
from agents.state import AgentState


# ---------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------

class ScriptedRouterLLM:
    """
    Stand-in for ChatOllama: returns a scripted queue of raw `.content`
    strings, one per call to ainvoke, and counts how many times it was
    actually called -- several tests below assert this stays at 0 (the
    iteration-cap path must not spend a model call once the cap is hit).

    Also records every `messages` list it was called with, in order --
    used by test_transcript_lives_in_human_turn_not_system_prompt to lock
    in the message-role fix directly (see that test's docstring for the
    live-run failure it guards against regressing).
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        content = self._responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(content=content)


async def _fake_specialist(state: AgentState) -> dict:
    """Placeholder node body -- supervisor.py never calls a specialist's
    body itself, it only ever routes to one by name, so this is never
    actually invoked by anything under test here."""
    return {"messages": []}


FAKE_SPECIALISTS = {
    "retrieval_qa": _fake_specialist,
    "corpus_meta": _fake_specialist,
    "multi_hop": _fake_specialist,
}


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


def _build(llm_responses, specialists=None, **kwargs):
    """Patch ChatOllama to return a ScriptedRouterLLM, then build a
    supervisor node against it. Returns (node_fn, fake_llm) so tests can
    inspect fake_llm.call_count afterward."""
    specialists = specialists if specialists is not None else FAKE_SPECIALISTS
    fake_llm = ScriptedRouterLLM(llm_responses)
    with patch("agents.supervisor.ChatOllama", return_value=fake_llm):
        node = supervisor.build_supervisor(specialists, **kwargs)
    return node, fake_llm


def _state(messages, iteration_count=0) -> AgentState:
    return {"messages": messages, "route": None, "iteration_count": iteration_count}


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

async def test_first_call_picks_a_valid_specialist():
    print("\n=== first call: no specialist has answered yet ===")
    node, llm = _build(['{"route": "retrieval_qa"}'])
    state = _state([HumanMessage(content="What is glazing?")])

    result = await node(state)

    _check("route is the specialist the (fake) model chose", result["route"] == "retrieval_qa")
    _check("iteration_count incremented from 0 to 1", result["iteration_count"] == 1)
    _check("the LLM was actually called once", llm.call_count == 1)


async def test_finish_after_specialist_already_answered():
    print("\n=== supervisor decides FINISH once a specialist has answered ===")
    node, llm = _build(['{"route": "FINISH"}'])
    state = _state(
        [
            HumanMessage(content="What is glazing?"),
            AIMessage(content="Glazing is ... [source.pdf]", name="retrieval_qa"),
        ],
        iteration_count=1,
    )

    result = await node(state)

    _check("route is FINISH", result["route"] == "FINISH")
    _check("iteration_count incremented from 1 to 2", result["iteration_count"] == 2)


async def test_premature_finish_before_any_specialist_is_overridden():
    print(
        "\n=== safety net 4: FINISH before any specialist has answered is "
        "overridden, not accepted (reproduces a real live-run failure) ==="
    )
    # Reproduces exactly what a real Phase 5 eval run showed for an
    # out-of-scope query ("What's a good recipe for chocolate chip
    # cookies?"): the supervisor's very first raw output was
    # '{"route": "FINISH"}', with zero specialists having run yet. Before
    # this net existed, that was accepted outright (FINISH is a
    # schema-valid, known route, and the repeat-route guard only fires on
    # non-FINISH repeats) -- no AIMessage was ever appended, and the turn
    # ended with literally no answer for the user to read.
    node, llm = _build(['{"route": "FINISH"}'])
    state = _state([HumanMessage(content="What's a good recipe for chocolate chip cookies?")])

    result = await node(state)

    _check(
        "route is NOT accepted as FINISH -- overridden to a real specialist instead",
        result["route"] != "FINISH",
    )
    _check(
        "overridden to the first specialist in build order (retrieval_qa), "
        "the same as _next_untried_route would pick with nothing yet tried",
        result["route"] == "retrieval_qa",
    )
    _check("iteration_count still incremented normally", result["iteration_count"] == 1)
    _check("no extra 'all tried' note appended -- this isn't that condition", result["messages"] == [])
    _check("the LLM was called exactly once", llm.call_count == 1)


async def test_malformed_json_falls_back_safely():
    print("\n=== malformed JSON from the model falls back, does not raise ===")
    node, llm = _build(["this is not json at all"])
    state = _state([HumanMessage(content="What is glazing?")])

    result = await node(state)

    _check(
        "falls back to the default fallback route instead of raising",
        result["route"] == supervisor.DEFAULT_FALLBACK_ROUTE,
    )
    _check("iteration_count still incremented despite the fallback", result["iteration_count"] == 1)


async def test_schema_rejects_out_of_enum_route():
    print("\n=== a route name outside RouteDecision's Literal fails schema validation ===")
    node, llm = _build(['{"route": "made_up_specialist"}'])
    state = _state([HumanMessage(content="What is glazing?")])

    result = await node(state)

    _check(
        "Pydantic's Literal rejects the hallucinated name, falls back safely",
        result["route"] == supervisor.DEFAULT_FALLBACK_ROUTE,
    )


async def test_unknown_specialist_route_is_rejected_by_the_live_schema():
    print(
        "\n=== a route naming a specialist NOT in this build's own dict is "
        "rejected by the (now per-build) schema itself ==="
    )
    # This scenario used to need safety net 2 (a SEPARATE membership
    # check) to catch, because RouteDecision's Literal used to be a
    # single, hand-maintained module-level constant that could silently
    # drift out of sync with whichever specialists dict a given
    # build_supervisor() call was actually handed -- see
    # supervisor.py's own module docstring and
    # _build_route_decision_model's docstring for the CONFIRMED bug that
    # caused (color_palette added to specialists.py without the old
    # hardcoded Literal being updated to match, so Ollama's own
    # structured-output constraint could never generate that route at
    # all). RouteDecision is now built FRESH, every call to
    # build_supervisor(), from THIS dict's own keys -- so a route for a
    # specialist this particular build doesn't have is rejected by
    # SCHEMA validation itself (net 1), not by a separate backstop. Net
    # 2 is kept anyway as defense-in-depth -- see its own comment in
    # supervisor.py -- but is now structurally redundant with net 1 by
    # construction, so this test can no longer distinguish which of the
    # two caught it; it only asserts the end-to-end safety property both
    # nets exist to guarantee still holds.
    reduced_specialists = {"retrieval_qa": _fake_specialist, "corpus_meta": _fake_specialist}
    node, llm = _build(['{"route": "multi_hop"}'], specialists=reduced_specialists)
    state = _state([HumanMessage(content="Compare tempera and oil glazing")])

    result = await node(state)

    _check(
        "a route naming a specialist this build doesn't have still falls back safely",
        result["route"] == supervisor.DEFAULT_FALLBACK_ROUTE,
    )


async def test_route_schema_is_rebuilt_fresh_from_each_build_not_hardcoded():
    print(
        "\n=== regression test for the confirmed bug: the route schema is built "
        "fresh from THIS call's specialists dict, not a hand-maintained constant ==="
    )
    # Directly locks in the fix: a specialist that exists ONLY in this
    # particular build's dict (not in FAKE_SPECIALISTS, and -- the whole
    # point -- not in any hardcoded list anywhere in supervisor.py) must
    # still be a schema-VALID route the model can choose directly. Before
    # the fix, a route like this could only ever be reached via the
    # repeat-route guard's fallback walk, never as the model's own first
    # choice -- this test would have failed against the old code.
    extra_specialists = {**FAKE_SPECIALISTS, "color_palette": _fake_specialist}
    node, llm = _build(['{"route": "color_palette"}'], specialists=extra_specialists)
    state = _state([HumanMessage(content="Give me a complementary palette for cerulean blue")])

    result = await node(state)

    _check(
        "a specialist that only exists in THIS build's dict is a valid schema "
        "route the model's own choice survives untouched through",
        result["route"] == "color_palette",
    )
    _check("the LLM was called exactly once (no fallback walk needed)", llm.call_count == 1)


async def test_routing_examples_only_include_this_builds_specialists():
    print(
        "\n=== the rendered system prompt's worked examples are filtered to "
        "THIS build's own specialists, same as specialist_descriptions ==="
    )
    node, llm = _build(['{"route": "retrieval_qa"}'])
    state = _state([HumanMessage(content="What is glazing?")])
    await node(state)

    system_msg = llm.calls[0][0].content
    _check(
        "the worked example for a specialist THIS build has (retrieval_qa) is present",
        SPECIALIST_ROUTING_EXAMPLES["retrieval_qa"] in system_msg,
    )
    _check(
        "the worked example for a specialist THIS build does NOT have "
        "(color_palette) is absent -- note this checks for the EXACT example "
        "line, not a bare '-> color_palette' substring, since the static "
        "'Specific routing distinctions' prose above the examples list "
        "legitimately mentions color_palette by name regardless of which "
        "specialists this particular build actually has",
        SPECIALIST_ROUTING_EXAMPLES["color_palette"] not in system_msg,
    )


async def test_looks_like_refusal_recognizes_color_palette_error_wording():
    print("\n=== _looks_like_refusal recognizes color_palette's own error phrasing ===")
    _check(
        "an unrecognized-color error is recognized as a refusal",
        supervisor._looks_like_refusal("I couldn't recognize 'xyz' as a color -- try a hex code..."),
    )
    _check(
        "an unrecognized-mood error is recognized as a refusal",
        supervisor._looks_like_refusal("I couldn't connect 'xyz' to a color -- try naming..."),
    )
    _check(
        "a genuine, confident color_palette answer is NOT flagged as a refusal",
        not supervisor._looks_like_refusal("**Base color:** Cerulean -- `#007ba7` ..."),
    )


async def test_iteration_cap_with_no_prior_specialist():
    print("\n=== iteration cap reached before any specialist ever answered ===")
    node, llm = _build([], iteration_cap=2)
    state = _state([HumanMessage(content="What is glazing?")], iteration_count=2)

    result = await node(state)

    _check("route is forced to FINISH", result["route"] == "FINISH")
    _check("no LLM call was spent once the cap was already exceeded", llm.call_count == 0)
    note = result["messages"][0].content
    _check(
        "partial-answer note explains no specialist ever got to answer",
        "before any specialist could answer" in note,
    )


async def test_iteration_cap_with_prior_specialist_answer():
    print("\n=== iteration cap reached after at least one specialist already answered ===")
    node, llm = _build([], iteration_cap=2)
    state = _state(
        [
            HumanMessage(content="What is glazing?"),
            AIMessage(content="Partial grounded answer.", name="retrieval_qa"),
        ],
        iteration_count=2,
    )

    result = await node(state)

    _check("route is forced to FINISH", result["route"] == "FINISH")
    _check("no LLM call was spent", llm.call_count == 0)
    note = result["messages"][0].content
    _check("partial-answer note names the specialist that was tried", "retrieval_qa" in note)
    _check("note explicitly calls the answer partial", "Partial answer" in note)


async def test_build_supervisor_rejects_invalid_fallback_route():
    print("\n=== build_supervisor() refuses a fallback_route that isn't a real specialist ===")
    raised = False
    try:
        _build(['{"route": "retrieval_qa"}'], fallback_route="not_a_real_specialist")
    except ValueError:
        raised = True
    _check("ValueError raised at build time, not at request time", raised)


async def test_repeat_route_guard_redirects_to_untried_specialist():
    print("\n=== repeat-route guard: model repeats an already-tried specialist ===")
    # retrieval_qa already answered; the (fake) model says retrieval_qa
    # again despite SUPERVISOR_SYSTEM_PROMPT's rule against it -- this is
    # the exact failure mode a live run surfaced (llama3.2 re-picking the
    # same specialist instead of following the don't-repeat instruction).
    node, llm = _build(['{"route": "retrieval_qa"}'])
    state = _state(
        [
            HumanMessage(content="What is glazing?"),
            AIMessage(content="Partial or unclear answer.", name="retrieval_qa"),
        ],
        iteration_count=1,
    )

    result = await node(state)

    _check(
        "guard overrides the repeat, routing to a different untried specialist "
        "(dict order: corpus_meta comes after retrieval_qa)",
        result["route"] == "corpus_meta",
    )
    _check("no extra message appended for a plain redirect", result.get("messages", []) == [])


async def test_repeat_route_guard_finishes_when_every_specialist_tried():
    print("\n=== repeat-route guard: every specialist already tried, model still repeats one ===")
    node, llm = _build(['{"route": "retrieval_qa"}'])
    state = _state(
        [
            HumanMessage(content="What is glazing?"),
            AIMessage(content="answer 1", name="retrieval_qa"),
            AIMessage(content="answer 2", name="corpus_meta"),
            AIMessage(content="answer 3", name="multi_hop"),
        ],
        iteration_count=3,
    )

    result = await node(state)

    _check("forced to FINISH once nothing untried remains", result["route"] == "FINISH")
    _check("an explanatory note is appended", len(result["messages"]) == 1)
    note = result["messages"][0].content
    _check("note explains every specialist was already tried", "already tried this turn" in note)
    _check(
        "note lists all three specialists that were tried",
        all(name in note for name in ("retrieval_qa", "corpus_meta", "multi_hop")),
    )


async def test_finish_still_works_even_with_prior_attempts():
    print("\n=== sanity check: a genuine FINISH decision is never treated as a repeat ===")
    node, llm = _build(['{"route": "FINISH"}'])
    state = _state(
        [
            HumanMessage(content="What is glazing?"),
            AIMessage(content="A confident, complete answer.", name="retrieval_qa"),
        ],
        iteration_count=1,
    )

    result = await node(state)

    _check("FINISH passes through untouched, not caught by the repeat guard", result["route"] == "FINISH")
    _check("no extra message appended", result.get("messages", []) == [])


async def test_transcript_lives_in_human_turn_not_system_prompt():
    print("\n=== regression: transcript travels in the human turn, system prompt stays static ===")
    # Reproduces the exact live-run diagnosis: with the transcript
    # embedded inside the system prompt, a real supervisor returned the
    # identical route on every call regardless of what had already been
    # tried, because the only content that ever varied (the transcript)
    # sat in a message role the model wasn't reliably attending to. The
    # fix moved it to the human turn instead -- this test asserts that
    # placement directly, at the message-list level, so a future edit
    # can't silently move it back without a test noticing.
    node, llm = _build(['{"route": "corpus_meta"}', '{"route": "FINISH"}'])

    state1 = _state([HumanMessage(content="What is glazing?")])
    await node(state1)

    state2 = _state(
        [
            HumanMessage(content="What is glazing?"),
            AIMessage(content="Some answer.", name="retrieval_qa"),
        ],
        iteration_count=1,
    )
    await node(state2)

    _check("the LLM was called exactly twice", len(llm.calls) == 2)

    system_msg_1 = llm.calls[0][0].content
    system_msg_2 = llm.calls[1][0].content
    _check(
        "system prompt is byte-identical across calls (fully static, no per-call transcript)",
        system_msg_1 == system_msg_2,
    )
    _check(
        "system prompt does not contain the dynamic transcript's per-attempt "
        "line format (only its own static routing-rules prose, which "
        "separately and legitimately mentions 'already answered')",
        'answered: "' not in system_msg_1,
    )

    human_msg_1 = llm.calls[0][1].content
    human_msg_2 = llm.calls[1][1].content
    _check(
        "first call's human turn reflects that nothing has answered yet",
        "No specialist has answered yet" in human_msg_1,
    )
    _check(
        "second call's human turn reflects that retrieval_qa already answered",
        "retrieval_qa already answered" in human_msg_2,
    )
    _check("the two calls' human turns differ (this is the actual fix under test)", human_msg_1 != human_msg_2)


async def main():
    await test_first_call_picks_a_valid_specialist()
    await test_finish_after_specialist_already_answered()
    await test_premature_finish_before_any_specialist_is_overridden()
    await test_malformed_json_falls_back_safely()
    await test_schema_rejects_out_of_enum_route()
    await test_unknown_specialist_route_is_rejected_by_the_live_schema()
    await test_route_schema_is_rebuilt_fresh_from_each_build_not_hardcoded()
    await test_routing_examples_only_include_this_builds_specialists()
    await test_looks_like_refusal_recognizes_color_palette_error_wording()
    await test_iteration_cap_with_no_prior_specialist()
    await test_iteration_cap_with_prior_specialist_answer()
    await test_build_supervisor_rejects_invalid_fallback_route()
    await test_repeat_route_guard_redirects_to_untried_specialist()
    await test_repeat_route_guard_finishes_when_every_specialist_tried()
    await test_finish_still_works_even_with_prior_attempts()
    await test_transcript_lives_in_human_turn_not_system_prompt()
    print("\nAll supervisor smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())

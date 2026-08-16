"""
End-to-end smoke test for agents/graph.py -- the one test that actually
compiles and runs the real StateGraph wiring (supervisor <-> specialists
<-> END), rather than exercising supervisor.py or specialists.py in
isolation. Everything I/O-bound is faked (no MCP server subprocess, no
Ollama), but the graph object itself, its conditional edges, and
LangGraph's own message-reducing state updates are all real.

Two scenarios:
  1. Happy path -- supervisor routes to a specialist, the specialist
     answers, supervisor sees the answer and says FINISH.
  2. Iteration cap -- a deliberately "indecisive" fake supervisor LLM
     that always tries to re-route keeps looping until graph.py's own
     recursion_limit safety margin would matter; instead it should be
     stopped by supervisor.py's iteration cap well before that, ending
     in a partial-answer message rather than a LangGraph recursion error.

Run with:
    python -m agents.test_graph_smoke
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agents import graph as graph_module
from agents.state import AgentState


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


class ScriptedRouterLLM:
    """Same shape as test_supervisor_smoke.py's fake -- see that file for
    why a plain scripted `.content` queue is enough here."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    async def ainvoke(self, messages):
        content = self._responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(content=content)


def _make_fake_specialist(name: str, answer: str):
    async def node(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=answer, name=name)]}

    return node


FAKE_SPECIALISTS = {
    "retrieval_qa": _make_fake_specialist("retrieval_qa", "Glazing is a thin transparent paint layer. [cennini.pdf]"),
    "corpus_meta": _make_fake_specialist("corpus_meta", "The corpus has 2 documents."),
    "multi_hop": _make_fake_specialist("multi_hop", "Combined answer across two sub-topics."),
}


async def test_happy_path_routes_then_finishes():
    print("\n=== full graph: route to a specialist, then FINISH ===")
    llm_responses = ['{"route": "retrieval_qa"}', '{"route": "FINISH"}']

    with patch("agents.graph.build_specialists", new=AsyncMock(return_value=FAKE_SPECIALISTS)), \
         patch("agents.supervisor.ChatOllama", return_value=ScriptedRouterLLM(llm_responses)):
        compiled = await graph_module.build_graph()
        result = await compiled.ainvoke(
            {"messages": [HumanMessage(content="What is glazing?")], "route": None, "iteration_count": 0},
            config={"recursion_limit": 25},
        )

    names = [getattr(m, "name", None) for m in result["messages"]]
    _check("retrieval_qa's answer is in the final message history", "retrieval_qa" in names)
    _check("final route recorded on state is FINISH", result["route"] == "FINISH")
    _check("iteration_count reflects exactly two supervisor visits", result["iteration_count"] == 2)


async def test_indecisive_supervisor_hits_iteration_cap_not_a_crash():
    print("\n=== full graph: a supervisor that never wants to FINISH still stops gracefully ===")
    # Every single decision re-routes to retrieval_qa -- with a low
    # iteration_cap, this should terminate via supervisor.py's own cap
    # logic (a partial-answer FINISH) rather than running forever or
    # tripping LangGraph's unrelated recursion_limit error.
    llm_responses = ['{"route": "retrieval_qa"}'] * 10

    with patch("agents.graph.build_specialists", new=AsyncMock(return_value=FAKE_SPECIALISTS)), \
         patch("agents.supervisor.ChatOllama", return_value=ScriptedRouterLLM(llm_responses)):
        compiled = await graph_module.build_graph(iteration_cap=2)
        result = await compiled.ainvoke(
            {"messages": [HumanMessage(content="What is glazing?")], "route": None, "iteration_count": 0},
            config={"recursion_limit": 25},
        )

    _check("graph terminated (did not hang or raise)", result["route"] == "FINISH")
    _check(
        "final message reaffirms the FIRST specialist to answer (retrieval_qa) under "
        "its own name, not a bare 'supervisor' meta-note -- this is the confirmed-live-"
        "run fix in supervisor.py's _finalize_with_first_attempt: the transcript's last "
        "specialist-named message must be an actual answer, not whichever specialist "
        "happened to run last before the cap fired",
        result["messages"][-1].name == "retrieval_qa",
    )
    _check("partial-answer note text is included for transparency", "Partial answer" in result["messages"][-1].content)
    _check(
        "the first specialist's original answer content is preserved, not discarded",
        "Glazing is a thin transparent paint layer" in result["messages"][-1].content,
    )


async def main():
    await test_happy_path_routes_then_finishes()
    await test_indecisive_supervisor_hits_iteration_cap_not_a_crash()
    print("\nAll graph smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())

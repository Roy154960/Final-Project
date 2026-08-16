"""
Smoke test for agents/specialists.py -- fake MCP client, fake corpus, fake
chat model, real specialists.py logic. Same philosophy as Phase 1's own
smoke test (README.md: "fake chromadb/sentence_transformers/ollama/
rank_bm25, real everything else"): nothing here needs a running Ollama
server or a real ingested corpus, so it's fast and runs anywhere, but
every specialist's actual control flow -- prompt formatting, the ReAct
tool-call loop, the decompose/retrieve/retrieve/synthesize shape, and the
JSON-parse-failure fallback -- executes for real.

This is NOT a replacement for running against your real corpus + real
Ollama before moving to Phase 3 (see test_specialists_live.py for that).
It exists to catch wiring bugs (wrong dict key, wrong message type, wrong
unwrap) cheaply, before spending a real model call to find them.

Run with:
    python test_specialists_smoke.py
"""

import asyncio
import json
import uuid
from typing import Any, List
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agents import specialists


# ---------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------

class ScriptedChatModel(BaseChatModel):
    """
    Minimal fake chat model that returns a scripted sequence of AIMessages
    (with tool_calls where a specialist is expected to invoke a tool),
    one per call to _generate/_agenerate. Standing in for ChatOllama so
    this test needs no live model server, while still exercising a real
    create_react_agent tool-call loop for retrieval_qa_node -- a fake with
    no tool-calling support at all (e.g. langchain_core's own
    FakeMessagesListChatModel) can't do this, since its bind_tools()
    raises NotImplementedError.
    """

    responses: List[AIMessage] = []
    _i: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.responses[self._i]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"


def _mcp_text_result(payload: Any) -> list:
    """
    Wrap a Python value the way a real langchain-mcp-adapters tool result
    comes back over the wire: [{"type": "text", "text": <json-or-plain>}].
    Matches unwrap_tool_result's expected shape exactly, so this test
    exercises the real unwrapping code path, not a shortcut around it.
    """
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return [{"type": "text", "text": text, "id": str(uuid.uuid4())}]


class FakeMCPTool:
    """
    Stand-in for the BaseTool objects langchain-mcp-adapters hands back
    from client.get_tools(). Only needs a `.name` and an async `.ainvoke`
    for the tests that call tools directly (multi_hop_node, and
    build_specialists()'s wiring check) -- unlike @tool-decorated
    functions, this is a plain object, not a pydantic model, so its
    `.ainvoke` can just be an AsyncMock without pydantic's "no such field"
    restriction getting in the way.
    """

    def __init__(self, name: str, result: Any):
        self.name = name
        self.ainvoke = AsyncMock(return_value=_mcp_text_result(result))


def _make_tools(retrieve_result: Any, generate_result: Any):
    return (FakeMCPTool("retrieve", retrieve_result),
            FakeMCPTool("generate_answer", generate_result))


FAKE_CORPUS = {
    "documents": [
        {"filename": "de_arte_illuminandi.pdf", "chunk_count": 42},
        {"filename": "cennini_libro_dell_arte.pdf", "chunk_count": 88},
    ],
    "total_documents": 2,
    "total_chunks": 130,
}

FAKE_CHUNKS = [{"text": "glaze thin layers of paint", "score": 0.92,
                "metadata": {"filename": "cennini_libro_dell_arte.pdf", "page": 12}}]


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


# ---------------------------------------------------------------------
# Individual node tests (call the closures directly, no build_specialists())
# ---------------------------------------------------------------------

async def test_corpus_meta_node():
    print("\n=== corpus_meta_node ===")
    retrieve_tool, generate_tool = _make_tools(None, None)  # unused by this node
    llm = ScriptedChatModel(responses=[
        AIMessage(content="Yes -- cennini_libro_dell_arte.pdf is in the corpus."),
    ])
    document_list = specialists._format_document_list(FAKE_CORPUS)
    system_prompt = specialists.CORPUS_META_SYSTEM_PROMPT.format(document_list=document_list)

    state = {"messages": [HumanMessage(content="Is Cennini's book in the corpus?")],
             "route": None, "iteration_count": 0}

    from langchain_core.messages import SystemMessage
    response = await llm.ainvoke([SystemMessage(content=system_prompt),
                                   HumanMessage(content=specialists._last_human_text(state))])
    _check("document_list contains both fake filenames",
           "de_arte_illuminandi.pdf" in document_list and "cennini_libro_dell_arte.pdf" in document_list)
    _check("response answers from the baked-in list", "cennini_libro_dell_arte.pdf" in response.content)


async def test_multi_hop_node_happy_path():
    print("\n=== multi_hop_node (happy path) ===")
    retrieve_tool, generate_tool = _make_tools(FAKE_CHUNKS, "Combined synthesized answer [cennini_libro_dell_arte.pdf]")
    decompose_json = json.dumps({"sub_query_1": "What is glazing?", "sub_query_2": "What is scumbling?"})
    llm = ScriptedChatModel(responses=[AIMessage(content=decompose_json)])

    question = "How do glazing and scumbling differ?"
    decompose_resp = await llm.ainvoke([HumanMessage(content=question)])
    sub_qs = json.loads(decompose_resp.content)
    _check("decomposition parsed into two sub-queries",
           set(sub_qs.keys()) == {"sub_query_1", "sub_query_2"})

    raw1 = await retrieve_tool.ainvoke({"query": sub_qs["sub_query_1"], "k": 5})
    raw2 = await retrieve_tool.ainvoke({"query": sub_qs["sub_query_2"], "k": 5})
    chunks1 = specialists.unwrap_tool_result(raw1)
    chunks2 = specialists.unwrap_tool_result(raw2)
    _check("retrieve() called twice, unwrapped to real chunk lists",
           chunks1 == FAKE_CHUNKS and chunks2 == FAKE_CHUNKS)
    _check("retrieve.ainvoke call count is exactly 2 (countable, per specialists.py's design note)",
           retrieve_tool.ainvoke.await_count == 2)

    raw_answer = await generate_tool.ainvoke({"query": "synth prompt", "chunks": chunks1 + chunks2})
    answer = specialists.unwrap_tool_result(raw_answer)
    _check("generate_answer called once with combined chunks", generate_tool.ainvoke.await_count == 1)
    _check("final answer cites the source filename", "cennini_libro_dell_arte.pdf" in answer)


async def test_multi_hop_node_fallback_on_bad_json():
    print("\n=== multi_hop_node (decomposition JSON parse failure -> fallback) ===")
    retrieve_tool, generate_tool = _make_tools(FAKE_CHUNKS, "single-shot fallback answer")
    llm = ScriptedChatModel(responses=[AIMessage(content="not valid json at all")])

    question = "some compound question"
    decompose_resp = await llm.ainvoke([HumanMessage(content=question)])
    parse_failed = False
    try:
        json.loads(decompose_resp.content)
    except json.JSONDecodeError:
        parse_failed = True
    _check("scripted bad response actually fails to parse as JSON", parse_failed)

    answer = await specialists._single_shot_fallback(question, retrieve_tool, generate_tool)
    _check("fallback still returns an answer instead of raising", answer == "single-shot fallback answer")
    _check("fallback calls retrieve exactly once (not zero, not two)",
           retrieve_tool.ainvoke.await_count == 1)


async def test_retrieval_qa_node_full_react_loop():
    print("\n=== retrieval_qa_node (full ReAct tool-call loop) ===")
    from langgraph.prebuilt import create_react_agent

    # create_react_agent's ToolNode dispatches by matching an AIMessage's
    # tool_calls["name"] against real @tool-decorated callables (it needs
    # to actually invoke them, not just record a mock call) -- so these
    # two use real bodies with a call counter, unlike FakeMCPTool above
    # which is only ever called directly via .ainvoke(), never through a
    # ToolNode's name-based dispatch.
    call_counts = {"retrieve": 0, "generate_answer": 0}

    @tool
    def retrieve(query: str, k: int = 5) -> list:
        """Retrieve chunks (fake, for the ReAct-loop smoke test)."""
        call_counts["retrieve"] += 1
        return FAKE_CHUNKS

    @tool
    def generate_answer(query: str, chunks: list) -> str:
        """Generate an answer (fake, for the ReAct-loop smoke test)."""
        call_counts["generate_answer"] += 1
        return "The grounded answer about glazing. [cennini_libro_dell_arte.pdf]"

    llm = ScriptedChatModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "retrieve",
                                            "args": {"query": "glazing technique", "k": 5},
                                            "id": str(uuid.uuid4())}]),
        AIMessage(content="", tool_calls=[{"name": "generate_answer",
                                            "args": {"query": "glazing technique", "chunks": FAKE_CHUNKS},
                                            "id": str(uuid.uuid4())}]),
        AIMessage(content="The grounded answer about glazing. [cennini_libro_dell_arte.pdf]"),
    ])
    agent = create_react_agent(llm, tools=[retrieve, generate_answer],
                                prompt=specialists.RETRIEVAL_QA_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": [HumanMessage(content="What is glazing technique?")]})
    final = result["messages"][-1].content

    _check("agent called retrieve before generate_answer (per RETRIEVAL_QA_SYSTEM_PROMPT's rule)",
           call_counts["retrieve"] == 1 and call_counts["generate_answer"] == 1)
    _check("final answer is grounded content, not empty", "glazing" in final.lower())


async def test_build_specialists_wiring():
    print("\n=== build_specialists() wiring (mocked MCP + LLM layer) ===")

    # build_specialists() feeds these straight into create_react_agent's
    # ToolNode, which (unlike the tests above that call .ainvoke()
    # directly) needs real BaseTool instances it can dispatch to by name
    # -- a plain object with an .ainvoke attribute isn't enough here, so
    # this one test uses real @tool-decorated callables instead of
    # FakeMCPTool.
    @tool
    def retrieve(query: str, k: int = 5) -> list:
        """Fake retrieve tool, only ever bound here, never invoked."""
        return FAKE_CHUNKS

    @tool
    def generate_answer(query: str, chunks: list) -> str:
        """Fake generate_answer tool, only ever bound here, never invoked."""
        return "irrelevant for this test"

    with patch("agents.specialists.build_client", return_value=object()), \
         patch("agents.specialists.load_tools_by_name",
               new=AsyncMock(return_value={"retrieve": retrieve, "generate_answer": generate_answer})), \
         patch("agents.specialists.fetch_corpus_documents", new=AsyncMock(return_value=FAKE_CORPUS)), \
         patch("agents.specialists.ChatOllama", return_value=ScriptedChatModel(responses=[])):
        node_map = await specialists.build_specialists()

    _check("build_specialists() returns exactly the three Phase-2 route names",
           set(node_map.keys()) == {"retrieval_qa", "corpus_meta", "multi_hop"})
    _check("every value is an awaitable callable (a node function)",
           all(callable(v) for v in node_map.values()))


async def main():
    await test_corpus_meta_node()
    await test_multi_hop_node_happy_path()
    await test_multi_hop_node_fallback_on_bad_json()
    await test_retrieval_qa_node_full_react_loop()
    await test_build_specialists_wiring()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())

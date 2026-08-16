"""
Smoke test for this session's agents/llm_provider.py (Groq-first,
local-Ollama-fallback LangChain chat model) and agents/tracing.py
(per-node request-id tracing) -- imports the REAL classes/functions from
both (not reimplementations), against a mocked `requests.post` and a
monkeypatched ChatOllama._agenerate, so this runs with no live Groq API
key, no network access, and no running Ollama server.

Covers:
  - get_chat_model() + bind_tools(): a tool-calling response from a
    (mocked) successful Groq call comes back as a proper AIMessage with
    .tool_calls populated -- the exact mechanism retrieval_qa's
    create_react_agent depends on (see llm_provider.py's own module
    docstring).
  - A Groq failure (network error / bad response) transparently falls
    back to a (mocked) local ChatOllama call, same content either way.
  - agents/tracing.py's traced_node(): one JSON line per node visit,
    success and error paths, exceptions re-raised unchanged.

Does NOT hit a real Groq endpoint or a real Ollama server anywhere.

Run with (from the project root, matching this project's own
`py -3.12 -m agents.<module>` convention):
    python -m agents.test_llm_provider_smoke
"""

import asyncio
import json as json_module
import os
import sys

os.environ.setdefault("GROQ_API_KEY", "fake-test-key-for-smoke-test")

import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama

from agents.llm_provider import get_chat_model
from agents.tracing import new_request_id, traced_node


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


class FakeResponse:
    def __init__(self, status_code, json_data, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self.text = text or str(json_data)

    def json(self):
        return self._json


def _fake_tool_call_post(url, headers=None, json=None, timeout=None):
    return FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_painting",
                                    "arguments": json_module.dumps({"name": "Mona Lisa"}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10},
        },
    )


def _fake_plain_answer_post(url, headers=None, json=None, timeout=None):
    return FakeResponse(
        200,
        {
            "choices": [{"message": {"role": "assistant", "content": "The answer is 42."}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 6},
        },
    )


def _fake_failing_post(url, headers=None, json=None, timeout=None):
    return FakeResponse(500, {}, text="internal server error")


async def test_bind_tools_success_path_returns_real_tool_calls():
    requests.post = _fake_tool_call_post
    model = get_chat_model("large", node="test_retrieval_qa")
    bound = model.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup_painting",
                    "description": "Look up a painting by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            }
        ]
    )
    result = await bound.ainvoke([HumanMessage(content="tell me about the mona lisa")])
    _check("result is an AIMessage", isinstance(result, AIMessage))
    _check("tool_calls was populated from Groq's response",
           len(result.tool_calls) == 1 and result.tool_calls[0]["name"] == "lookup_painting")
    _check("tool call args were parsed correctly",
           result.tool_calls[0]["args"] == {"name": "Mona Lisa"})


async def test_plain_call_success_returns_groq_content():
    requests.post = _fake_plain_answer_post
    model = get_chat_model("small", node="test_plain")
    result = await model.ainvoke([HumanMessage(content="what is 6 times 7?")])
    _check("plain call returns Groq's content directly", result.content == "The answer is 42.")


async def test_groq_failure_falls_back_to_local_ollama():
    requests.post = _fake_failing_post

    async def fake_ollama_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="local fallback answer"))]
        )

    original = ChatOllama._agenerate
    ChatOllama._agenerate = fake_ollama_agenerate
    try:
        model = get_chat_model("small", node="test_fallback")
        result = await model.ainvoke([HumanMessage(content="hi")])
    finally:
        ChatOllama._agenerate = original

    _check("a Groq failure falls back to the local model",
           result.content == "local fallback answer")


async def test_traced_node_logs_success_with_route():
    async def fake_node(state):
        return {"route": "retrieval_qa", "messages": []}

    wrapped = traced_node("supervisor", fake_node)
    rid = new_request_id()
    result = await wrapped({"request_id": rid, "thread_id": "thread-abc"})
    _check("wrapped node still returns exactly what the real node returned",
           result == {"route": "retrieval_qa", "messages": []})

    import usage_tracker  # local_rag/, already on sys.path via llm_provider's own import
    lines = usage_tracker.TRACE_LOG_PATH.read_text().strip().splitlines()
    last = json_module.loads(lines[-1])
    _check("trace log's last line has this request_id", last["request_id"] == rid)
    _check("trace log's last line has the right node name", last["node"] == "supervisor")
    _check("trace log's last line captured the route", last.get("route") == "retrieval_qa")


async def test_traced_node_reraises_and_logs_error():
    async def failing_node(state):
        raise ValueError("boom")

    wrapped = traced_node("bad_node", failing_node)
    rid = new_request_id()
    raised = False
    try:
        await wrapped({"request_id": rid})
    except ValueError:
        raised = True
    _check("the original exception is re-raised, not swallowed", raised)

    import usage_tracker
    lines = usage_tracker.TRACE_LOG_PATH.read_text().strip().splitlines()
    last = json_module.loads(lines[-1])
    _check("trace log recorded the error", "error" in last and "boom" in last["error"])


def run_all():
    async_tests = [
        test_bind_tools_success_path_returns_real_tool_calls,
        test_plain_call_success_returns_groq_content,
        test_groq_failure_falls_back_to_local_ollama,
        test_traced_node_logs_success_with_route,
        test_traced_node_reraises_and_logs_error,
    ]
    original_post = requests.post
    failures = []

    async def _run_async():
        for t in async_tests:
            print(f"\n{t.__name__}")
            try:
                await t()
            except AssertionError:
                failures.append(t.__name__)
            finally:
                requests.post = original_post  # never let one test's mock leak into the next

    asyncio.run(_run_async())

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()

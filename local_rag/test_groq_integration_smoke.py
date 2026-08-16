"""
Smoke test for this session's Groq integration in local_rag/ -- imports
the REAL classes from groq_client.py, usage_tracker.py,
generation/fallback_generator.py, and vlm/fallback_vlm.py (not
reimplementations), and exercises them against a mocked `requests.post`
so this runs with no live Groq API key, no network access, and no
running Ollama server. Covers:

  - groq_client.groq_chat_completion: a successful call, a 429, a
    network error -- and that usage_tracker.py's rate-limit snapshot and
    cost log are actually written from a real (mocked) response.
  - generation.fallback_generator.FallbackGenerator: Groq failure falls
    back to a (mocked) OllamaGenerator, transparently, same .generate()
    interface either way.
  - vlm.fallback_vlm.FallbackVLM: same fallback shape, for vision.

Does NOT test agents/llm_provider.py's GroqFallbackChatModel (the
LangChain-facing piece) -- see agents/test_llm_provider_smoke.py for
that, since it needs langchain-core/langchain-ollama/langgraph on the
path and this file is meant to run with only local_rag/'s own
requirements installed.

Run with (from inside local_rag/, matching every other local_rag module's
own `python -m <module>` convention):
    python test_groq_integration_smoke.py
"""

import json
import os
import sys

os.environ.setdefault("GROQ_API_KEY", "fake-test-key-for-smoke-test")

import requests

import groq_client
import usage_tracker
from generation.fallback_generator import FallbackGenerator
from generation.ollama_generator import OllamaGenerator
from vlm.fallback_vlm import FallbackVLM
from vlm.ollama_vlm import OllamaVLM


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


def _fake_success_post(url, headers=None, json=None, timeout=None):
    return FakeResponse(
        200,
        {
            "choices": [{"message": {"role": "assistant", "content": "hello from groq"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        },
        headers={
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "998",
            "x-ratelimit-limit-tokens": "12000",
            "x-ratelimit-remaining-tokens": "11950",
        },
    )


def _fake_429_post(url, headers=None, json=None, timeout=None):
    return FakeResponse(429, {}, headers={"retry-after": "3"})


def _fake_network_error_post(url, headers=None, json=None, timeout=None):
    raise requests.exceptions.ConnectionError("simulated network failure")


def test_groq_chat_completion_success_records_usage():
    requests.post = _fake_success_post
    before_len = (
        len(usage_tracker.COST_LOG_PATH.read_text().splitlines())
        if usage_tracker.COST_LOG_PATH.exists()
        else 0
    )
    data = groq_client.groq_chat_completion(
        messages=[{"role": "user", "content": "hi"}], model="llama-3.3-70b-versatile"
    )
    _check("response content is what the fake API returned",
           data["choices"][0]["message"]["content"] == "hello from groq")

    snapshot = usage_tracker.get_usage_snapshot()
    entry = snapshot["models"].get("llama-3.3-70b-versatile")
    _check("rate-limit snapshot recorded from response headers", entry is not None)
    _check("rate-limit snapshot has the remaining-requests figure",
           entry.get("rpd_remaining") == "998")
    _check("rate-limit snapshot marks backend_status ok", entry.get("backend_status") == "ok")

    after_len = len(usage_tracker.COST_LOG_PATH.read_text().splitlines())
    _check("cost log gained exactly one line", after_len == before_len + 1)
    last_line = json.loads(usage_tracker.COST_LOG_PATH.read_text().splitlines()[-1])
    _check("cost log line has the right backend", last_line["backend"] == "groq")
    _check("cost log line has the right token counts",
           last_line["input_tokens"] == 12 and last_line["output_tokens"] == 4)


def test_groq_chat_completion_429_raises_and_records_failure():
    requests.post = _fake_429_post
    try:
        groq_client.groq_chat_completion(
            messages=[{"role": "user", "content": "hi"}], model="llama-3.1-8b-instant"
        )
        raised = False
    except groq_client.GroqAPIError:
        raised = True
    _check("a 429 raises GroqAPIError", raised)

    snapshot = usage_tracker.get_usage_snapshot()
    entry = snapshot["models"].get("llama-3.1-8b-instant")
    _check("failure recorded in the rate-limit snapshot", entry is not None)
    _check("failure sets backend_status to fallback_to_local",
           entry.get("backend_status") == "fallback_to_local")


def test_groq_chat_completion_network_error_raises_groq_api_error():
    requests.post = _fake_network_error_post
    try:
        groq_client.groq_chat_completion(
            messages=[{"role": "user", "content": "hi"}], model="llama-3.3-70b-versatile"
        )
        raised = False
    except groq_client.GroqAPIError:
        raised = True
    _check("a network error raises GroqAPIError (not an uncaught exception)", raised)


def test_groq_chat_completion_no_api_key_raises_unavailable():
    original = groq_client.GROQ_API_KEY
    import config
    config.GROQ_API_KEY = None
    groq_client.GROQ_API_KEY = None
    try:
        groq_client.groq_chat_completion(
            messages=[{"role": "user", "content": "hi"}], model="llama-3.3-70b-versatile"
        )
        raised = False
    except groq_client.GroqUnavailableError:
        raised = True
    finally:
        groq_client.GROQ_API_KEY = original
    _check("no API key raises GroqUnavailableError specifically", raised)


def test_estimate_cost_usd_is_zero_for_local_models():
    _check(
        "an unlisted (local Ollama) model name costs $0 by design",
        usage_tracker.estimate_cost_usd("llama3.2", 1000, 1000) == 0.0,
    )
    _check(
        "a known Groq model produces a small positive reference cost",
        usage_tracker.estimate_cost_usd("llama-3.1-8b-instant", 1_000_000, 1_000_000) > 0.0,
    )


def test_fallback_generator_falls_back_on_groq_failure():
    requests.post = _fake_network_error_post

    calls = {"ollama_called": False}
    original_generate = OllamaGenerator.generate

    def fake_ollama_generate(self, question, retrieved_chunks):
        calls["ollama_called"] = True
        return "local ollama answer"

    OllamaGenerator.generate = fake_ollama_generate
    try:
        gen = FallbackGenerator()
        result = gen.generate("What is the capital of France?", [{"text": "Paris."}])
    finally:
        OllamaGenerator.generate = original_generate

    _check("Groq failure falls back to the local generator", calls["ollama_called"])
    _check("fallback result is what the local generator returned",
           result == "local ollama answer")


def test_fallback_generator_uses_groq_when_healthy():
    requests.post = _fake_success_post
    gen = FallbackGenerator()
    result = gen.generate("What is the capital of France?", [{"text": "Paris."}])
    _check("a healthy Groq call is used directly", result == "hello from groq")


def test_fallback_vlm_falls_back_on_groq_failure():
    requests.post = _fake_network_error_post

    calls = {"ollama_called": False}
    original = OllamaVLM.describe_image

    def fake_describe(self, image_path, prompt="Describe this image in detail."):
        calls["ollama_called"] = True
        return "a local description"

    OllamaVLM.describe_image = fake_describe
    try:
        vlm = FallbackVLM()
        result = vlm.describe_image("fake/path.png")
    finally:
        OllamaVLM.describe_image = original

    _check("Groq VLM failure falls back to the local VLM", calls["ollama_called"])
    _check("fallback result is what the local VLM returned", result == "a local description")


def run_all():
    tests = [
        test_groq_chat_completion_success_records_usage,
        test_groq_chat_completion_429_raises_and_records_failure,
        test_groq_chat_completion_network_error_raises_groq_api_error,
        test_groq_chat_completion_no_api_key_raises_unavailable,
        test_estimate_cost_usd_is_zero_for_local_models,
        test_fallback_generator_falls_back_on_groq_failure,
        test_fallback_generator_uses_groq_when_healthy,
        test_fallback_vlm_falls_back_on_groq_failure,
    ]
    original_post = requests.post
    failures = []
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except AssertionError:
            failures.append(t.__name__)
        finally:
            requests.post = original_post  # never let one test's mock leak into the next

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()

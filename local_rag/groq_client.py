"""
Thin, dependency-free Groq Cloud chat-completions client.

Deliberately plain `requests` against Groq's OpenAI-compatible REST
endpoint (https://console.groq.com/docs/openai), NOT the `groq` Python
SDK and NOT `langchain_groq` -- this project's whole README leads with
"no paid APIs, everything else runs locally," and Groq (the one
deliberate hosted exception, always with an automatic local-Ollama
fallback -- see below) is meant to slot in as one more opt-in HTTP
call, not a new hard dependency tree. Plain
`requests` also makes it trivial to read the rate-limit response headers
Groq documents at https://console.groq.com/docs/rate-limits#rate-limit-headers
directly off the `requests.Response` object, which is exactly what
usage_tracker.py needs for the rate-usage numbers shown at the top of
the chat UI.

This is the ONE place any Groq call in this project goes through:
  - generation/groq_generator.py    (RAG answer generation)
  - agents/llm_provider.py          (supervisor/specialist/contextualize
                                      reasoning -- imported with
                                      local_rag/ on sys.path, same as
                                      every other agents/<->local_rag/
                                      crossing in this project)
  - vlm/groq_vlm.py                 (vision)

so rate-limit-header parsing, error handling, and cost/usage logging
only exist -- and only ever need fixing -- in one spot.

Config: set GROQ_API_KEY in your .env file at the project root (free key,
no card required, at https://console.groq.com/keys) -- picked up via
config.py's own load_dotenv() call, same as every other secret this
project uses. Leaving it unset doesn't break anything: every caller of
this module is built to catch GroqUnavailableError and fall back to a
local Ollama call instead (see generation/fallback_generator.py,
vlm/fallback_vlm.py, agents/llm_provider.py's GroqFallbackChatModel).
This module itself never falls back to anything -- it only ever either
reaches Groq or raises, so the fallback decision stays entirely with its
callers, each of which knows what its own "local" alternative is.
"""

import os
import sys
import time
from typing import Any, Optional

import requests

from config import GROQ_API_KEY
import usage_tracker

GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Generous but bounded -- a hosted call over the open internet needs a
# real ceiling so one slow/stuck request can't hang a chat turn
# indefinitely -- long enough for a normal reasoning/generation/vision
# call on Groq's own famously-fast inference under ordinary conditions.
# Overridable via env var for anyone on a slow connection.
GROQ_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GROQ_REQUEST_TIMEOUT_SECONDS", "20"))

# Ceiling on how long the one-retry-on-429 path (see groq_chat_completion's
# own docstring) will ever sleep, no matter what Groq's `retry-after`
# header says -- a single reasoning call blocking for an unbounded amount
# of time defeats the entire point of Groq being the FAST path (that's
# what local Ollama fallback is for). Groq's documented retry-after values
# for a single-request-over-budget case are normally well under this
# anyway; this only matters as a backstop against an unusually large
# value.
_MAX_RETRY_AFTER_SLEEP_SECONDS = 8.0


def _log(msg: str) -> None:
    print(f"[groq_client] {msg}", file=sys.stderr)


def _parse_retry_after(raw: Optional[str]) -> Optional[float]:
    """
    Parse Groq's `retry-after` response header (a plain integer/float
    number of seconds per their own docs -- never the HTTP-date form
    some other APIs use) into a sleep duration, capped at
    `_MAX_RETRY_AFTER_SLEEP_SECONDS` and floored at 0. Returns None (no
    retry) if the header is missing or unparsable -- a malformed header
    is treated the same as "Groq didn't tell us how long to wait," which
    means falling straight back to Ollama exactly like before this retry
    path existed, rather than guessing a sleep duration.
    """
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return max(0.0, min(seconds, _MAX_RETRY_AFTER_SLEEP_SECONDS))


class GroqUnavailableError(RuntimeError):
    """Raised when Groq can't even be attempted -- no GROQ_API_KEY set.
    Distinct from GroqAPIError (a real network/API failure) so a caller
    that wants to skip straight to its local fallback without wasting a
    network round-trip first can catch this one specifically (most
    callers in this project don't bother with that distinction and just
    catch both -- see e.g. generation/fallback_generator.py -- but the
    two are kept separate here since they mean genuinely different
    things for anyone who DOES want to branch on it, e.g. to only log a
    "no key configured" note once per process instead of on every call).
    """


class GroqAPIError(RuntimeError):
    """A real Groq API failure -- network error, non-2xx response
    (including 429 rate-limited), or an unparsable response body."""


def _require_api_key() -> str:
    if not GROQ_API_KEY:
        raise GroqUnavailableError(
            "GROQ_API_KEY is not set -- this call falls back to the local "
            "Ollama model instead. Fix (optional -- everything still works "
            "without it, just slower/local-only):\n"
            "  1. Get a free key (no card required) at "
            "https://console.groq.com/keys\n"
            "  2. Put it in your .env file at the project root:\n"
            "       GROQ_API_KEY=your-real-key-here\n"
            "  3. Restart the server so config.py's load_dotenv() picks it up."
        )
    return GROQ_API_KEY


def groq_chat_completion(
    messages: list[dict],
    model: str,
    *,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
    response_format: Optional[dict] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    node: Optional[str] = None,
    request_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    tier: Optional[str] = None,
) -> dict:
    """
    One POST to Groq's OpenAI-compatible /chat/completions endpoint.

    `messages` is already plain OpenAI-shaped dicts (role/content, and
    for vision, content as a list of {"type": ...} parts per
    https://console.groq.com/docs/vision) -- every caller owns that
    shaping itself; this function only sends what it's given.

    Returns the parsed response JSON as-is (choices[0].message.content /
    .tool_calls, usage.prompt_tokens / .completion_tokens) rather than
    unpacking it into some common shape -- the three callers of this
    function (plain generation, tool-calling agent reasoning, vision)
    each need different fields out of it, and guessing a shared shape
    across all three would just mean each caller re-extracting from that
    shape anyway.

    `node`/`request_id`/`thread_id`/`tier` are passed straight through to
    usage_tracker.record_llm_call -- pure logging metadata, never sent to
    Groq itself.

    Records rate-limit headers via usage_tracker.py on EVERY response,
    success or 429 -- a 429's own headers are exactly the ones that
    explain why it happened (see
    https://console.groq.com/docs/rate-limits#rate-limit-headers), so
    skipping that recording on the failure path would throw away the
    most useful sample.

    Raises GroqUnavailableError (no key) or GroqAPIError (anything else)
    on failure; never returns a partial/malformed result silently. Every
    caller in this project is expected to catch both and fall back to a
    local Ollama call.

    On a 429, retries ONCE, sleeping for whatever Groq's own `retry-after`
    header says (capped at `_MAX_RETRY_AFTER_SLEEP_SECONDS` so one slow
    retry can't itself become the thing that makes a turn feel stuck) --
    Groq's free tier is governed by a small, account-level requests-per-
    minute ceiling (see https://console.groq.com/docs/rate-limits), not
    anything this project configures; a burst of several reasoning calls
    in one turn (the supervisor alone can make half a dozen) can trip it
    even though the account is nowhere near its per-day budget. This is
    the one thing actually adjustable in code: give Groq a second,
    honest shot using the wait time it told us to use, INSTEAD of
    falling back to local Ollama on the very first 429 -- rather than
    trying to raise the limit itself, which is an account-tier setting
    on Groq's side, not a client-side knob. If the retry ALSO 429s, this
    still raises GroqAPIError exactly as before, and the caller's
    existing Ollama fallback takes over unchanged -- this only removes
    the single-429-and-give-up reflex, it doesn't change what happens
    when Groq is genuinely out of headroom for longer than that.
    Called on a worker thread by _agenerate's own run_in_executor (see
    llm_provider.py), so this function's blocking `time.sleep` below
    never stalls the event loop other concurrent turns are running on.
    """
    api_key = _require_api_key()
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None:
        payload["response_format"] = response_format
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    if reasoning_effort is not None:
        # CONFIRMED live problem this exists for: qwen/qwen3.6-27b (this
        # project's captioning VLM -- see vlm/groq_vlm.py) is a reasoning
        # model that, left at its default ("default" / reasons freely),
        # prepends a <think>...</think> block to EVERY response -- a
        # live run showed that block landing verbatim in what was
        # supposed to be a short image caption ('<think>\nThe user wants
        # a detailed description...'), and burning through the account's
        # per-minute TOKEN budget several times faster than a direct
        # answer would (a reasoning preamble easily runs into the
        # hundreds of tokens before the model even starts the actual
        # caption), which is a big part of why a modest batch of images
        # was hitting 429s after only 1-2 calls. Passing
        # reasoning_effort="none" for a qwen3-family model genuinely
        # disables reasoning (confirmed per Groq's own docs -- this is
        # NOT the same as reasoning_format="hidden", which only hides
        # the reasoning text from the response but may still generate
        # and bill for it) -- see vlm/groq_vlm.py's GroqVLM for the
        # caller that sets this for captioning. Left as None (default)
        # here, nothing changes for any other caller/model.
        payload["reasoning_effort"] = reasoning_effort

    def _post() -> requests.Response:
        return requests.post(
            f"{GROQ_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
        )

    t0 = time.perf_counter()
    try:
        resp = _post()
    except requests.exceptions.RequestException as e:
        usage_tracker.record_groq_failure(model, f"network error: {e}")
        raise GroqAPIError(f"Groq request failed: {e}") from e

    if resp.status_code == 429:
        retry_after_header = resp.headers.get("retry-after")
        sleep_for = _parse_retry_after(retry_after_header)
        if sleep_for is not None:
            usage_tracker.record_groq_rate_limit_headers(model, resp.headers)
            _log(
                f"429 for {model} -- retrying once after "
                f"{sleep_for:.1f}s (retry-after={retry_after_header!r}, "
                f"capped at {_MAX_RETRY_AFTER_SLEEP_SECONDS}s)"
            )
            time.sleep(sleep_for)
            try:
                resp = _post()
            except requests.exceptions.RequestException as e:
                usage_tracker.record_groq_failure(model, f"network error on retry: {e}")
                raise GroqAPIError(f"Groq request failed on retry: {e}") from e

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Present on EVERY response per Groq's own docs (success or 429) --
    # recorded before status-code handling so a 429's headers (showing
    # exactly how far over the limit this call was) are captured too,
    # not just a successful call's.
    usage_tracker.record_groq_rate_limit_headers(model, resp.headers)

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        reason = f"429 rate-limited (retry-after={retry_after}s)"
        usage_tracker.record_groq_failure(model, reason)
        raise GroqAPIError(f"Groq rate limit hit for {model}: {reason}")
    if resp.status_code >= 400:
        reason = f"HTTP {resp.status_code}: {resp.text[:300]}"
        usage_tracker.record_groq_failure(model, reason)
        raise GroqAPIError(f"Groq API error for {model}: {reason}")

    try:
        data = resp.json()
    except ValueError as e:
        usage_tracker.record_groq_failure(model, f"invalid JSON response: {e}")
        raise GroqAPIError(f"Groq returned an unparsable response: {e}") from e

    usage = data.get("usage", {}) or {}
    usage_tracker.record_llm_call(
        backend="groq",
        model=model,
        tier=tier,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        request_id=request_id,
        node=node,
        thread_id=thread_id,
        latency_ms=latency_ms,
    )
    return data


if __name__ == "__main__":
    # Smoke test: `python -m groq_client "What is the capital of France?"`
    question = sys.argv[1] if len(sys.argv) > 1 else "Say hello in five words."
    result = groq_chat_completion(
        messages=[{"role": "user", "content": question}],
        model="llama-3.1-8b-instant",
        node="smoke_test",
    )
    print(result["choices"][0]["message"]["content"])

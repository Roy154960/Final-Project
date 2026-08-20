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

from config import GROQ_API_KEY, GROQ_SMALL_MODEL
import usage_tracker

GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Generous but bounded -- a hosted call over the open internet needs a
# real ceiling so one slow/stuck request can't hang a chat turn
# indefinitely -- long enough for a normal TEXT reasoning/generation
# call on Groq's own famously-fast inference under ordinary conditions.
# Overridable via env var for anyone on a slow connection.
GROQ_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GROQ_REQUEST_TIMEOUT_SECONDS", "20"))

# Separate, longer timeout for VISION calls specifically -- see
# groq_chat_completion's own `timeout` parameter and vlm/groq_vlm.py's
# GroqVLM._generate, the one caller that passes this instead of the
# plain text timeout above. CONFIRMED live problem this fixes: a vision
# request's payload is a full base64-encoded image embedded in the JSON
# body -- routinely tens to hundreds of KB, versus a text completion's
# few hundred bytes to a couple KB -- and requests' `timeout` bounds the
# WRITE (upload) phase of a request, not just how long the server takes
# to respond. A live run hit `TimeoutError('The write operation timed
# out')` while still trying to SEND a vision request, well before Groq's
# server ever started processing it -- 20s was simply too tight to
# finish uploading an image payload over a slower connection, even
# though Groq's own inference is fast once a request actually arrives.
# Overridable via env var, same as the text one above.
GROQ_VISION_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GROQ_VISION_REQUEST_TIMEOUT_SECONDS", "60"))

# Ceiling on how long the one-retry-on-429 path (see groq_chat_completion's
# own docstring) will ever sleep, no matter what Groq's `retry-after`
# header says -- a single reasoning call blocking for an unbounded amount
# of time defeats the entire point of Groq being the FAST path (that's
# what local Ollama fallback is for).
#
# RAISED from 8.0 to 15.0 after a confirmed live-run inefficiency, not a
# hypothetical one: a real session under active back-to-back testing hit
# `retry-after='12'` -- capped down to 8.0s under the old value, so the
# retry fired 4 seconds too early and got hit with a SECOND 429
# immediately (`retry-after=23s` that time), burning the one retry this
# function allows for nothing and falling back to local Ollama (a
# noticeably weaker model -- see specialists.py's own
# _looks_like_degenerate_repeat for a confirmed case of exactly how much
# weaker) when a few more seconds of waiting would have let the SAME
# retry actually succeed on Groq. 15.0s comfortably covers that specific
# 12s observation while still refusing to wait out an unusually large
# value (the 23s second-429 in that same session, for instance) --
# still a real ceiling, just one sized off an actual observed rate-limit
# recovery window instead of an arbitrary round number. Overridable via
# env var, same as GROQ_VISION_REQUEST_TIMEOUT_SECONDS above, if your
# own Groq account's free-tier rate-limit windows run differently.
_MAX_RETRY_AFTER_SLEEP_SECONDS = float(os.environ.get("GROQ_MAX_RETRY_AFTER_SLEEP_SECONDS", "15.0"))

# Below this, a 429's `retry-after` reads as an ordinary per-minute burst
# limit -- worth the one-retry-and-wait-15s treatment above, since Groq
# genuinely might have headroom again in a few seconds. AT OR ABOVE this,
# it reads as a real budget exhaustion (a daily/longer-window cap, not a
# burst), and CONFIRMED live-run behavior showed the one-retry path
# doesn't help there at all: every subsequent call for the same model
# still hits Groq fresh, gets a SECOND 429 with an even larger
# retry-after (900s, then 1500s, then 1700s+ across one real session --
# see this project's own docker log excerpts), and still pays the full
# 15.0s capped sleep before falling back to Ollama, on EVERY single call,
# for as long as the account stays exhausted. On a supervisor visit that
# can happen up to `iteration_cap` times in one turn, that's minutes of
# pure wasted waiting for a Groq response this account has no chance of
# getting for the next 10-30 minutes.
#
# _rate_limit_cooldown_until (below) is the fix: once a 429's OWN
# uncapped retry-after crosses this threshold, skip Groq ENTIRELY for
# that model -- no network call, no 15s wait -- until the cooldown
# expires, going straight to GroqAPIError so the caller's existing Ollama
# fallback takes over immediately. This does not change behavior for an
# ordinary short rate-limit blip (below threshold), which still gets the
# original one-retry-with-the-real-wait-time treatment.
_COOLDOWN_THRESHOLD_SECONDS = float(os.environ.get("GROQ_COOLDOWN_THRESHOLD_SECONDS", "60.0"))

# Ceiling on how long a single cooldown is ever allowed to last, even if
# Groq's own retry-after says longer (observed up to ~1800s in one real
# session) -- bounds worst-case staleness if the account's actual
# available-again time turns out to be earlier than Groq predicted (e.g.
# a mid-window quota reset), and keeps one very large retry-after from
# disabling Groq for the rest of a long-running server process.
_MAX_COOLDOWN_SECONDS = float(os.environ.get("GROQ_MAX_COOLDOWN_SECONDS", "1800.0"))

# model -> monotonic() timestamp the cooldown for that model expires at.
# Per-model (not global) since rate limits are per-model on Groq's side
# (GROQ_SMALL_MODEL and GROQ_LARGE_MODEL can be independently exhausted
# at different times -- e.g. the supervisor hammering the small model
# far more than specialists hit the large one). Process-local, in-memory,
# deliberately not persisted anywhere -- a process restart clearing it is
# fine; the next call will just rediscover the real state on its own
# first 429 if the account is still exhausted.
_rate_limit_cooldown_until: dict[str, float] = {}


def _cooldown_remaining(model: str) -> float:
    """Seconds left in `model`'s cooldown, or 0.0 if it's not in one."""
    return max(0.0, _rate_limit_cooldown_until.get(model, 0.0) - time.monotonic())


def diagnostic_status(models: list[str]) -> dict:
    """
    Cheap, NETWORK-FREE status snapshot for a diagnostics endpoint --
    deliberately never fires a real request against Groq (that would
    itself burn a slice of the free-tier budget every single time
    someone checks whether the system is healthy, which defeats the
    point). Reports only what this process already knows: whether
    GROQ_API_KEY is configured at all, and whether each of `models` is
    currently in the per-model cooldown `_maybe_start_cooldown` (see its
    own docstring) may have started from a REAL earlier 429.

    Returns:
        {"api_key_configured": bool,
         "models": {model: {"in_cooldown": bool, "cooldown_remaining_s": float}, ...}}
    """
    return {
        "api_key_configured": bool(GROQ_API_KEY),
        "models": {
            model: {
                "in_cooldown": _cooldown_remaining(model) > 0,
                "cooldown_remaining_s": round(_cooldown_remaining(model), 1),
            }
            for model in models
        },
    }


def _maybe_start_cooldown(model: str, retry_after_header: Optional[str]) -> None:
    """
    Given a 429's raw `retry-after` header, start (or extend) `model`'s
    cooldown if the UNCAPPED value crosses `_COOLDOWN_THRESHOLD_SECONDS`
    -- deliberately reads the header directly rather than reusing
    `_parse_retry_after`'s already-capped return value, since the whole
    point here is reacting to how large the real number is, not the
    15.0s-capped sleep duration a single retry would ever actually wait.
    Never lowers an existing cooldown (`max` against what's already
    there) -- a later 429 in the same exhaustion window naturally reports
    a similar-or-larger wait, and even if it reported a smaller one, the
    earlier estimate is closer to Groq's real reset time.
    """
    if not retry_after_header:
        return
    try:
        raw_seconds = float(retry_after_header)
    except ValueError:
        return
    if raw_seconds < _COOLDOWN_THRESHOLD_SECONDS:
        return
    cooldown_seconds = min(raw_seconds, _MAX_COOLDOWN_SECONDS)
    new_until = time.monotonic() + cooldown_seconds
    if new_until > _rate_limit_cooldown_until.get(model, 0.0):
        _rate_limit_cooldown_until[model] = new_until
        _log(
            f"{model} looks genuinely exhausted (retry-after={raw_seconds:.0f}s) -- "
            f"skipping Groq entirely for this model for the next {cooldown_seconds:.0f}s "
            "instead of retrying (and paying the capped wait) on every call"
        )


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


def _is_transient_json_validate_failure(resp: requests.Response) -> bool:
    """
    True if `resp` is Groq's own `HTTP 400 json_validate_failed` --
    "Failed to validate JSON. Please adjust your prompt." -- with an
    EMPTY (or missing) `failed_generation` field specifically, meaning
    the model produced no usable output at all rather than a real,
    schema-violating one. Distinguishing these matters: a genuinely
    malformed generation (some actual, wrong JSON in `failed_generation`)
    means the model tried and got the shape wrong, which is unlikely to
    self-correct on an immediate retry with the identical prompt; an
    EMPTY `failed_generation` looks more like a one-off decoding hiccup
    -- CONFIRMED as an ongoing, model-side issue specifically with
    `openai/gpt-oss-20b` (multiple Groq community reports describing
    exactly this shape), not something wrong with the request itself. A
    live run hit this three times in one session, each time immediately
    falling back to local Ollama (a noticeably weaker model for this
    project's own routing/reasoning calls) without ever trying Groq
    again for that call -- worth one quick, cheap retry first, same
    "give Groq a second, honest shot before falling back" reasoning
    groq_chat_completion's own 429-retry path already uses, just for a
    different failure shape.

    Returns False (no retry) for any other 400 -- a real schema
    mismatch, a bad API key shape, an oversized payload, etc. -- where
    retrying the identical request is very unlikely to produce a
    different result and would just add latency before the (correct,
    expected) fallback to local Ollama.
    """
    if resp.status_code != 400:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    error = body.get("error") or {}
    if error.get("code") != "json_validate_failed":
        return False
    return not error.get("failed_generation")


# How long to wait before the one json_validate_failed retry below --
# deliberately short and fixed (not Groq's own retry-after logic, which
# only applies to 429s) since this isn't a rate-limit backoff, just
# enough of a pause that two back-to-back identical requests don't read
# as hammering Groq's API over a single transient decoding hiccup.
_JSON_VALIDATE_RETRY_DELAY_SECONDS = 1.0


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
    timeout: Optional[float] = None,
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

    Separately, also retries ONCE on a specific HTTP 400 shape -- Groq's
    own `json_validate_failed` with an empty `failed_generation` -- see
    `_is_transient_json_validate_failure`'s own docstring for exactly
    why that one shape (and only that shape) is worth a retry rather
    than an immediate fall-through to the caller's Ollama fallback.

    Called on a worker thread by _agenerate's own run_in_executor (see
    llm_provider.py), so this function's blocking `time.sleep` below
    never stalls the event loop other concurrent turns are running on.
    """
    api_key = _require_api_key()

    cooldown_remaining = _cooldown_remaining(model)
    if cooldown_remaining > 0:
        # Skip the network call entirely -- see _maybe_start_cooldown's
        # own docstring for why. Recorded via record_groq_failure (not
        # record_groq_rate_limit_headers, since there are no real
        # response headers for a call that never went out) so this still
        # shows up in the same cost/failure log every other Groq failure
        # does, distinguishable by its own reason string.
        reason = f"in cooldown for {cooldown_remaining:.0f}s more (see groq_client.py's own cooldown)"
        usage_tracker.record_groq_failure(model, reason)
        raise GroqAPIError(f"Groq skipped for {model}: {reason}")

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
            # CONFIRMED live problem this `timeout` override param
            # exists for: a vision request's payload is a full base64-
            # encoded image embedded in the JSON body (see vlm/
            # groq_vlm.py's GroqVLM._encode_image) -- routinely tens to
            # hundreds of KB even for a modest image, versus a text
            # completion's payload which is a few hundred bytes to a
            # couple KB of plain text. requests' `timeout` bounds the
            # WRITE (upload) phase of the request, not just how long the
            # server takes to respond -- a live run hit exactly this:
            # `TimeoutError('The write operation timed out')` while
            # still trying to SEND a vision request, well before Groq's
            # server even started processing it. The default
            # GROQ_REQUEST_TIMEOUT_SECONDS (20s) is generous for a text
            # completion's tiny payload but was too tight for finishing
            # the upload of an image payload over a slower connection --
            # see GroqVLM._generate's own call for the longer timeout it
            # passes here instead. Defaults to
            # GROQ_REQUEST_TIMEOUT_SECONDS when the caller doesn't
            # override it, unchanged from before this parameter existed.
            timeout=timeout if timeout is not None else GROQ_REQUEST_TIMEOUT_SECONDS,
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
    elif _is_transient_json_validate_failure(resp):
        # See _is_transient_json_validate_failure's own docstring for
        # exactly why this specific 400 shape gets one retry -- same
        # "give Groq a second shot before falling back" reasoning as the
        # 429 branch above, just triggered by a different response shape
        # and with a short fixed delay instead of Groq's own
        # rate-limit-specific retry-after value (which doesn't apply
        # here at all).
        _log(
            f"400 json_validate_failed for {model} (empty failed_generation -- "
            f"looks transient, see this project's own comment) -- retrying once "
            f"after {_JSON_VALIDATE_RETRY_DELAY_SECONDS:.1f}s"
        )
        time.sleep(_JSON_VALIDATE_RETRY_DELAY_SECONDS)
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
        _maybe_start_cooldown(model, retry_after)
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
    # Uses GROQ_SMALL_MODEL from config.py, not a hardcoded literal --
    # this file's own model string went stale once before (hardcoded
    # "llama-3.1-8b-instant" here kept working right up until Groq
    # decommissioned it in 2026-08, at which point this smoke test would
    # have started failing with a confusing 404 for a reason that had
    # nothing to do with groq_client.py's own code), so importing the
    # live value config.py already tracks means this can't drift out of
    # sync with the rest of the project a second time.
    question = sys.argv[1] if len(sys.argv) > 1 else "Say hello in five words."
    result = groq_chat_completion(
        messages=[{"role": "user", "content": question}],
        model=GROQ_SMALL_MODEL,
        node="smoke_test",
    )
    print(result["choices"][0]["message"]["content"])

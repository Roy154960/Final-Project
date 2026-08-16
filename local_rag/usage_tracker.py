"""
Three small, file-backed logs, all under logs/ at the project root, that
together close checklist items #6 (request-id tracing), #13's rate-limit
half (Groq's OWN limits, not this server's -- see agents/api.py for the
server-side rate limiter/timeouts, which is a separate concern from
anything in this file), and #15 (per-request cost tracking):

  logs/groq_rate_limits.json
      One small JSON object, one entry per Groq model this process has
      actually called, updated from the REAL rate-limit response headers
      Groq sends on every call (see
      https://console.groq.com/docs/rate-limits#rate-limit-headers) --
      never a hand-rolled counter that could drift from what Groq's
      org-level limits actually are. get_usage_snapshot() below is the
      SAFE subset of this file: rate-limit numbers only, no cost, no
      tokens, no prompts -- this is the one thing in this module a
      person (not just the dev running the server) is meant to see, via
      agents/api.py's GET /v1/usage and the small usage badge at the top
      of the chat UI (see frontend/src/App.tsx / agents/static/chat.html).

  logs/cost_log.jsonl
      Append-only, one line per completed LLM/VLM call (Groq or the
      local Ollama fallback), with token counts and a reference cost in
      USD (see GROQ_LIST_PRICES_PER_1M's own docstring below for what
      that number actually means on Groq's free tier). This is
      DEV-ONLY -- see this module's own top-level docstring point above
      and agents/api.py's own docstring for the HTTP boundary that keeps
      it that way: nothing in this file is ever serialized into an HTTP
      response body. The person chatting sees the rate-usage badge, not
      this. Only someone with filesystem access to the machine running
      the server (i.e. Dominic) ever sees this file's contents, exactly
      the "only accessible to the dev, as a saved file" split this was
      built for.

  logs/request_trace.jsonl
      Append-only, one line per LangGraph node visit, written by
      agents/tracing.py's traced_node() wrapper (see that module's own
      docstring for why the wrapping happens at the graph-assembly layer
      in agents/graph.py rather than inside every individual specialist).
      Also dev-only, for the same reason as cost_log.jsonl above -- a
      request_id threading through this file lets a dev reconstruct one
      turn's whole path (which specialist, how long each node took, what
      it routed to next) the way the course's own Part 6 "What to Log"
      slide describes, without needing a hosted tracing product.

All three are plain JSON/JSONL specifically so they're grep/jq-able
without any extra tooling -- consistent with the course's own "structured
JSON, not prose... you will want to query this" guidance (Part 6).
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_STATE_PATH = LOGS_DIR / "groq_rate_limits.json"
COST_LOG_PATH = LOGS_DIR / "cost_log.jsonl"
TRACE_LOG_PATH = LOGS_DIR / "request_trace.jsonl"

# One process-wide lock for every write below -- these logs are written
# from request handlers that can genuinely run concurrently (FastAPI is
# async, multiple chat turns can be in flight at once), and both the
# JSONL append and the rate-limit JSON's read-modify-write need to not
# interleave with each other. Cheap: every write here is a few hundred
# bytes at most, held only for the duration of one open()/write()/close().
_lock = Lock()

# Reference LIST prices, USD per 1,000,000 tokens -- NOT what Dominic is
# actually billed. Groq's free tier (what GROQ_API_KEY on
# https://console.groq.com/keys gives you with no card on file) costs
# $0 regardless of these numbers; they're recorded in cost_log.jsonl
# purely so a dev skimming that file later has a feel for "what this
# session would have cost on a paid plan," the same spirit as the
# course's own Part 11 PRICING dict ("verify against current provider
# pricing before you quote anything" -- these were checked against
# Groq's published rates as of 2026-08; re-check
# https://groq.com/pricing before using this for anything that actually
# matters financially).
GROQ_LIST_PRICES_PER_1M = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "qwen/qwen3.6-27b": {"input": 0.60, "output": 3.00},
}

# The rate-limit response headers Groq documents at
# https://console.groq.com/docs/rate-limits#rate-limit-headers, mapped
# to the short keys this module persists them under. Note the docs'
# own caveat, preserved here rather than "fixed": x-ratelimit-limit-requests
# / x-ratelimit-remaining-requests / x-ratelimit-reset-requests ALWAYS
# refer to Requests Per Day (RPD), not RPM, despite the header name --
# and x-ratelimit-*-tokens always refers to Tokens Per Minute (TPM), not
# a daily figure. Named rpd_*/tpm_* here (not requests_*/tokens_*) so a
# reader of groq_rate_limits.json isn't left to rediscover that mismatch
# themselves.
_RATE_LIMIT_HEADER_MAP = {
    "x-ratelimit-limit-requests": "rpd_limit",
    "x-ratelimit-remaining-requests": "rpd_remaining",
    "x-ratelimit-reset-requests": "rpd_reset",
    "x-ratelimit-limit-tokens": "tpm_limit",
    "x-ratelimit-remaining-tokens": "tpm_remaining",
    "x-ratelimit-reset-tokens": "tpm_reset",
}


def new_request_id() -> str:
    """Short, unique-enough-for-one-server id threaded through every node
    of one graph turn (see agents/state.py's request_id field and
    agents/tracing.py's traced_node()) -- a full uuid4 would work too,
    but a 12-hex-char slice is plenty for grepping one turn's own lines
    out of request_trace.jsonl/cost_log.jsonl by eye, and stays short
    enough to log alongside every single node visit without the log
    itself becoming mostly id text."""
    return uuid.uuid4().hex[:12]


def _append_jsonl(path: Path, record: dict) -> None:
    """Never raises -- a logging failure (disk full, permissions) should
    degrade to a stderr note, not take down the chat turn it was trying
    to log, same "a missing side-effect is strictly less bad than a
    crashed request" preference this project applies elsewhere (see e.g.
    local_rag/personal_rag.py's _persist_personal_image docstring)."""
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            print(f"[usage_tracker] failed to write {path}: {e}", file=sys.stderr)


def _load_rate_limit_state() -> dict:
    if not RATE_LIMIT_STATE_PATH.exists():
        return {}
    try:
        with _lock:
            return json.loads(RATE_LIMIT_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[usage_tracker] failed to read {RATE_LIMIT_STATE_PATH}: {e}", file=sys.stderr)
        return {}


def _save_rate_limit_state(snapshot: dict) -> None:
    with _lock:
        try:
            RATE_LIMIT_STATE_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"[usage_tracker] failed to persist {RATE_LIMIT_STATE_PATH}: {e}", file=sys.stderr)


def record_node_trace(request_id: Optional[str], node: str, ms: float, **extra) -> None:
    """Called by agents/tracing.py's traced_node() once per LangGraph
    node visit -- see this module's own top docstring for the shape and
    purpose of logs/request_trace.jsonl."""
    record = {"ts": time.time(), "request_id": request_id, "node": node, "ms": ms, **extra}
    _append_jsonl(TRACE_LOG_PATH, record)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """0.0 for any model not in GROQ_LIST_PRICES_PER_1M (e.g. every local
    Ollama model -- those are never billed by anyone) or for zero-token
    calls. See GROQ_LIST_PRICES_PER_1M's own docstring for what a
    non-zero result here does and doesn't mean on Groq's free tier."""
    prices = GROQ_LIST_PRICES_PER_1M.get(model)
    if not prices or (not input_tokens and not output_tokens):
        return 0.0
    cost = (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
    return round(cost, 6)


def record_llm_call(
    *,
    backend: str,
    model: str,
    tier: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    request_id: Optional[str] = None,
    node: Optional[str] = None,
    thread_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
    fallback_reason: Optional[str] = None,
) -> None:
    """
    One line into logs/cost_log.jsonl per completed call, Groq OR local
    Ollama -- backend distinguishes which. Called from groq_client.py
    (after every successful Groq response) and from
    generation/fallback_generator.py / vlm/fallback_vlm.py /
    agents/llm_provider.py whenever a call actually fell back to Ollama
    (so the log shows the true backend split over a session, not just
    "every call was attempted via Groq").
    """
    record = {
        "ts": time.time(),
        "backend": backend,
        "model": model,
        "tier": tier,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "list_price_cost_usd": estimate_cost_usd(model, input_tokens, output_tokens) if backend == "groq" else 0.0,
        "request_id": request_id,
        "node": node,
        "thread_id": thread_id,
        "latency_ms": latency_ms,
    }
    if fallback_reason:
        record["fallback_reason"] = fallback_reason
    _append_jsonl(COST_LOG_PATH, record)


def record_groq_rate_limit_headers(model: str, headers) -> None:
    """
    Update the persisted per-model rate-limit snapshot from a live Groq
    response's own headers -- called from groq_client.py on EVERY Groq
    response (success or 429). `headers` is anything with a `.get()`
    (a `requests.Response.headers` CaseInsensitiveDict in practice).
    """
    snapshot = _load_rate_limit_state()
    entry = snapshot.setdefault(model, {})
    changed = False
    for header_name, key in _RATE_LIMIT_HEADER_MAP.items():
        val = headers.get(header_name)
        if val is not None:
            entry[key] = val
            changed = True
    if changed:
        entry["updated_at"] = time.time()
        entry["backend_status"] = "ok"
        entry.pop("last_error", None)
        entry.pop("last_error_at", None)
        _save_rate_limit_state(snapshot)


def record_groq_failure(model: str, reason: str) -> None:
    """Called from groq_client.py whenever a Groq call couldn't complete
    (network error, 4xx/5xx, unparsable body) -- kept separate from
    record_groq_rate_limit_headers so a failure is visible in
    groq_rate_limits.json (and therefore GET /v1/usage / the chat UI's
    own badge) even on the rare response that somehow arrives with no
    rate-limit headers at all, e.g. a connection error before Groq's own
    server ever responds."""
    snapshot = _load_rate_limit_state()
    entry = snapshot.setdefault(model, {})
    entry["backend_status"] = "fallback_to_local"
    entry["last_error"] = reason
    entry["last_error_at"] = time.time()
    _save_rate_limit_state(snapshot)


def get_usage_snapshot() -> dict:
    """
    The SAFE-to-expose subset of this module's state -- rate-limit
    numbers only, per model, straight off Groq's own response headers.
    Backs agents/api.py's GET /v1/usage, which the chat UI polls for the
    usage badge at the top of the page. Deliberately does NOT read
    cost_log.jsonl or request_trace.jsonl -- see this module's own top
    docstring for why those stay dev-only, filesystem-only.
    """
    return {
        "models": _load_rate_limit_state(),
        "free_tier": True,
        "docs": "https://console.groq.com/docs/rate-limits",
    }

"""
Together AI chat-completion client -- a thin `requests`-based HTTP
client, mirroring local_rag/groq_client.py's own shape (same "thin HTTP
client, not a new SDK dependency" choice groq_client.py already makes --
see llm_provider.py's own top docstring) so GroqFallbackChatModel can add
a second hosted provider without changing how it already talks to Groq.

Diffed against the real groq_client.py (see its own module docstring):
this file deliberately does NOT replicate its 429-retry-with-sleep,
per-model cooldown, or usage_tracker rate-limit-header recording --
those exist because Groq's free tier needs babying its account-level
limits across a whole session. Together sits behind Groq in the
fallback chain specifically to relieve pressure on that same small
budget, so there's no matching "protect Together's own limits" need
yet; a raw 429/5xx here just falls straight through to the next link
in the chain (local Ollama) via TogetherUnavailableError, the same way
a Groq failure always could. Worth adding if Together itself starts
getting hit hard enough to want its own cooldown tracking.

Scope: wired in for EVERY small-tier call site in agents/ -- not just
supervisor.py's routing decision and contextualize.py's follow-up
rewrite, but specialists.py's own small-tier calls too (see
llm_provider.py's get_chat_model(): the Together branch is gated on
`tier == "small"` alone, not on which node is calling). Not wired into
the large tier anywhere: Groq's large-tier budget isn't the bottleneck
this client exists to relieve, and the person asked for this scoped to
small-tier only.

Together's REST API is OpenAI-compatible (https://docs.together.ai/docs/quickstart,
https://docs.together.ai/docs/openai-compatibility) -- same request/
response JSON shape Groq's own API already uses, which is why this
module's request-building and response-parsing mirror groq_client.py's
presumed shape closely enough that llm_provider.py's Together branch
(added in this same change) reuses the exact same
_lc_message_to_openai_dict / _openai_message_to_ai_message helpers it
already had for Groq, unchanged.
"""

from typing import Any, Optional

import requests

from config import TOGETHER_API_KEY  # noqa: E402 -- mirrors groq_client.py's own `from config import GROQ_API_KEY`

# https://docs.together.ai/docs/openai-compatibility -- Together's own
# OpenAI-compatible base URL. api.together.ai/v1 also works per their
# newer docs; .xyz is the longer-standing one and still current as of
# this writing.
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Mirrors whatever timeout groq_chat_completion presumably already uses
# for its own `requests.post` call -- kept as a plain module constant so
# it's one line to tune if Together's small model turns out slower/
# faster to first-token than Groq's.
_TIMEOUT_SECONDS = 30


class TogetherAPIError(Exception):
    """Together returned a non-2xx response that ISN'T rate-limiting or
    a transient outage -- a genuine API-side error (bad request, auth
    failure, model not found, etc.), distinct from
    TogetherUnavailableError below the same way groq_client.py's own
    GroqAPIError is presumed distinct from GroqUnavailableError (see
    llm_provider.py's `except (GroqUnavailableError, GroqAPIError)`
    clause, which treats both as "fall back," so the split matters less
    for that call site than it might for future callers that want to
    distinguish "give up" from "retry elsewhere")."""


class TogetherUnavailableError(Exception):
    """Together is rate-limiting (429), returning a 5xx, or unreachable
    (connection error / timeout) -- the cases GroqFallbackChatModel's
    Together branch should treat as "fall through to the next provider
    in the chain," mirroring which cases groq_client.py's own
    GroqUnavailableError is presumed to cover for Groq."""


def together_chat_completion(
    *,
    messages: list,
    model: str,
    tools: Optional[list] = None,
    tool_choice: Any = None,
    response_format: Optional[dict] = None,
    temperature: float = 0.0,
    node: Optional[str] = None,
    tier: Optional[str] = None,
) -> dict:
    """
    Deliberately the SAME keyword signature as groq_chat_completion
    (same OpenAI-shaped `messages`/`tools` in, same raw OpenAI-shaped
    response dict out) so llm_provider.py's Together branch is a
    near-copy of its existing Groq branch rather than needing its own
    bespoke request/response handling.

    `node`/`tier` accepted for signature parity with groq_chat_completion
    (llm_provider.py passes both through unconditionally on every call)
    but NOT currently used here -- groq_client.py may thread these into
    this project's own usage_tracker.py for per-node cost logging; this
    function doesn't do that yet. Wire it in the same way if per-
    provider cost tracking should cover Together calls too.
    """
    if not TOGETHER_API_KEY:
        # Raised as TogetherAPIError, not TogetherUnavailableError -- a
        # missing key is a config problem that retrying elsewhere in the
        # chain won't fix by itself, but llm_provider.py's Together
        # branch (below) catches both the same way Groq's does, so this
        # distinction is about correct logging/diagnosis, not behavior.
        raise TogetherAPIError(
            "TOGETHER_API_KEY is not set -- this call falls through to "
            "the next link in the chain (local Ollama) instead. Fix "
            "(optional -- everything still works without it, just "
            "degrades straight to local on every Groq rate limit):\n"
            "  1. Get a key at https://api.together.ai/settings/api-keys\n"
            "  2. Put it in your .env file at the project root:\n"
            "       TOGETHER_API_KEY=your_together_api_key_here\n"
            "  3. Restart the server so config.py's load_dotenv() picks it up."
        )

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

    try:
        resp = requests.post(
            TOGETHER_API_URL,
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as e:
        raise TogetherUnavailableError(f"Together request failed: {e}") from e

    if resp.status_code == 429:
        raise TogetherUnavailableError(f"Together rate-limited (429): {resp.text[:300]}")
    if resp.status_code >= 500:
        raise TogetherUnavailableError(f"Together server error ({resp.status_code}): {resp.text[:300]}")
    if resp.status_code >= 400:
        raise TogetherAPIError(f"Together API error ({resp.status_code}): {resp.text[:300]}")

    return resp.json()

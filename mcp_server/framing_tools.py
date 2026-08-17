"""
System A's ONLY connection to System B (framing_agent/, a separate
Google ADK + FastAPI service in its own container -- see that
package's own README.md for what it is and why it exists). This module
makes a plain `requests.post()` call over the network to System B's
`/quote` route; nothing in this file imports anything from the
framing_agent/ package, and nothing in framing_agent/ imports anything
from here. That's the whole architectural point being tested: two
independent stacks cooperating across a real network boundary, not a
function call dressed up as one.

Same "every public function returns a structured result and never
raises on a network failure" contract web_tools.py's own module
docstring already documents for System A's OTHER internet-facing
calls (Wikipedia, DuckDuckGo) -- extended here to "the other service
is a sibling container that might not be running yet, might have
crashed, or might just be slow," which is a more LIKELY failure mode
for this one specifically than a public API being down, since
framing-agent is a piece of THIS project's own infrastructure someone
has to remember to actually start.
"""

import os
import sys
from typing import Optional

import requests

_REQUEST_TIMEOUT_SECONDS = 15  # generous vs. web_tools.py's 8s -- a real
# quote request may include an ADK/Gemini round trip on System B's own
# side (see framing_agent/server.py's own _ADK_TIMEOUT_SECONDS=12), not
# just a fast local lookup.

_USER_AGENT = "multi-agent-pipeline-coursework/1.0 (educational project; no commercial use)"


def _log(msg: str) -> None:
    print(f"[framing_tools] {msg}", file=sys.stderr)


def _framing_agent_base_url() -> str:
    """
    Overridable via FRAMING_AGENT_URL so a containerized mcp-server can
    reach framing-agent by its docker-compose service name
    (http://framing-agent:8090 -- see docker-compose.yml) while a plain
    non-Docker dev run still defaults to localhost, the same
    "env var overrides a sensible local default" convention
    local_rag/config.py's own OLLAMA_HOST already uses.
    """
    return os.environ.get("FRAMING_AGENT_URL", "http://localhost:8090").rstrip("/")


def request_quote(
    width_cm: float,
    height_cm: float,
    medium: str,
    destination_country: str,
    frame_style: str = "",
) -> dict:
    """
    The function behind the `get_framing_quote` MCP tool (see server.py
    for the tool wrapper, agents/specialists.py's framing_quote_node for
    the specialist that calls it).

    POSTs to System B's own POST /quote route -- see
    framing_agent/server.py's QuoteRequest/QuoteResponse models for the
    exact JSON shape on the wire. Never raises: a connection failure,
    timeout, non-2xx status, or malformed response all degrade to the
    same {"available": False, ...} shape below, so the calling
    specialist can say plainly that the framing service isn't reachable
    right now, rather than the whole turn crashing because a sibling
    container happened to be down or still starting up.

    Args:
        width_cm, height_cm: artwork dimensions in centimeters.
        medium: free text, e.g. "oil on canvas", "watercolor".
        destination_country: free text country name, e.g. "Lebanon".
        frame_style: optional requested frame style; empty string for
            System B's own default.

    Returns:
        On success:
          {"available": True, "quote": {...}, "explanation": str,
           "explanation_source": "groq" | "ollama" | "template"}
          -- `quote` is System B's own pricing.compute_quote() shape
          verbatim (see that function's docstring on System B's side
          for every field); `explanation` is a ready-to-show
          natural-language paragraph, `explanation_source` says which
          of System B's own three tiers produced it -- "groq" (its
          first choice), "ollama" (local, free, tried automatically if
          Groq isn't configured or fails), or "template" (deterministic,
          if neither LLM tier worked) -- see framing_agent/server.py's
          own module docstring for the full fallback order. System A
          doesn't need to treat these differently, but showing the
          caller which one happened is honest, same "say plainly what
          happened" pattern this project's other tool results already
          follow.
        On failure:
          {"available": False, "quote": None, "explanation": None,
           "explanation_source": None, "error": str}
          -- `error` is a short, human-readable reason (connection
          refused, timeout, HTTP status, malformed JSON), never a raw
          stack trace, safe to surface directly in a chat answer.
    """
    url = f"{_framing_agent_base_url()}/quote"
    payload = {
        "width_cm": width_cm,
        "height_cm": height_cm,
        "medium": medium,
        "destination_country": destination_country,
        "frame_style": frame_style,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        _log(f"POST {url} failed: {e}")
        return {
            "available": False, "quote": None, "explanation": None,
            "explanation_source": None,
            "error": (
                "The framing & shipping quote service isn't reachable right now "
                f"(tried {url}). It may not be running -- see framing_agent/README.md."
            ),
        }

    if resp.status_code != 200:
        _log(f"POST {url} returned HTTP {resp.status_code}: {resp.text[:300]!r}")
        return {
            "available": False, "quote": None, "explanation": None,
            "explanation_source": None,
            "error": f"The framing & shipping quote service returned an error (HTTP {resp.status_code}).",
        }

    try:
        data = resp.json()
    except ValueError:
        _log(f"POST {url} returned non-JSON response: {resp.text[:300]!r}")
        return {
            "available": False, "quote": None, "explanation": None,
            "explanation_source": None,
            "error": "The framing & shipping quote service returned an unreadable response.",
        }

    return {
        "available": True,
        "quote": data.get("quote"),
        "explanation": data.get("explanation"),
        "explanation_source": data.get("explanation_source"),
        "error": None,
    }


def framing_agent_health() -> Optional[dict]:
    """
    Best-effort GET /health against System B -- used only by
    tool_status() (server.py) for a diagnostic "is System B up right
    now" check, never on the hot path of an actual quote request.
    Returns None on any failure (never raises), same degrade-not-crash
    contract as request_quote above.
    """
    url = f"{_framing_agent_base_url()}/health"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        _log(f"GET {url} failed: {e}")
        return None

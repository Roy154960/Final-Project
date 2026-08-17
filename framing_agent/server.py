"""
framing_agent/server.py

System B's actual network boundary -- the ONLY thing System A ever
talks to. A plain FastAPI app, deliberately: System A's own MCP tool
(mcp_server/framing_tools.py, on the other side of this boundary) calls
POST /quote over plain HTTP, inside the docker-compose network, never a
Python import of anything in this package. See this project's own
docker-compose.yml for the "framing-agent" service this runs as, on its
own port, over the same inmind-net bridge network every other service
already shares.

Three routes:
  - GET  /health            -- liveness probe (docker-compose's own
                                HEALTHCHECK, and a human curling it)
  - GET  /.well-known/agent.json -- a minimal A2A-style agent card
                                (name, description, skills, this
                                service's own URL) -- see its own
                                docstring below for exactly how much of
                                the real A2A protocol this does and
                                doesn't implement
  - POST /quote              -- the actual contract: structured
                                dimensions/medium/destination in,
                                a structured quote (+ a short
                                natural-language explanation) out

The LLM-written explanation (agent.py's ADK agent) is BEST-EFFORT, not
required for this endpoint to function, and tries THREE tiers in order
-- Groq, then local Ollama, then a deterministic template -- matching
this project's own "no paid APIs, Groq is the one hosted exception,
always with an automatic free local fallback" convention (see
local_rag/groq_client.py's own module docstring). If GROQ_API_KEY isn't
set, or the Groq call fails for any reason, this tries local Ollama
next (no key needed). If THAT also fails or isn't reachable, /quote
still returns the full deterministic quote from
pricing.compute_quote(), with a templated (non-LLM) explanation string
instead -- never a 500, never a missing quote, regardless of which (if
any) of the two LLM tiers actually worked. `explanation_source` in the
response tells the caller exactly which tier produced the explanation
("groq" / "ollama" / "template") -- this is the one place in the whole
System A + System B setup where you can see, per-request, exactly which
backend actually answered, rather than only knowing it from reading
this file's source.

This three-tier degrade is the same "a missing capability degrades, it
never breaks the response" convention System A's own specialists.py
follows throughout (e.g. _best_personal_image_result falling through to
None rather than raising) -- applied here across the network boundary
between the two systems, which is exactly where it matters most:
System A's own framing_tools.py already treats "System B is
unreachable" as a plain, expected outcome, but System A can only
degrade GRACEFULLY (not just avoid crashing) if System B's OWN internal
failures -- Groq down, Ollama not running, both -- are just as honestly
absorbed and reported back as structured data, not a generic error page.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from pricing import compute_quote

# Only meaningful for a standalone (non-Docker) run: loads
# framing_agent/.env, if one exists, into this process's os.environ --
# the exact same "copy .env.example to .env" convention this project's
# OWN local_rag/config.py already uses for the project root's .env, now
# extended to this package's own separate .env. Deliberately a no-op,
# not an error, if framing_agent/.env doesn't exist -- GROQ_API_KEY and
# OLLAMA_HOST both staying at their defaults (unset / localhost) is a
# supported, fully-working configuration (see this module's own
# docstring below), not a missing prerequisite. Under docker-compose,
# this call is harmless but redundant -- the container already has real
# env vars injected by Docker itself before Python even starts; there's
# no framing_agent/.env file inside the image for this to find
# (docker/framing_agent.Dockerfile's own COPY step never copies a .env
# file in).
load_dotenv()

app = FastAPI(
    title="InMind Framing & Shipping Quote Agent (System B)",
    description=(
        "Independent framing/shipping quote service -- a separate stack "
        "(Google ADK + FastAPI) System A's LangGraph pipeline calls over "
        "HTTP, never as a Python import."
    ),
    version="1.0.0",
)

_PORT = int(os.environ.get("PORT", "8090"))
# Best-effort budget PER TIER (Groq attempt, then separately the Ollama
# attempt) -- long enough for a real LLM call, short enough that a hung
# request still reaches the deterministic explanation within a time
# System A's own framing_tools.py timeout (see that file's own
# _REQUEST_TIMEOUT_SECONDS=15) can actually wait out even in the worst
# case (Groq times out, THEN Ollama also times out, THEN the template
# path runs) -- see _try_llm_explanation's own docstring for why two
# back-to-back timeouts this size still fit inside that budget.
_ADK_TIMEOUT_SECONDS = 6


def _log(msg: str) -> None:
    print(f"[framing_agent] {msg}", file=sys.stderr)


class QuoteRequest(BaseModel):
    # Deliberately NOT `Field(..., gt=0)` on width_cm/height_cm -- an
    # earlier version of this file used that constraint, which meant
    # FastAPI's own request validation rejected a non-positive
    # dimension with a bare HTTP 422 before pricing.compute_quote() ever
    # ran, silently bypassing that function's own carefully-written
    # graceful "error" field (see its docstring) and the specific,
    # human-readable message it produces. Confirmed by actually POSTing
    # width_cm=-5 during testing: the 422 path gave
    # framing_tools.request_quote() nothing but a generic "HTTP 422"
    # string to relay, while compute_quote()'s own validation says
    # exactly what's wrong. Plain `float` here lets EVERY dimension
    # value through to compute_quote(), which is the one place that
    # validation was always meant to live -- consistent with this
    # project's own "the tool owns its numbers, including deciding
    # they're invalid" convention (see pricing.py's own module
    # docstring).
    width_cm: float = Field(..., description="Artwork width in centimeters.")
    height_cm: float = Field(..., description="Artwork height in centimeters.")
    medium: str = Field(..., description="What the artwork is made of, e.g. 'oil on canvas'.")
    destination_country: str = Field(..., description="Shipping destination country, e.g. 'Lebanon'.")
    frame_style: str = Field("", description="Optional requested frame style; empty for the shop default.")


class QuoteResponse(BaseModel):
    quote: dict
    explanation: str
    explanation_source: str  # "groq" | "ollama" | "template" -- always tells the caller which tier produced it
    generated_at: str


def _groq_configured() -> bool:
    """
    Cheap pre-check, purely to skip a doomed Groq attempt fast: is
    GROQ_API_KEY present at all. NOT a guarantee the call will succeed
    (a bad/expired key still fails downstream, caught the same way by
    _try_llm_explanation's own try/except) -- just avoids paying an
    import + network round trip on the common "never configured this"
    case. There's no equivalent pre-check for Ollama: it needs no key,
    so a genuinely unreachable/not-running local server is
    indistinguishable from "not configured" here, and both are handled
    identically -- by attempting the call and catching the failure.
    """
    return bool(os.environ.get("GROQ_API_KEY"))


def _template_explanation(quote: dict) -> str:
    """
    The final, always-works fallback -- plain string formatting over
    compute_quote()'s own structured fields, never a guess at a number
    that isn't already sitting in `quote`. This is what /quote returns
    whenever BOTH the Groq and Ollama tiers are unconfigured, time out,
    or error for any reason -- see this module's own top docstring for
    why that's a deliberate three-tier degrade, not an afterthought.
    """
    if quote.get("error"):
        return quote["error"]

    frame = quote["frame"]
    glazing = quote["glazing"]
    shipping = quote["shipping"]
    dims = quote["dimensions_cm"]

    parts = [
        f"For a {dims['width']:.1f}cm x {dims['height']:.1f}cm {quote['medium']} piece, "
        f"a {frame['style']} frame runs ${frame['cost_usd']:.2f}."
    ]
    if not frame["style_recognized"]:
        parts.append("No specific frame style was recognized, so the shop's standard default was used.")
    if glazing["needed"]:
        parts.append(f"Glazing ({glazing['chosen']}) adds ${glazing['cost_usd']:.2f}.")
    else:
        parts.append("No glazing was included, which is standard for this kind of medium.")
    parts.append(
        f"Shipping to {shipping['destination_country']} is estimated at "
        f"${shipping['cost_usd']:.2f} (~{shipping['estimated_weight_kg']:.1f}kg, "
        f"{shipping['zone']} zone)."
    )
    if not shipping["destination_recognized"]:
        parts.append(
            "That destination wasn't in the shop's own rate table, so this shipping "
            "figure is a rougher estimate than usual."
        )
    parts.append(f"Estimated total: ${quote['subtotal_usd']:.2f}.")
    return " ".join(parts)


async def _try_llm_explanation(req: QuoteRequest, quote: dict) -> tuple[str, str]:
    """
    (explanation, source) -- source is "groq" or "ollama" ONLY if that
    specific tier's ADK round trip genuinely completed; ANY failure at
    either tier (missing/bad credentials, a google-adk/litellm import
    error, an unreachable Ollama server, a timeout) falls through to the
    next tier. The deterministic template is the final tier and never
    fails -- see _template_explanation's own body. Never raises up into
    the /quote route.

    Order: Groq is tried first, but ONLY if GROQ_API_KEY is actually
    set (skip fast otherwise, see _groq_configured's own docstring).
    Local Ollama is ALWAYS attempted next (or first, if no Groq key at
    all) -- it needs no key, so there's no equivalent "skip fast"
    pre-check for it; a genuinely unreachable/not-running Ollama server
    is handled by the exact same try/except as any other Ollama-side
    failure, falling through to the template.
    """
    if quote.get("error"):
        # Nothing sensible for an LLM to explain about a request that
        # failed validation -- go straight to the template path, which
        # already just relays quote["error"] plainly (see its own body).
        return _template_explanation(quote), "template"

    if _groq_configured():
        try:
            result = await asyncio.wait_for(_run_adk_agent(req, "groq"), timeout=_ADK_TIMEOUT_SECONDS)
            if result:
                return result, "groq"
            _log("Groq agent returned no text -- trying local Ollama next.")
        except asyncio.TimeoutError:
            _log(f"Groq agent call exceeded {_ADK_TIMEOUT_SECONDS}s -- trying local Ollama next.")
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see this module's + agent.py's own docstrings
            _log(f"Groq agent call failed ({type(exc).__name__}: {exc}) -- trying local Ollama next.")
    else:
        _log("GROQ_API_KEY not set -- skipping straight to local Ollama.")

    try:
        result = await asyncio.wait_for(_run_adk_agent(req, "ollama"), timeout=_ADK_TIMEOUT_SECONDS)
        if result:
            return result, "ollama"
        _log("Ollama agent returned no text -- falling back to the template explanation.")
    except asyncio.TimeoutError:
        _log(f"Ollama agent call exceeded {_ADK_TIMEOUT_SECONDS}s -- falling back to the template explanation.")
    except Exception as exc:  # noqa: BLE001
        _log(f"Ollama agent call failed ({type(exc).__name__}: {exc}) -- falling back to the template explanation.")

    return _template_explanation(quote), "template"


async def _run_adk_agent(req: QuoteRequest, backend: str) -> str | None:
    """
    The actual google-adk round trip for ONE backend ("groq" or
    "ollama") -- imports agent.py's build_agent() lazily (so a
    missing/broken google-adk[extensions] install never breaks module
    import, only this one call path), runs exactly one turn through
    InMemoryRunner, and returns the final response text, or None if no
    final-response event was ever produced. Every exception this raises
    is caught by _try_llm_explanation above, never here.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from agent import build_agent

    runner = InMemoryRunner(agent=build_agent(backend), app_name="framing_agent")
    session_id = f"quote-{backend}-{id(req)}-{datetime.now(timezone.utc).timestamp()}"
    await runner.session_service.create_session(
        app_name="framing_agent", user_id="system-a", session_id=session_id
    )

    prompt = (
        f"Quote framing and shipping for a {req.width_cm}cm x {req.height_cm}cm "
        f"{req.medium} piece shipping to {req.destination_country}."
        + (f" Requested frame style: {req.frame_style}." if req.frame_style else "")
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(
        user_id="system-a", session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            return event.content.parts[0].text
    return None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "framing-agent",
        "groq_configured": _groq_configured(),
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        # NOT a liveness check of Ollama itself -- that would need an
        # actual network call, which a GET /health handler shouldn't
        # make on every poll. Whether Ollama is ACTUALLY reachable right
        # now is only observable from a real /quote call's own
        # "explanation_source" field.
    }


@app.get("/.well-known/agent.json")
def agent_card() -> dict:
    """
    A minimal, hand-written A2A-style agent card -- enough for a caller
    to discover what this service is and where its one real capability
    lives, NOT a full implementation of the A2A protocol's own task
    lifecycle (task submission/polling, streaming updates, pushed
    artifacts). System A calls the plain POST /quote route below
    directly, over a known docker-compose service URL, rather than
    negotiating through this card at request time -- that's a
    deliberate scope cut for this coursework project, documented here
    rather than silently implied: implementing the full A2A task
    lifecycle on top of this same /quote logic is a reasonable
    follow-up, not required for System A and System B to genuinely be
    two independent stacks cooperating over a real network boundary.
    """
    return {
        "name": "InMind Framing & Shipping Quote Agent",
        "description": (
            "Given an artwork's dimensions, medium, and shipping destination, "
            "returns a framing, glazing, and shipping cost estimate."
        ),
        "url": f"http://framing-agent:{_PORT}",
        "version": "1.0.0",
        "skills": [
            {
                "id": "get_framing_quote",
                "description": "Compute a framing, glazing, and shipping quote for one artwork.",
                "endpoint": "/quote",
                "method": "POST",
            }
        ],
    }


@app.post("/quote", response_model=QuoteResponse)
async def quote(req: QuoteRequest) -> QuoteResponse:
    computed = compute_quote(
        width_cm=req.width_cm,
        height_cm=req.height_cm,
        medium=req.medium,
        destination_country=req.destination_country,
        frame_style=req.frame_style or None,
    )
    explanation, source = await _try_llm_explanation(req, computed)
    return QuoteResponse(
        quote=computed,
        explanation=explanation,
        explanation_source=source,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_PORT)


@app.get("/.well-known/agent.json")
def agent_card() -> dict:
    """
    A minimal, hand-written A2A-style agent card -- enough for a caller
    to discover what this service is and where its one real capability
    lives, NOT a full implementation of the A2A protocol's own task
    lifecycle (task submission/polling, streaming updates, pushed
    artifacts). System A calls the plain POST /quote route below
    directly, over a known docker-compose service URL, rather than
    negotiating through this card at request time -- that's a
    deliberate scope cut for this coursework project, documented here
    rather than silently implied: implementing the full A2A task
    lifecycle on top of this same /quote logic is a reasonable
    follow-up, not required for System A and System B to genuinely be
    two independent stacks cooperating over a real network boundary.
    """
    return {
        "name": "InMind Framing & Shipping Quote Agent",
        "description": (
            "Given an artwork's dimensions, medium, and shipping destination, "
            "returns a framing, glazing, and shipping cost estimate."
        ),
        "url": f"http://framing-agent:{_PORT}",
        "version": "1.0.0",
        "skills": [
            {
                "id": "get_framing_quote",
                "description": "Compute a framing, glazing, and shipping quote for one artwork.",
                "endpoint": "/quote",
                "method": "POST",
            }
        ],
    }


@app.post("/quote", response_model=QuoteResponse)
async def quote(req: QuoteRequest) -> QuoteResponse:
    computed = compute_quote(
        width_cm=req.width_cm,
        height_cm=req.height_cm,
        medium=req.medium,
        destination_country=req.destination_country,
        frame_style=req.frame_style or None,
    )
    explanation, source = await _try_llm_explanation(req, computed)
    return QuoteResponse(
        quote=computed,
        explanation=explanation,
        explanation_source=source,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_PORT)

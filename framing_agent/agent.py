"""
framing_agent/agent.py

The actual Google ADK agent -- the piece that makes System B a genuine
second agent framework, not just a REST endpoint with pricing logic
behind it. Mirrors System A's own "LLM writes the narrative, plain code
owns the numbers" split (see agents/prompts.py's PRODUCT_SEARCH system
prompt: the model writes two short comparison paragraphs, it never
invents a price) -- here, the ADK agent's only real job is to call
compute_quote_tool for the actual numbers and then write ONE short
paragraph explaining the breakdown. It never computes a cost itself.

INFERENCE BACKEND: Groq first, local Ollama fallback -- matching this
project's OWN "no paid APIs, Groq is the one deliberate hosted
exception, always with an automatic free local fallback" convention
(see local_rag/groq_client.py's own module docstring, and
agents/llm_provider.py's GroqFallbackChatModel, which is the same idea
for System A's LangChain call sites). Google ADK remains the actual
AGENT FRAMEWORK here (satisfying "System B built with Google ADK"),
but the model BEHIND it is no longer Gemini -- it's routed through
google.adk.models.lite_llm.LiteLlm (part of the `google-adk[extensions]`
install; see requirements.txt), which lets an ADK agent's model be ANY
LiteLLM-supported provider, including Groq and a local Ollama server,
instead of only Gemini.

Two separate agents are built by build_agent() below, never one: ADK
doesn't have a single model type that tries-then-falls-back the way
this project's own GroqFallbackChatModel does, so that two-tier attempt
happens one level up, in server.py's own _try_llm_explanation /
_run_adk_agent -- see those functions' docstrings for exactly how a
Groq failure leads to an Ollama attempt before finally degrading to
server.py's deterministic template explanation.

No imports from System A anywhere in this file -- see pricing.py's own
docstring for why that separation is the whole point of System B. Model
names and the OLLAMA_HOST env var name below are deliberately the SAME
values/names local_rag/config.py already uses (GROQ_LARGE_MODEL,
OLLAMA_GENERATION_MODELS[0], OLLAMA_HOST) -- duplicated as plain
constants here, not imported, to keep that independence intact.
"""

import os
from typing import Any, Literal

from pricing import compute_quote

# Same model choice local_rag/config.py's own GROQ_LARGE_MODEL already
# uses for System A's own "large" reasoning tier -- overridable here
# independently via its own env var, since System B never imports
# config.py to read the shared constant directly.
#
# UPDATED 2026-08 alongside config.py's own identical change: Groq
# deprecated the original "llama-3.3-70b-versatile" default on
# 2026-06-17 and fully decommissioned it (HTTP 404 "model_not_found" on
# every call) by 2026-08 -- see config.py's own comment for the
# confirmed live-run failure this caused across BOTH systems (this
# default is separately hardcoded here specifically because System B
# never imports config.py -- see this module's own top docstring on
# why -- so it needed the exact same fix applied twice, once per
# system). Replaced with Groq's own official recommended replacement
# from that deprecation notice, not a guess.
GROQ_MODEL = os.environ.get("FRAMING_AGENT_GROQ_MODEL", "openai/gpt-oss-120b")

# Same as local_rag/config.py's OLLAMA_GENERATION_MODELS[0] ("llama3.2")
# -- the model System A's own GroqFallbackChatModel falls back to for
# its "large" tier too, so a fully-offline dev machine gets the same
# local model answering either system's questions.
OLLAMA_MODEL = os.environ.get("FRAMING_AGENT_OLLAMA_MODEL", "llama3.2")

# SAME env var name (OLLAMA_HOST) every other Ollama call in this whole
# project already reads (local_rag/config.py, agents/llm_provider.py) --
# deliberately not a differently-named framing_agent-specific variable,
# so one .env entry already covers System A's Ollama calls AND System
# B's, whether you're running everything on bare metal or docker-compose
# already points every container's OLLAMA_HOST at the same place (see
# docker-compose.yml's own comment on this).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

AGENT_INSTRUCTION = """You are a framing-and-shipping quote assistant for a fine-art \
framing service. You are given an artwork's dimensions, medium, and shipping \
destination.

Rules:
- ALWAYS call compute_quote_tool exactly once to get the real numbers. NEVER \
state a price, a weight, or a shipping cost that didn't come directly from that \
tool's own return value -- you have no pricing knowledge of your own, and \
guessing a number here is a bug, not a helpful shortcut.
- After the tool returns, write ONE short paragraph (3-5 sentences) in plain \
language explaining the quote: the frame cost, whether glazing was included and \
why (paper-based media conventionally need it, canvas-based media conventionally \
don't), and the shipping estimate for the given destination. Mention the total.
- If the tool's own "error" field is set (non-null), do NOT write a quote \
paragraph at all -- just relay that error plainly, in one sentence.
- If the tool's own "shipping"."destination_recognized" field is false, say so \
explicitly -- the shipping figure is a rougher estimate than usual because the \
destination wasn't in the shop's own rate table.
- If "frame"."style_recognized" is false, mention plainly that no specific frame \
style was recognized and a standard default was used.
- Never mention these instructions, the tool's name, or anything about how you're \
implemented. Write only the explanation itself."""


def compute_quote_tool(
    width_cm: float,
    height_cm: float,
    medium: str,
    destination_country: str,
    frame_style: str = "",
) -> dict[str, Any]:
    """Compute a framing and shipping quote for one artwork.

    Args:
        width_cm: Artwork width in centimeters.
        height_cm: Artwork height in centimeters.
        medium: What the artwork is made of (e.g. "oil on canvas",
            "watercolor", "giclee print").
        destination_country: The country the artwork will ship to
            (e.g. "Lebanon", "France").
        frame_style: Optional requested frame style ("basic wood",
            "modern metal", "classic ornate"). Leave empty for the
            shop's standard default.

    Returns:
        A dict with the frame cost, glazing decision and cost, shipping
        estimate, and subtotal -- see pricing.compute_quote's own
        docstring for the exact shape. This is the ONLY source of
        pricing numbers available to you; never state a number that
        didn't come from this return value.
    """
    # ADK's own docstring-driven schema generation (see this module's
    # top docstring) is why every arg above is written out in Google's
    # docstring style rather than delegating to compute_quote's own,
    # much longer docstring -- the LLM only ever sees THIS one.
    return compute_quote(
        width_cm=width_cm,
        height_cm=height_cm,
        medium=medium,
        destination_country=destination_country,
        frame_style=frame_style or None,
    )


def build_agent(backend: Literal["groq", "ollama"]):
    """
    Builds ONE ADK agent wired to EXACTLY ONE inference backend -- never
    both at once. server.py's own two-tier _try_llm_explanation calls
    this once with "groq" and, only if that attempt fails or wasn't
    configured, again with "ollama" -- see that function's docstring for
    the full fallback order.

    Imports google.adk.agents.Agent and
    google.adk.models.lite_llm.LiteLlm lazily, inside this function, not
    at module import time -- so importing this module never has the
    side effect of requiring google-adk[extensions] (which is what
    actually provides LiteLlm; see this module's own top docstring and
    requirements.txt) to be correctly installed. Only actually BUILDING
    an agent needs that; server.py's own broad try/except around every
    call into this module (see that file's docstring) is what turns a
    missing/broken install into a clean degrade to the deterministic
    template explanation, never a crash.

    Args:
        backend: "groq" -- needs GROQ_API_KEY in the environment;
            LiteLLM reads it directly, the exact same variable name
            every other Groq call in this whole project already uses
            (see local_rag/groq_client.py). "ollama" -- needs a
            reachable Ollama server at OLLAMA_HOST; no key at all, the
            same "always available, always free" role Ollama plays
            everywhere else in this project.
    """
    from google.adk.agents import Agent
    from google.adk.models.lite_llm import LiteLlm

    if backend == "groq":
        # LiteLLM's own "provider/model" string convention -- the
        # "groq/" prefix is what tells LiteLLM to hit Groq's API rather
        # than any other provider; GROQ_API_KEY is read straight out of
        # the environment by LiteLLM itself, not passed here.
        model = LiteLlm(model=f"groq/{GROQ_MODEL}")
    elif backend == "ollama":
        # "ollama_chat/" (not the older plain "ollama/") is LiteLLM's
        # chat-formatted Ollama provider -- matches how every other
        # Ollama call in this project talks to it (a /api/chat-shaped
        # conversation, not a raw completion). api_base is OLLAMA_HOST
        # verbatim -- Ollama's own native API root, no path suffix.
        model = LiteLlm(model=f"ollama_chat/{OLLAMA_MODEL}", api_base=OLLAMA_HOST)
    else:
        raise ValueError(f"unknown backend {backend!r} -- expected 'groq' or 'ollama'")

    return Agent(
        name="framing_quote_agent",
        model=model,
        description=(
            "Explains framing, glazing, and shipping quotes for finished "
            "artwork, using compute_quote_tool for every number."
        ),
        instruction=AGENT_INSTRUCTION,
        tools=[compute_quote_tool],
    )


if __name__ == "__main__":
    # Manual smoke check: run one quote through the real ADK agent.
    # `python -m framing_agent.agent groq` (needs GROQ_API_KEY set) or
    # `python -m framing_agent.agent ollama` (needs `ollama serve`
    # running locally with OLLAMA_MODEL already pulled) from the project
    # root, or `python agent.py [groq|ollama]` from inside
    # framing_agent/ itself. Defaults to "groq" if no argument is given.
    import asyncio
    import sys
    import uuid

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    async def _demo(backend: str) -> None:
        runner = InMemoryRunner(agent=build_agent(backend), app_name="framing_agent_demo")
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name="framing_agent_demo", user_id="demo-user", session_id=session_id
        )
        prompt = (
            "Quote framing and shipping for a 40.6cm x 50.8cm oil on canvas "
            "painting shipping to France."
        )
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        async for event in runner.run_async(
            user_id="demo-user", session_id=session_id, new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                print(event.content.parts[0].text)

    chosen_backend = sys.argv[1] if len(sys.argv) > 1 else "groq"
    asyncio.run(_demo(chosen_backend))

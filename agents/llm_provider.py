"""
Groq-first, local-Ollama-fallback LangChain chat model for every
reasoning call site in agents/ -- supervisor.py's routing decision,
contextualize.py's follow-up rewrite, and every specialist's own
create_react_agent / plain LLM call in specialists.py. Mirrors
local_rag/generation/fallback_generator.py's "online first, automatic
local fallback" shape (see that module's own docstring), adapted to
LangChain's BaseChatModel interface so bind_tools() / create_react_agent
/ ainvoke() all keep working unchanged at every existing call site --
get_chat_model() below is the one function those call sites should call
instead of constructing ChatOllama directly.

Why a custom BaseChatModel instead of langchain_groq.ChatGroq: this
project deliberately calls Groq via plain `requests`
(local_rag/groq_client.py) rather than adding the `groq` SDK or
`langchain_groq` as a new dependency -- same "thin HTTP client, not a
new SDK dependency" choice groq_client.py already makes. Doing the
fallback INSIDE one BaseChatModel (rather than, say,
picking ChatGroq vs ChatOllama once at startup) is also what makes the
fallback genuinely per-call and automatic: Groq going down/rate-limited
mid-conversation degrades this turn to the local model without the
caller (supervisor.py/specialists.py/contextualize.py) needing to know
or care which backend actually answered.

How tool-calling survives the fallback: GroqFallbackChatModel.bind_tools()
converts every tool to the same OpenAI tool-call JSON shape ChatOllama's
own bind_tools() already uses (see langchain_ollama's ChatOllama.bind_tools
-- confirmed by reading its source directly rather than assumed) and
attaches it as a `tools` kwarg via the standard LangChain `.bind()`
mechanism. _generate/_agenerate below read that same `tools` kwarg for
the Groq branch AND forward it, completely unchanged, into the local
ChatOllama fallback's own _generate/_agenerate -- Ollama's /api/chat
endpoint accepts that exact OpenAI-shaped tool schema natively, so one
tool-conversion happens regardless of which backend a given call
actually lands on. This matters most for retrieval_qa (specialists.py),
the one specialist whose react agent structurally cannot function
without tool-calling.

Structured JSON routing (supervisor.py's `ollama_format`): forwarded
ONLY to the local ChatOllama fallback, where it's still Ollama's own
native `format=<json schema>` structured-decoding parameter (unchanged
from before Groq was added). The Groq branch doesn't have that exact
mechanism available for these particular models, so it instead sets
`response_format={"type": "json_object"}` whenever `ollama_format` is
set at all -- see https://console.groq.com/docs/structured-outputs:
llama-3.3-70b-versatile/llama-3.1-8b-instant aren't on Groq's small
strict-schema-mode model list, so JSON Object Mode (valid JSON
guaranteed, schema NOT guaranteed) is the right degrade rather than an
unsupported request. supervisor.py's own downstream handling already
treats a malformed/schema-violating JSON response as one more
"fallback route" case (see its own module docstring's four safety
nets), so this degrade is absorbed by machinery that already existed
before Groq was added, not a new failure mode this file introduces.
"""

import asyncio
import functools
import json
import sys
from pathlib import Path
from typing import Any, List, Literal, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama


def _find_pipeline_root() -> Path:
    """
    Locate the directory that actually contains config.py -- the same
    duplicated-per-module helper agents/specialists.py, agents/guardrails.py,
    and agents/api.py each already carry their own copy of (see e.g.
    agents/api.py's own docstring on why this is duplicated rather than
    imported: no other dependency on local_rag/'s internals otherwise,
    shouldn't gain one just to reuse eight lines of path-checking).
    """
    here = Path(__file__).resolve().parent  # agents/
    parent = here.parent
    grandparent = parent.parent
    candidates = [
        parent / "config.py",
        parent / "local_rag" / "config.py",
        grandparent / "config.py",
        grandparent / "local_rag" / "config.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.parent
    raise ModuleNotFoundError(
        "Could not find config.py near agents/. Checked:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nEdit _find_pipeline_root() in agents/llm_provider.py to add your actual path."
    )


_PIPELINE_ROOT = _find_pipeline_root()
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from config import GROQ_LARGE_MODEL, GROQ_SMALL_MODEL, OLLAMA_GENERATION_MODELS, OLLAMA_NUM_CTX  # noqa: E402
from groq_client import GroqAPIError, GroqUnavailableError, groq_chat_completion  # noqa: E402

# Local-fallback model names -- mirrors agents/specialists.py's own
# _LARGE_REASONING_MODEL (OLLAMA_GENERATION_MODELS[0], "llama3.2") /
# _SMALL_REASONING_MODEL (OLLAMA_GENERATION_MODELS[2], "phi3") split.
# Duplicated here rather than imported from specialists.py specifically
# to avoid a circular import (specialists.py imports get_chat_model from
# THIS module) -- same "small, independent copy rather than a
# cross-module dependency" tradeoff _find_pipeline_root() above already
# makes.
_LOCAL_LARGE_MODEL = OLLAMA_GENERATION_MODELS[0]
_LOCAL_SMALL_MODEL = OLLAMA_GENERATION_MODELS[2]


def _lc_message_to_openai_dict(message: BaseMessage) -> dict:
    """Translate one LangChain message into the OpenAI-shaped dict Groq's
    REST API expects (https://console.groq.com/docs/openai) -- written
    out plainly here (rather than reusing langchain_groq's internal
    converter) since this project deliberately calls Groq via plain
    `requests` and doesn't depend on langchain_groq at all (see this
    module's own top docstring)."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    if isinstance(message, AIMessage):
        out: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                }
                for tc in message.tool_calls
            ]
        return out
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    # Anything else (rare -- FunctionMessage, etc.): degrade to a plain
    # user turn rather than raising, same "an unrecognized shape
    # shouldn't break the whole call" preference this project applies
    # elsewhere (e.g. graph.py's _resolve_forced_route).
    return {"role": "user", "content": str(message.content)}


def _openai_message_to_ai_message(message: dict) -> AIMessage:
    tool_calls = []
    for tc in message.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            args = {}
        tool_calls.append({"name": tc["function"]["name"], "args": args, "id": tc.get("id", "")})
    return AIMessage(content=message.get("content") or "", tool_calls=tool_calls)


class GroqFallbackChatModel(BaseChatModel):
    """
    One LangChain BaseChatModel that tries Groq first and falls back to a
    local ChatOllama instance on ANY failure -- see this module's own top
    docstring for the full reasoning. Built once per (tier, node)
    combination by get_chat_model() below and reused for the whole graph
    run, the same "build once, reuse across every visit" granularity
    every existing `llm_large`/`llm_small` pair in
    supervisor.py/specialists.py/contextualize.py already used for their
    own raw ChatOllama instances before this change.

    Fields (plain pydantic fields, per BaseChatModel's own convention):
        groq_model:    Groq model id to try first (e.g. "llama-3.3-70b-versatile").
        ollama_model:  local Ollama model to fall back to (e.g. "llama3.2").
        tier:          "large" or "small" -- logging/tracing metadata only,
                       never sent to either backend.
        node:          a short name for the call site (e.g. "supervisor",
                       "specialists.multi_hop") -- logging/tracing metadata
                       only, threaded into usage_tracker's cost log so a
                       dev can see which node a given call came from.
        ollama_format: forwarded ONLY to the local ChatOllama fallback --
                       see this module's own top docstring for what the
                       Groq branch does instead when this is set.
    """

    groq_model: str
    ollama_model: str
    tier: str = "large"
    node: Optional[str] = None
    ollama_format: Any = None
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "groq-fallback-chat-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Same conversion ChatOllama's own bind_tools does (see this
        module's top docstring) -- OpenAI tool-call JSON shape, attached
        via the standard `.bind()` mechanism so create_react_agent (and
        anything else that calls bind_tools) works unchanged against
        this class."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def _ollama_instance(self) -> ChatOllama:
        # Rebuilt fresh on each fallback rather than cached on self --
        # ChatOllama is a thin client wrapper with no real connection
        # held open (same point supervisor.py's own pre-Groq comment
        # already made about its two pre-built ChatOllama instances), and
        # this keeps GroqFallbackChatModel itself a plain, stateless-
        # enough pydantic model rather than needing __init__ overrides to
        # cooperate with BaseChatModel's own pydantic machinery.
        #
        # num_ctx=OLLAMA_NUM_CTX (config.py) is NOT optional here: with no
        # explicit context length, Ollama's own default for a model can be
        # the model's full trained max context (128K+ for some models),
        # which asks for a KV-cache buffer far bigger than most machines
        # have free RAM for -- a confirmed live failure (out-of-memory at
        # model-load time, not a clean error) on a fallback call this
        # class makes. Every local Ollama call in this project shares this
        # same config.py default rather than each hardcoding its own
        # number.
        kwargs: dict = {
            "model": self.ollama_model,
            "temperature": self.temperature,
            "num_ctx": OLLAMA_NUM_CTX,
        }
        if self.ollama_format is not None:
            kwargs["format"] = self.ollama_format
        return ChatOllama(**kwargs)

    def _log_fallback(self, exc: Exception) -> None:
        print(
            f"[llm_provider] Groq unavailable for node={self.node!r} "
            f"tier={self.tier!r} ({exc}) -- falling back to local Ollama "
            f"({self.ollama_model})",
            file=sys.stderr,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = [_lc_message_to_openai_dict(m) for m in messages]
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        response_format = {"type": "json_object"} if self.ollama_format is not None else None
        try:
            data = groq_chat_completion(
                messages=openai_messages,
                model=self.groq_model,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                temperature=self.temperature,
                node=self.node,
                tier=self.tier,
            )
            ai_message = _openai_message_to_ai_message(data["choices"][0]["message"])
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        except (GroqUnavailableError, GroqAPIError) as e:
            self._log_fallback(e)
        except Exception as e:  # noqa: BLE001 -- any other Groq-side surprise still shouldn't break the turn
            self._log_fallback(e)
        ollama = self._ollama_instance()
        return ollama._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = [_lc_message_to_openai_dict(m) for m in messages]
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        response_format = {"type": "json_object"} if self.ollama_format is not None else None
        try:
            # groq_chat_completion is a blocking `requests` call --
            # offloaded to a thread so it never blocks the event loop
            # every other concurrent chat turn is also running on (this
            # project's whole /chat endpoint is async -- see
            # agents/api.py's own docstring on why async is non-negotiable
            # here).
            loop = asyncio.get_running_loop()
            call = functools.partial(
                groq_chat_completion,
                messages=openai_messages,
                model=self.groq_model,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                temperature=self.temperature,
                node=self.node,
                tier=self.tier,
            )
            data = await loop.run_in_executor(None, call)
            ai_message = _openai_message_to_ai_message(data["choices"][0]["message"])
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        except (GroqUnavailableError, GroqAPIError) as e:
            self._log_fallback(e)
        except Exception as e:  # noqa: BLE001
            self._log_fallback(e)
        ollama = self._ollama_instance()
        return await ollama._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def get_chat_model(
    tier: Literal["small", "large"],
    *,
    node: Optional[str] = None,
    ollama_format: Any = None,
    temperature: float = 0.0,
) -> GroqFallbackChatModel:
    """
    The one function every reasoning call site in agents/ should call
    instead of constructing ChatOllama directly -- returns a
    GroqFallbackChatModel that tries Groq first (see this module's own
    top docstring) and transparently falls back to the SAME local Ollama
    model this project used for that tier before Groq was added (see
    specialists.py's own _LARGE_REASONING_MODEL / _SMALL_REASONING_MODEL,
    mirrored here as _LOCAL_LARGE_MODEL / _LOCAL_SMALL_MODEL).

    `node`: a short, stable label for the call site (e.g. "supervisor",
    "contextualize", "specialists.multi_hop") -- pure logging/tracing
    metadata, threaded into the dev-only cost log
    (local_rag/usage_tracker.py) so a session's log shows which node each
    call actually came from.

    `ollama_format`: passed straight through to the Ollama fallback ONLY
    -- see this module's own top docstring for what the Groq branch does
    instead when this is set (supervisor.py's structured routing decision
    is the one caller that sets this).
    """
    groq_model = GROQ_LARGE_MODEL if tier == "large" else GROQ_SMALL_MODEL
    ollama_model = _LOCAL_LARGE_MODEL if tier == "large" else _LOCAL_SMALL_MODEL
    return GroqFallbackChatModel(
        groq_model=groq_model,
        ollama_model=ollama_model,
        tier=tier,
        node=node,
        ollama_format=ollama_format,
        temperature=temperature,
    )

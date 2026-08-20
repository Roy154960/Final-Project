"""
Groq-first LangChain chat model for every reasoning call site in
agents/ -- supervisor.py's routing decision, contextualize.py's
follow-up rewrite, and every specialist's own create_react_agent /
plain LLM call in specialists.py. Large-tier calls fall back straight
to local Ollama on a Groq failure, same as always. Small-tier calls get
ONE MORE hosted hop first: Groq -> Together AI -> local Ollama, added
because supervisor/contextualize/specialists' small-tier calls all draw
from the SAME 8,000 TPM Groq ceiling for GROQ_SMALL_MODEL and can
exhaust it within seconds of each other in a single turn -- see
GroqFallbackChatModel's own class docstring below for the full
reasoning, and TOGETHER_SMALL_MODEL / TOGETHER_API_KEY in config.py for
how to opt in (entirely optional; nothing breaks if that key is unset).
Mirrors local_rag/generation/fallback_generator.py's "online first,
automatic local fallback" shape (see that module's own docstring),
adapted to LangChain's BaseChatModel interface so bind_tools() /
create_react_agent / ainvoke() all keep working unchanged at every
existing call site -- get_chat_model() below is the one function those
call sites should call instead of constructing ChatOllama directly.

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
the Groq branch, forward it unchanged into the Together branch
(Together's own REST API is OpenAI-compatible too -- same shape, no
second conversion needed), AND forward it, still unchanged, into the
local ChatOllama fallback's own _generate/_agenerate -- Ollama's
/api/chat endpoint accepts that exact OpenAI-shaped tool schema
natively, so one tool-conversion happens regardless of which of the
three backends a given call actually lands on. This matters most for
retrieval_qa (specialists.py), the one specialist whose react agent
structurally cannot function without tool-calling.

Structured JSON routing (supervisor.py's `ollama_format`): forwarded
ONLY to the local ChatOllama fallback, where it's still Ollama's own
native `format=<json schema>` structured-decoding parameter (unchanged
from before Groq was added). The Groq AND Together branches instead
both set `response_format={"type": "json_object"}` whenever
`ollama_format` is set at all -- JSON Object Mode (valid JSON
guaranteed, schema NOT guaranteed), not Groq's stricter
`json_schema`/`strict: true` mode
(https://console.groq.com/docs/structured-outputs; Together documents
the same `json_object` mode at
https://docs.together.ai/docs/json-mode). supervisor.py's own
downstream handling already treats a malformed/schema-violating JSON
response as one more "fallback route" case (see its own module
docstring's four safety nets), so this degrade is absorbed by machinery
that already existed before Groq was added, not a new failure mode this
file introduces.

WORTH REVISITING, not changed here: this project's ORIGINAL two Groq
models (llama-3.3-70b-versatile/llama-3.1-8b-instant, deprecated by
Groq 2026-06-17 -- see local_rag/config.py's own comment) genuinely
weren't on Groq's strict-schema-mode model list, which is why this file
settled on json_object mode. Their replacements
(openai/gpt-oss-120b/openai/gpt-oss-20b, config.py's new defaults) are
documented by Groq and LangChain's own ChatGroq reference as
SUPPORTING strict mode -- but at least one Groq community report
(Oct 2025) describes `strict: true` being silently ignored by
openai/gpt-oss-120b in practice, contradicting that documentation. Not
confident enough in either source alone to flip this file's actual
behavior without testing it against a real account first -- json_object
mode still works regardless of which of those two is currently true, so
staying on it here is the safe default, not a stale oversight. If you
want the stricter guarantee, test `response_format={"type":
"json_schema", "json_schema": {...}, "strict": true}` against your own
GROQ_API_KEY before switching this file over.
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

from config import (  # noqa: E402
    GROQ_LARGE_MODEL,
    GROQ_SMALL_MODEL,
    OLLAMA_GENERATION_MODELS,
    OLLAMA_NUM_CTX,
    TOGETHER_SMALL_MODEL,
)
from groq_client import GroqAPIError, GroqUnavailableError, groq_chat_completion  # noqa: E402
from together_client import TogetherAPIError, TogetherUnavailableError, together_chat_completion  # noqa: E402

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
    One LangChain BaseChatModel with a THREE-link fallback chain: Groq,
    then (small tier only) Together AI, then always local ChatOllama --
    see this module's own top docstring for the full reasoning. Built
    once per (tier, node) combination by get_chat_model() below and
    reused for the whole graph run, the same "build once, reuse across
    every visit" granularity every existing `llm_large`/`llm_small` pair
    in supervisor.py/specialists.py/contextualize.py already used for
    their own raw ChatOllama instances before this change.

    Why Together sits between Groq and Ollama, small tier only:
    supervisor.py's routing decision, contextualize.py's follow-up
    rewrite, AND specialists.py's own small-tier calls all draw from the
    SAME 8,000 TPM Groq ceiling for GROQ_SMALL_MODEL (config.py) -- a
    single turn can fire several of those calls within seconds of each
    other, and every one of them used to degrade straight to local phi3
    on the first 429. Together AI hosts the identical model
    (TOGETHER_SMALL_MODEL, config.py) behind its own independent
    rate-limit bucket, so a Groq rate limit now gets one more real,
    still-hosted shot before falling all the way down to the weaker
    local model. Large-tier calls are untouched: `together_model` is
    None for them (see get_chat_model() below), so `_generate`/
    `_agenerate`'s Together branch is simply skipped and the chain is
    exactly what it always was -- Groq, then Ollama.

    Fields (plain pydantic fields, per BaseChatModel's own convention):
        groq_model:     Groq model id to try first (e.g. "openai/gpt-oss-120b").
        together_model: Together AI model id to try second, or None to
                        skip Together entirely and go straight from Groq
                        to Ollama on failure (get_chat_model() only sets
                        this for tier="small").
        ollama_model:   local Ollama model to fall back to last (e.g. "llama3.2").
        tier:           "large" or "small" -- logging/tracing metadata only,
                        never sent to any backend.
        node:           a short name for the call site (e.g. "supervisor",
                        "specialists.multi_hop") -- logging/tracing metadata
                        only, threaded into usage_tracker's cost log so a
                        dev can see which node a given call came from.
        ollama_format:  forwarded ONLY to the local ChatOllama fallback --
                        see this module's own top docstring for what the
                        Groq/Together branches do instead when this is set.
        try_groq:       whether to attempt Groq at all -- False skips
                        straight to Together (if configured) or Ollama.
                        See get_chat_model()'s `use_groq` param.
    """

    groq_model: str
    together_model: Optional[str] = None
    try_groq: bool = True
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

    def _log_groq_fallback(self, exc: Exception) -> None:
        next_hop = f"Together AI ({self.together_model})" if self.together_model else f"local Ollama ({self.ollama_model})"
        print(
            f"[llm_provider] Groq unavailable for node={self.node!r} "
            f"tier={self.tier!r} ({exc}) -- falling back to {next_hop}",
            file=sys.stderr,
        )

    def _log_together_fallback(self, exc: Exception) -> None:
        print(
            f"[llm_provider] Together AI unavailable for node={self.node!r} "
            f"tier={self.tier!r} model={self.together_model!r} ({exc}) -- "
            f"falling back to local Ollama ({self.ollama_model})",
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

        if self.try_groq:
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
                self._log_groq_fallback(e)
            except Exception as e:  # noqa: BLE001 -- any other Groq-side surprise still shouldn't break the turn
                self._log_groq_fallback(e)

        if self.together_model is not None:
            try:
                data = together_chat_completion(
                    messages=openai_messages,
                    model=self.together_model,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    temperature=self.temperature,
                    node=self.node,
                    tier=self.tier,
                )
                ai_message = _openai_message_to_ai_message(data["choices"][0]["message"])
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            except (TogetherUnavailableError, TogetherAPIError) as e:
                self._log_together_fallback(e)
            except Exception as e:  # noqa: BLE001 -- any other Together-side surprise still shouldn't break the turn
                self._log_together_fallback(e)

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
        # Every blocking `requests` call below (Groq, Together) is
        # offloaded to a thread via this one loop reference -- acquired
        # once up front (rather than inside each provider's own try
        # block) so it's available for the Together branch even when
        # try_groq=False skips the Groq branch entirely. This project's
        # whole /chat endpoint is async (see agents/api.py's own
        # docstring on why async is non-negotiable here).
        loop = asyncio.get_running_loop()

        if self.try_groq:
            try:
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
                self._log_groq_fallback(e)
            except Exception as e:  # noqa: BLE001
                self._log_groq_fallback(e)

        if self.together_model is not None:
            try:
                # Same off-the-event-loop treatment as the Groq call
                # above, and for the same reason: together_chat_completion
                # is a blocking `requests` call, and this project's whole
                # /chat endpoint is async (see agents/api.py).
                together_call = functools.partial(
                    together_chat_completion,
                    messages=openai_messages,
                    model=self.together_model,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    temperature=self.temperature,
                    node=self.node,
                    tier=self.tier,
                )
                data = await loop.run_in_executor(None, together_call)
                ai_message = _openai_message_to_ai_message(data["choices"][0]["message"])
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            except (TogetherUnavailableError, TogetherAPIError) as e:
                self._log_together_fallback(e)
            except Exception as e:  # noqa: BLE001
                self._log_together_fallback(e)

        ollama = self._ollama_instance()
        return await ollama._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def get_chat_model(
    tier: Literal["small", "large"],
    *,
    node: Optional[str] = None,
    ollama_format: Any = None,
    temperature: float = 0.0,
    use_groq: Optional[bool] = None,
    use_together: Optional[bool] = None,
) -> GroqFallbackChatModel:
    """
    The one function every reasoning call site in agents/ should call
    instead of constructing ChatOllama directly -- returns a
    GroqFallbackChatModel with a three-link fallback chain for the small
    tier (Together AI -> Groq -> local Ollama, see `use_groq` below) and
    the original two-link chain for the large tier (Groq -> local
    Ollama, unchanged). Every call site (supervisor.py, contextualize.py,
    specialists.py) gets Together as its DEFAULT small-tier backend just
    by asking for tier="small" -- no node-name allowlist, no per-call
    kwargs needed.

    `node`: a short, stable label for the call site (e.g. "supervisor",
    "contextualize", "specialists.multi_hop") -- pure logging/tracing
    metadata, threaded into the dev-only cost log
    (local_rag/usage_tracker.py) so a session's log shows which node each
    call actually came from.

    `ollama_format`: passed straight through to the Ollama fallback ONLY
    -- see this module's own top docstring for what the Groq/Together
    branches do instead when this is set (supervisor.py's structured
    routing decision is the one caller that sets this).

    `use_groq` (default None = auto): None resolves PER TIER -- False
    (skip Groq) for tier="small", True (try Groq first, same as always)
    for tier="large". This is what makes Together the DEFAULT small-tier
    backend without any call site needing to opt in: every existing
    `get_chat_model("small", node=...)` call, unchanged, now goes
    straight to Together first, only touching Groq's shared 8,000 TPM
    small-tier budget at all if explicitly asked to (pass True). Large
    tier is untouched either way -- Groq stays first there since there's
    no TOGETHER_LARGE_MODEL to default to instead. Pass an explicit
    True/False yourself to override either tier's default, e.g. a call
    site that specifically WANTS Groq tried first for the small tier
    despite the new default.

    `use_together` (default None = auto): None means "Together AI if
    this is the small tier, never for large" -- the normal case for
    every call site in this project today. Pass True to force Together
    on even for a call that wouldn't normally get it (has no effect on
    the large tier, since there's no TOGETHER_LARGE_MODEL configured --
    see config.py). Pass False to disable Together for this call site
    even though it's small tier (falls straight from Groq to Ollama,
    the pre-Together behavior).
    """
    groq_model = GROQ_LARGE_MODEL if tier == "large" else GROQ_SMALL_MODEL
    want_groq = (tier != "small") if use_groq is None else bool(use_groq)
    want_together = True if use_together is None else bool(use_together)
    together_model = TOGETHER_SMALL_MODEL if (tier == "small" and want_together) else None
    ollama_model = _LOCAL_LARGE_MODEL if tier == "large" else _LOCAL_SMALL_MODEL
    return GroqFallbackChatModel(
        groq_model=groq_model,
        together_model=together_model,
        try_groq=want_groq,
        ollama_model=ollama_model,
        tier=tier,
        node=node,
        ollama_format=ollama_format,
        temperature=temperature,
    )

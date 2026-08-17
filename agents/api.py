"""
FastAPI wrapper around the Phase 2-4 agent graph (agents/graph.py), for
talking to the multi-agent system like a normal chatbot -- multi-turn,
one thread_id per conversation, history that survives across requests
(and across server restarts) -- without adding Open WebUI or any other
chat framework in front of it.

Why this needed to be a new module rather than a thin script around
agents.graph.ask(): ask() builds a FRESH graph on every single call --
new MultiServerMCPClient, new spawned mcp_server/server.py subprocess,
new empty `messages` list (see ask()'s own docstring: "Convenience
one-shot entry point"). That is exactly right for the CLI and for the
Phase 5 eval script, where every question is independent and process
spin-up cost is a non-issue run once. It is exactly wrong for a chatbot:
spawning a fresh MCP subprocess per HTTP request would make every turn
pay several seconds of startup cost, AND every turn would start with
amnesia -- `invoice` (specialists.py) reads PAST `product_search`
messages out of state["messages"] to know what to invoice, which is
silently unusable across separate ask() calls no matter how the caller
phrases the request, because state["messages"] is always empty except
for the one HumanMessage ask() itself just constructed.

This module fixes both problems the same way graph.py's own docstring
said a persistent caller should: build ONE graph, ONCE, at process
startup (one MCP client, one live server subprocess, shared by every
specialist AND every HTTP request this process ever serves -- same
sharing build_specialists() already does within a single run, just
extended to the process's whole lifetime instead of one call's), and
compile it with a real checkpointer (LangGraph's AsyncSqliteSaver) so
that invoking it repeatedly with the same `thread_id` accumulates
`messages` across turns instead of starting over each time.

What does NOT persist across turns, deliberately, even with the
checkpointer wired in: `route`, `iteration_count`, `blocked`, and
`injection_patterns`. These four have no reducer in state.py (only
`messages` does), which means LangGraph's default merge behavior would
otherwise carry each one's LAST CHECKPOINTED VALUE into the next turn
untouched -- confirmed by hand against a minimal synthetic StateGraph
with the same shape before wiring this up for real: a field with no
reducer that a node updates via plain dict return (state.get(key, 0) + 1,
same pattern supervisor.py uses for iteration_count) keeps climbing
across separate ainvoke() calls on the same thread_id unless the caller
explicitly re-supplies it as part of that call's input. That is never
what a NEW turn wants here: a fresh turn needs its own iteration cap
(supervisor.py's cap counts visits *within one turn's routing loop*, not
across a whole conversation's history of turns), its own guardrail
check (a clean message on turn 3 must not inherit turn 1's `blocked`),
and its own routing decision. So every call to /chat below explicitly
resets all four to their turn-zero values in the SAME dict that carries
the new HumanMessage -- see `_new_turn_state` -- exactly mirroring what
graph.py's own ask() already does for its one-shot case, just repeated
per turn instead of per process.

Endpoints:
    POST   /chat                       -- send a message, get an answer
    POST   /chat/{thread_id}/retry     -- regenerate an answer, replacing the old one
    POST   /chat/{thread_id}/edit      -- edit a past prompt, branching a new thread from it
    POST   /chat/{thread_id}/upload    -- attach an image/PDF, ingested into this thread's own
                                           personal RAG (see local_rag/personal_rag.py)
    GET    /chat/{thread_id}/history   -- read back a thread's messages
    DELETE /chat/{thread_id}           -- forget a thread (and its personal-RAG uploads)
    POST   /chat/{thread_id}/branch    -- copy a thread's history onto a new, independent thread_id
    GET    /chats                      -- list past conversations (browsing)
    GET    /tools                      -- list specialists forced_route accepts
    GET    /health                     -- readiness probe
    GET    /                           -- the built-in browser chat UI

ChatRequest.tool (optional): when set to one of GET /tools's names, that
ONE specialist answers the turn directly and the supervisor's own
routing is skipped entirely for it -- see graph.py's module docstring
and agents/state.py's `forced_route` field for exactly what this does
and doesn't change. Omitted (the default, and what every existing caller
that predates this field still gets) means the normal, unmodified
behavior: the supervisor picks among every specialist, same as always.
This is an isolation/debugging knob (e.g. "does image_qa alone handle
this correctly"), not a way to skip guardrails or the contextualize
rewrite -- input_guard, contextualize, and output_guard all still run
exactly as they do on a normal turn; only the supervisor's own
multi-specialist loop is bypassed.

Run with:
    python -m agents.api
    # or: uvicorn agents.api:app --reload --port 8001
Then open http://localhost:8001/ in a browser.

Known, stated limitations (worth listing plainly rather than glossing
over, same spirit as this project's other module docstrings):
  - Persistence is a single local SQLite file (see CHECKPOINT_DB_PATH).
    Fine for one person's local chatbot use; not multi-process-safe, not
    a substitute for a real database if this ever needed to serve
    multiple concurrent server processes.
  - Most specialists still only read the LATEST HumanMessage as their
    question (specialists.py's `_last_human_text` / equivalent) -- the
    checkpointer makes the FULL conversation visible in state["messages"]
    and lets `invoice` specifically read back past `product_search`
    results, but it does not, by itself, give retrieval_qa/corpus_meta/
    multi_hop/etc. the ability to resolve "what about that painting I
    asked about earlier" -- that would be a further change to how each
    specialist builds its question, not something this API layer can add
    from outside.
  - No streaming. /chat blocks until the supervisor loop reaches FINISH
    (or the iteration cap fires a partial answer), the same way ask()
    always has.
"""

import asyncio
import os
import re
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from agents.graph import build_graph
from agents.prompts import SPECIALIST_PUBLIC_DESCRIPTIONS
from agents.state import AgentState
import agents.specialists as specialists
from agents.supervisor import (
    DEFAULT_FALLBACK_ROUTE,
    DEFAULT_ITERATION_CAP,
    DEFAULT_ROUTE_FORMAT,
)
from agents.tracing import new_request_id


def _find_pipeline_root() -> Path:
    """
    Locate the directory that actually contains config.py (and therefore
    personal_rag.py) -- the same duplicated-per-module helper
    agents/specialists.py and agents/guardrails.py each already carry
    their own copy of, for the same reason both give: this file has no
    other dependency on local_rag/'s internals (everything else it needs
    comes through agents.graph/agents.state), and shouldn't gain an
    import of specialists.py or guardrails.py just to reuse eight lines
    of path-checking. Relied on below ONLY for personal_rag -- ingesting
    an uploaded file and deleting a thread's personal-RAG data are both
    plain, deterministic pipeline calls this HTTP layer makes directly
    (see POST /chat/{thread_id}/upload and DELETE /chat/{thread_id}
    below), the same "HTTP endpoint calls straight into local_rag/"
    pattern local_rag/api.py's own /ingest endpoint already uses -- never
    through the MCP server, which is reserved for calls a specialist's
    LLM makes mid-conversation (see personal_rag.py's own module
    docstring for that split).
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
        + "\nEdit _find_pipeline_root() in agents/api.py to add your actual path."
    )


_PIPELINE_ROOT = _find_pipeline_root()
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from config import RAW_DOCS_DIR  # noqa: E402
import personal_rag  # noqa: E402
import usage_tracker  # noqa: E402

# All three are overridable by environment variable so a grader / a
# second machine can point this at a different DB file or a different
# cap without editing source -- same ".env for anything you'd rather not
# hardcode" spirit the project README already asks for re: API keys,
# even though nothing here is a secret.
CHECKPOINT_DB_PATH = os.environ.get(
    "AGENT_API_DB_PATH", str(Path(__file__).resolve().parent / "chat_history.sqlite3")
)
ITERATION_CAP = int(os.environ.get("AGENT_API_ITERATION_CAP", DEFAULT_ITERATION_CAP))
ROUTE_FORMAT: Literal["json_schema", "json"] = os.environ.get(
    "AGENT_API_ROUTE_FORMAT", DEFAULT_ROUTE_FORMAT
)  # type: ignore[assignment]

# How long ONE turn (_invoke_turn below) is allowed to run before this
# HTTP layer gives up on it and returns a clean 503 -- the course's own
# Part 10 "API Best Practice" guidance ("Timeouts on everything...
# Return a partial answer or a clean error, never hang") applied to the
# one call site that can genuinely run long: a Groq fallback to a local
# CPU-bound Ollama model, several specialists deep into a re-route loop,
# is slower than this project's usual demo-on-a-good-GPU case. A single
# turn can chain multiple local LLM calls back to back -- supervisor
# routing, the specialist's own react-agent reasoning loop, and a
# generate_answer call -- each of which can itself take well over a
# minute on slow CPU-only hardware, so this needs real headroom rather
# than a number tuned for GPU inference. Overridable via env var for
# anyone who wants it tighter/looser than this default.
#
# RAISED from 600 to 1200 after a confirmed live-run timeout, not a
# hypothetical one: a turn genuinely hit this ceiling and got cut off
# with a 503 while the underlying work was still legitimately in
# progress, not stuck -- a Groq 429 retry wait, THEN a large-tier
# fallback to local Ollama (llama3.2), THEN a repeat-route override
# trying a SECOND specialist (also falling back to local), THEN a
# small-tier fallback to phi3 for the next supervisor call -- four-plus
# sequential CPU-bound local generations chained into one turn, each
# individually reasonable but adding up past the old ceiling well before
# any of them was actually stuck. The specific incident that triggered
# it (see local_rag/config.py's own comment) was Groq's two model IDs
# being fully decommissioned, forcing EVERY call in the turn onto the
# slow path instead of just occasional overflow -- fixing that removes
# the worst-case FREQUENCY of this chain, but not the chain's own
# worst-case DURATION when local fallback genuinely is needed (an
# occasional real rate limit, a network hiccup), which is what this
# ceiling actually has to budget for. 1200s is roughly double the old
# ceiling -- generous headroom for that chained-fallback case without
# going unbounded; a turn that's still not done after 20 minutes really
# is worth cutting off with a clean error rather than leaving the
# person waiting indefinitely with no feedback.
TURN_TIMEOUT_SECONDS = float(os.environ.get("AGENT_API_TURN_TIMEOUT_SECONDS", "1200"))

# Generous, single-developer-testing rate limit for this server's OWN
# HTTP endpoints -- a DIFFERENT concern from Groq's own free-tier rate
# limits (local_rag/usage_tracker.py / GET /v1/usage, below), which this
# number has no effect on. "Generous" is deliberate: this process is
# meant for one person's own local testing, so the point of this limiter
# is the STRUCTURAL habit (per-user limits are what stop one buggy
# client from exhausting a shared budget -- course Part 10/11) rather
# than actually constraining real usage today. Override via env var if a
# grader's test script needs a tighter or looser number. slowapi's own
# limit-string format: "<count>/<period>", e.g. "120/minute".
AGENT_API_RATE_LIMIT = os.environ.get("AGENT_API_RATE_LIMIT", "120/minute")

# Hard boundary on how large a single message the HTTP layer will even
# accept, independent of guardrails.py's own, much-more-nuanced
# `_MAX_INPUT_CHARS` (6000) check INSIDE the graph. The two are
# deliberately not the same value or the same mechanism: guardrails.py's
# check is a SEMANTIC guard -- it still runs the turn through input_guard
# and returns a normal 200 response explaining, in-conversation, why the
# turn was refused (so the person sees a real chat message, and
# `blocked` is set for the caller to key off of). This one is a
# STRUCTURAL boundary sitting in front of that: a message so large it
# couldn't possibly be a legitimate chat turn (a pasted book, a script
# gone wrong) is rejected immediately, before spending a checkpointer
# write or a graph invocation on it at all, with a 422 and a plain-text
# reason (see the RequestValidationError handler below for why that
# reason is a clean string, not FastAPI's default list-of-dicts shape).
_MAX_MESSAGE_CHARS = int(os.environ.get("AGENT_API_MAX_MESSAGE_CHARS", 12000))

# Loose enough to admit any UUID4 (what this file itself generates for a
# fresh thread) plus any other reasonable client-supplied identifier,
# strict enough to reject something that isn't actually an id at all
# (whitespace, a stray path separator, a multi-kilobyte paste into the
# wrong field) before it ever reaches the checkpointer as a SQLite key.
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_CHAT_HTML_PATH = _STATIC_DIR / "chat.html"

# Same recursion_limit margin ask() in graph.py uses, and for the same
# reason: generous enough that a legitimately high ITERATION_CAP never
# trips LangGraph's own unrelated recursion guard before supervisor.py's
# own cap logic gets a chance to fire first.
_RECURSION_LIMIT = max(25, ITERATION_CAP * 4)

# ---------------------------------------------------------------------
# Internal markup stripped from every message before it ever reaches an
# HTTP caller
# ---------------------------------------------------------------------
# Two things end up INSIDE a stored AIMessage's `content` that are meant
# for OTHER CODE to read back out of the transcript, not for a person to
# see in a chat bubble:
#
#   1. `<!--PRODUCT_DATA:[...]-->` -- specialists.py's product_search_node
#      embeds this HTML-comment footer so invoice_node (specialists.py's
#      own _parse_product_data / _collect_product_catalog) can later
#      reconstruct structured item data from a plain-text transcript. It
#      is a data channel between two specialists, not part of the answer.
#   2. `[All specialists already tried this turn (...) ...]` / `[Partial
#      answer -- iteration cap (N) ...]` -- supervisor.py's own forced-
#      FINISH notes (_all_tried_note / _partial_answer_note), prepended
#      ahead of a reaffirmed specialist's real answer so the checkpointed
#      transcript stays an honest, append-only record of what the
#      supervisor actually did (see supervisor.py's own docstring on why
#      that record is kept, deliberately, rather than silently cleaned
#      up at the source).
#
# Both are exactly right to keep IN THE GRAPH STATE (that's what makes
# invoice_node work at all, and what makes the transcript debuggable/
# eval-able) and exactly wrong to show a person chatting -- neither is
# ever this project's OWN "no" answer to something asked; both are
# bookkeeping. So the stripping happens HERE, once, at the HTTP boundary,
# on the way OUT to a caller -- never touching what's actually persisted
# by the checkpointer, so a second /chat call against the same thread_id
# still sees the real, un-stripped content invoice_node and the
# supervisor both depend on.
_PRODUCT_DATA_FOOTER_RE = re.compile(r"\n*<!--PRODUCT_DATA:\[.*?\]-->\s*$", re.DOTALL)

# Matches ONLY the two known forced-FINISH note prefixes supervisor.py
# itself ever generates (by their literal opening words), immediately
# followed by the "\n\n" separator _finalize_with_first_attempt always
# inserts between the note and the reaffirmed specialist's real content
# -- see that function's docstring. Deliberately narrow (not "any
# leading [...] text") so this can never eat a specialist's own
# legitimate bracketed content (e.g. a markdown link's `[text]`) that
# happens to open a message. The one supervisor note that has NO real
# content after it (iteration cap hit before any specialist ever ran) has
# no trailing "\n\n" to match against, so it's correctly left alone --
# stripping it would leave the person with an empty message instead of
# the only explanation they'd otherwise get for why there's no answer.
_SUPERVISOR_FORCED_FINISH_NOTE_RE = re.compile(
    r"^\[(?:All specialists already tried this turn|Partial answer -- iteration cap)\b.*?\]\n\n",
    re.DOTALL,
)


def _coerce_message_content_to_text(content) -> str:
    """
    Normalize a LangChain message's .content to plain text before any
    further string processing. Every node in this graph is INTENDED to
    only ever produce plain string content (state.py's own docstring
    says so, and _format_transcript's docstring notes non-string
    content "shouldn't occur"), but a confirmed live crash shows that
    guarantee doesn't hold at every edge: retrieval_qa's
    create_react_agent wrapper produced a final AIMessage whose .content
    was a LIST of content blocks (e.g. [{"type": "text", "text": "..."}])
    rather than a plain string -- observed specifically on an
    Arabic-language question, which crashed this exact function outright
    (`TypeError: expected string or bytes-like object, got 'list'`) the
    moment its regex substitution ran, before this normalization
    existed. guardrails.py's output_guard_node already guards the same
    class of message with `isinstance(msg.content, str)` and skips
    non-string content rather than crashing -- but skipping there just
    means the list-shaped content passes through UNCHANGED into this
    function, which had no equivalent guard of its own.

    Concatenates every string/"text"-shaped part it can find, in order;
    an unrecognized part is skipped rather than raised on -- the same
    "degrade, don't crash" contract every other content-handling helper
    in this project already follows.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _strip_internal_markup(content) -> str:
    """
    The person-facing version of a stored message's content -- see the
    module-level comment above this function for exactly what gets
    removed and why. Applied at every point this file hands message
    content to an HTTP caller (POST /chat's `answer` and `turn_messages`,
    GET /chat/{thread_id}/history's `content`) so every consumer of this
    API -- this repo's own React frontend, agents/static/chat.html, curl,
    anything else -- sees the same clean text, rather than each caller
    needing to know to strip this itself.
    """
    text = _coerce_message_content_to_text(content)
    stripped = _SUPERVISOR_FORCED_FINISH_NOTE_RE.sub("", text, count=1)
    stripped = _PRODUCT_DATA_FOOTER_RE.sub("", stripped)
    return stripped.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the checkpointer and the graph exactly once, when the server
    process starts -- not per request. `app.state.graph` and
    `app.state.checkpointer` are what every endpoint below reuses.

    AsyncSqliteSaver.from_conn_string() is itself an async context
    manager (opens one aiosqlite connection, closes it on exit), so this
    lifespan function's own `async with` block is what keeps that
    connection alive for exactly the server process's lifetime -- the
    same reason build_specialists() elsewhere in this project is called
    once per run rather than once per tool call.
    """
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        await checkpointer.setup()  # creates the checkpoint tables on first run; no-op after
        graph = await build_graph(
            iteration_cap=ITERATION_CAP,
            fallback_route=DEFAULT_FALLBACK_ROUTE,
            route_format=ROUTE_FORMAT,
            checkpointer=checkpointer,
        )
        app.state.graph = graph
        app.state.checkpointer = checkpointer
        print(
            f"[agents.api] graph ready -- iteration_cap={ITERATION_CAP}, "
            f"route_format={ROUTE_FORMAT!r}, checkpoint_db={CHECKPOINT_DB_PATH!r}"
        )
        yield
        # AsyncSqliteSaver's own __aexit__ closes the connection here.


app = FastAPI(title="Multi-Agent RAG Chat API", lifespan=lifespan)

# Server-side rate limiting (checklist item: "Rate limiting and
# timeouts... no slowapi, no asyncio.wait_for, nothing enforcing a time
# or request cap") -- generous, single-developer-testing default (see
# AGENT_API_RATE_LIMIT's own comment above for why "generous" is
# deliberate here, not an oversight). key_func=get_remote_address means
# the limit is per-CALLING-IP, so this laptop's own frontend/chat.html
# traffic is the only thing it ever throttles in practice; a genuinely
# multi-user deployment would want a per-API-key or per-thread_id key
# instead, but that's not this project's current shape (see this
# module's own docstring: "Fine for one person's local chatbot use").
# Applied via SlowAPIMiddleware + default_limits, below, rather than a
# per-route @limiter.limit(...) decorator on each endpoint -- covers
# every route (including ones added later) without every endpoint
# signature needing its own `request: Request` parameter threaded in
# just for this.
limiter = Limiter(key_func=get_remote_address, default_limits=[AGENT_API_RATE_LIMIT])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    slowapi's own default handler returns `{"error": "..."}` -- a
    different shape than every other error response in this file (see
    `_validation_error_handler`, just below, and its own docstring on
    why `detail` specifically is the one key frontend/src/api.ts's
    `handle()` reads). This collapses slowapi's response into that same
    `{"detail": "..."}` shape so a 429 renders in the chat UI exactly
    like any other error bubble, rather than silently stringifying to
    something unhelpful.

    Deliberately a SYNC function (`def`, not `async def`): SlowAPIMiddleware's
    own default-limit enforcement path (sync_check_limits, in
    slowapi/middleware.py) runs outside an event loop and falls back to
    slowapi's OWN default handler -- silently discarding a custom one --
    if it's a coroutine function. A plain sync function is what actually
    gets invoked here; confirmed by reading slowapi's own
    sync_check_limits source rather than assumed.
    """
    response = JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded ({exc.detail}). Please slow down and try again shortly."},
    )
    return limiter._inject_headers(response, request.state.view_rate_limit)


app.add_middleware(SlowAPIMiddleware)

# Needed once the UI is a separately-served app (frontend/, the
# assistant-ui React frontend, run via `npm run dev` on Vite's own port)
# rather than the single-file agents/static/chat.html this same process
# already serves same-origin at "/". Origins are overridable via env var
# for anyone deploying the frontend somewhere other than localhost.
# Comma-separated, no trailing slashes.
_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "AGENT_API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    FastAPI's own default 422 body is `{"detail": [{"loc": [...], "msg":
    ..., "type": ...}, ...]}` -- accurate, but a LIST, not the plain
    string every other error response in this file returns as `detail`
    (see e.g. POST /chat's own 400 for an unknown `tool`, a few hundred
    lines below). frontend/src/api.ts's `handle()` reads `body.detail`
    straight through as the error message; handed a list, `new
    Error(list)` stringifies it as something like
    "[object Object],[object Object]" -- technically not a crash, but
    not a message a person asked to fix their input could act on either.

    This collapses that same pydantic error list into one readable
    string under the SAME "detail" key, so every caller of this API
    (this repo's own frontend, agents/static/chat.html, curl, anything
    else) sees one consistent shape for "your request was rejected,
    here's why" everywhere, instead of one shape for validation errors
    and a different one for every other 4xx this file raises via
    HTTPException.
    """
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = err.get("msg", "Invalid input")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(parts) or "Invalid request."},
    )


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_MESSAGE_CHARS,
        description="The user's message.",
    )
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Conversation to continue. Omit on the first message of a new "
            "conversation -- the server generates one and returns it."
        ),
    )
    tool: Optional[str] = Field(
        default=None,
        description=(
            "Force this ONE specialist to answer the turn, bypassing the "
            "supervisor's own routing entirely (see GET /tools for valid "
            "names). Omit for the default: the supervisor picks among "
            "every specialist, exactly as it always has."
        ),
    )

    @field_validator("message")
    @classmethod
    def _message_is_not_just_whitespace(cls, v: str) -> str:
        """
        `min_length=1` alone accepts a single space, a tab, or a string
        of newlines -- non-empty by character count, empty by anything
        that actually matters to the graph (input_guard, every
        specialist's own `_last_human_text`, etc. all operate on this
        same string). Rejected here, at the boundary, with a message a
        person typing into a chat box can immediately understand, rather
        than silently becoming a HumanMessage with blank content that
        every downstream specialist then has to independently guard
        against.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message can't be empty or contain only whitespace.")
        return stripped

    @field_validator("thread_id")
    @classmethod
    def _thread_id_looks_like_an_id(cls, v: Optional[str]) -> Optional[str]:
        """
        A `thread_id` reaches the checkpointer directly as a SQLite
        lookup key (see every `{"configurable": {"thread_id": ...}}`
        config below) -- this doesn't need to defend against SQL
        injection (the checkpointer parameterizes its own queries), but
        a stray multi-kilobyte paste, embedded whitespace, or a path-like
        string in this field is never a legitimate thread_id (every real
        one this server itself ever hands out is a plain uuid4 hex
        string -- see POST /chat's `thread_id = req.thread_id or
        str(uuid4())`), so it's rejected here with a clear reason instead
        of quietly being accepted and creating a thread nothing else can
        ever meaningfully reference.
        """
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not _THREAD_ID_RE.match(v):
            raise ValueError(
                "thread_id may only contain letters, digits, hyphens, and "
                "underscores (max 128 characters)."
            )
        return v


class RetryRequest(BaseModel):
    message_id: Optional[str] = Field(
        default=None,
        description=(
            "The id (TurnMessage.id / HistoryMessage.id / ChatResponse."
            "human_message_id) of the HumanMessage to regenerate an answer "
            "for. Omit to retry the most recent turn -- the common case, "
            "and what a plain 'retry'/'regenerate' button should send. "
            "Retrying anything other than the LAST turn discards every "
            "message after the targeted one (this thread has no in-thread "
            "branching -- see POST /chat/{thread_id}/edit if you want the "
            "old continuation to stay reachable)."
        ),
    )
    tool: Optional[str] = Field(
        default=None,
        description="Same as ChatRequest.tool -- force one specialist to "
        "answer the regenerated turn. Omit for the default supervisor-"
        "routed behavior.",
    )

    @field_validator("message_id")
    @classmethod
    def _message_id_looks_like_an_id(cls, v: Optional[str]) -> Optional[str]:
        # LangGraph message ids are uuid4 strings (add_messages assigns
        # one automatically to any message that doesn't already have
        # one -- see state.py's own docstring) -- same shape check
        # ChatRequest.thread_id already applies to itself, reused here
        # for the same reason: reject a stray paste or an obviously-not-
        # an-id string before it ever reaches a linear scan over the
        # thread's messages.
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not _THREAD_ID_RE.match(v):
            raise ValueError("message_id doesn't look like a real message id.")
        return v


class EditRequest(BaseModel):
    message_id: str = Field(
        description="The id of the HumanMessage to edit (see RetryRequest."
        "message_id's docstring for where this id comes from)."
    )
    content: str = Field(
        min_length=1,
        max_length=_MAX_MESSAGE_CHARS,
        description="The edited message text.",
    )
    tool: Optional[str] = Field(default=None, description="Same as ChatRequest.tool.")

    @field_validator("content")
    @classmethod
    def _content_is_not_just_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Edited message can't be empty or contain only whitespace.")
        return stripped

    @field_validator("message_id")
    @classmethod
    def _message_id_looks_like_an_id(cls, v: str) -> str:
        if not _THREAD_ID_RE.match(v.strip()):
            raise ValueError("message_id doesn't look like a real message id.")
        return v.strip()


class TurnMessage(BaseModel):
    """One message produced during a single turn -- exposed so a caller
    can see the routing path (e.g. which specialist(s) ran, and any
    supervisor meta-note), not just the final answer text."""

    id: str = Field(
        description="This message's own persisted id (state.py's add_messages "
        "reducer assigns one to every message on first write) -- pass THIS "
        "back as `message_id` to POST /chat/{thread_id}/retry or "
        "POST /chat/{thread_id}/edit to target it, rather than re-deriving an "
        "id client-side (e.g. runtime.ts no longer invents its own "
        "`local-...` ids for a freshly-sent turn -- it uses these real, "
        "checkpointer-persisted ones instead, so a retry/edit sent "
        "immediately after this response -- with no GET history round trip "
        "in between -- still targets a real message)."
    )
    name: Optional[str]
    content: str


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    answered_by: Optional[str] = Field(
        description="Name of the message that produced `answer`: a specialist "
        "name, 'input_guard' if the turn was refused, or 'supervisor' in the "
        "rare case no specialist ever ran before the cap fired."
    )
    blocked: bool
    iteration_count: int
    human_message_id: str = Field(
        description="The persisted id of the HumanMessage this turn answered -- "
        "pass this as `message_id` to POST /chat/{thread_id}/retry (regenerate "
        "this exact turn's answer) or POST /chat/{thread_id}/edit (branch a new "
        "thread from an edited version of this message)."
    )
    turn_messages: list[TurnMessage] = Field(
        description="Every message produced this turn, in order (for "
        "debugging/eval -- mirrors graph.py's own __main__ printing logic)."
    )


class HistoryMessage(BaseModel):
    id: str = Field(
        description="This message's persisted id -- see TurnMessage.id's own "
        "docstring; used the same way here to target a retry/edit against a "
        "message loaded from history rather than one just returned by /chat."
    )
    role: Literal["human", "ai"]
    name: Optional[str]
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class ChatSummary(BaseModel):
    """One row of GET /chats -- enough to render a browsable list of past
    conversations without pulling every message of every thread over the
    wire, same "summary now, full history on demand via the existing
    GET /chat/{thread_id}/history" split a normal chat UI's sidebar uses."""

    thread_id: str
    title: str = Field(description="First human message, truncated -- there's no separate "
        "user-set conversation title anywhere in this system, so the opening "
        "question doubles as one, same as most chat UIs default to.")
    updated_at: Optional[str] = Field(
        description="ISO timestamp of this thread's most recent checkpoint, "
        "straight from LangGraph's own StateSnapshot.created_at -- None only "
        "if the checkpointer itself didn't record one (shouldn't happen in "
        "practice, kept Optional rather than asserted so a browsing request "
        "can't 500 on a checkpoint format Claude hasn't seen)."
    )
    message_count: int


class ChatListResponse(BaseModel):
    chats: list[ChatSummary] = Field(
        description="Most recently updated first."
    )


class BranchResponse(BaseModel):
    """Response for POST /chat/{thread_id}/branch -- see that endpoint's
    own docstring for what "branch" means here."""

    thread_id: str = Field(description="The NEW thread_id -- start sending /chat requests here to continue the branch.")
    branched_from: str = Field(description="The thread_id this branch was copied from, unchanged by anything that happens on the branch.")
    message_count: int = Field(description="How many messages were copied onto the new thread.")


class EditResponse(BaseModel):
    """Response for POST /chat/{thread_id}/edit -- a BranchResponse (a new
    thread was created, same as POST /chat/{thread_id}/branch) plus a
    ChatResponse (that new thread's first generated answer, to the edited
    message), so a caller gets both "here's the new thread" and "here's
    what it said" in one round trip instead of needing to immediately
    follow up with a second POST /chat call."""

    thread_id: str = Field(description="The NEW thread_id -- the edit's branch.")
    branched_from: str = Field(description="The thread_id this branch was edited from.")
    edited_message_id: str = Field(description="The id of the message that was edited "
        "(its ORIGINAL id, from `branched_from` -- the edited copy on the new thread "
        "gets its own fresh id, returned as chat.human_message_id below).")
    chat: ChatResponse = Field(description="The new thread's own id (mirrors `thread_id` "
        "above), the freshly generated answer to the edited message, and that turn's "
        "own message ids.")


class ToolInfo(BaseModel):
    name: str
    description: str


class ToolListResponse(BaseModel):
    tools: list[ToolInfo] = Field(
        description="Valid values for ChatRequest.tool, in the same order "
        "the supervisor itself tries them as an untried-route fallback "
        "(specialists.py's own build order) -- not alphabetized, so a "
        "grader/UI reading this list top-to-bottom sees the same priority "
        "order the graph does."
    )


def _new_turn_state(
    message: str, thread_id: str, forced_route: Optional[str] = None
) -> AgentState:
    """
    The per-turn input to graph.ainvoke(): a new HumanMessage (which
    add_messages appends onto whatever the checkpointer already has for
    this thread_id) plus an explicit reset of the fields that have no
    reducer and would otherwise silently carry the previous turn's
    values forward -- see this module's own docstring for why that
    matters and how it was checked.

    `forced_route` defaults to None (normal supervisor-routed turn) and
    is deliberately NOT persisted forward by anything -- a `tool` set on
    turn 2 of a conversation must not keep silently overriding the
    supervisor on turn 3, which is exactly why this is reset here every
    turn rather than added to the four originally-reset fields as a
    fifth one without a reducer, same reasoning, same fix.

    `thread_id` never actually changes turn to turn (it's the same
    conversation), but is resupplied explicitly here anyway for the same
    reason -- state.py's `thread_id` field has no reducer either, and
    this module would rather be explicit about it every call than lean
    on LastValue's "unchanged if omitted" behavior implicitly carrying it
    forward. See state.py's own docstring for what this scopes
    (personal_docs_node's search of THIS thread's own uploads).
    """
    return {
        "messages": [HumanMessage(content=message)],
        "route": None,
        "iteration_count": 0,
        "blocked": False,
        "injection_patterns": [],
        "forced_route": forced_route,
        "thread_id": thread_id,
        "request_id": new_request_id(),
    }


def _retry_turn_state(thread_id: str, forced_route: Optional[str] = None) -> AgentState:
    """
    Sibling to `_new_turn_state` for POST /chat/{thread_id}/retry: the
    SAME per-turn reset (route/iteration_count/blocked/injection_patterns/
    forced_route/thread_id all back to their turn-zero values), but no new
    HumanMessage -- the message being retried is already the last one in
    the (already-trimmed, see retry_message() below) persisted history, so
    contextualize.py's own `_split_last_human` / every specialist's
    `_last_human_text` pick it up exactly as if it had just arrived,
    without this needing to duplicate it in as a second, freshly-`id`'d
    message right next to the original.
    """
    return {
        "messages": [],
        "route": None,
        "iteration_count": 0,
        "blocked": False,
        "injection_patterns": [],
        "forced_route": forced_route,
        "thread_id": thread_id,
        "request_id": new_request_id(),
    }


def _find_message_by_id(messages: list[BaseMessage], message_id: str) -> Optional[int]:
    """Index of the message with this id, or None -- used by retry/edit to
    locate the target message a caller referenced by TurnMessage.id /
    HistoryMessage.id (see those models' own docstrings), rather than
    always assuming "the last one."""
    for i, m in enumerate(messages):
        if getattr(m, "id", None) == message_id:
            return i
    return None


def _find_last_human_message(messages: list[BaseMessage]) -> Optional[HumanMessage]:
    """The most recent HumanMessage in a message list, or None -- mirrors
    guardrails.py's own `_last_human_message` (kept as a local copy for
    the same "no cross-module dependency for a few lines" reasoning that
    module already gives), used here after a turn completes to report
    ChatResponse.human_message_id back to the caller."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None


def _messages_since_last_human(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Every message produced during the turn that just ran: everything
    after the LAST HumanMessage in the full (now checkpointer-persisted,
    possibly multi-turn) list. Deliberately a local copy of the same
    scoping guardrails.py's `_messages_since_last_human` and
    supervisor.py's `_current_turn_context` already use, rather than an
    import of either -- both are underscore-private to their own module,
    and this is the same "no cross-module dependency for eight lines"
    call guardrails.py itself makes about _find_pipeline_root().
    """
    last_human_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    if last_human_idx is None:
        return list(messages)
    return messages[last_human_idx + 1 :]


async def _invoke_turn(state: AgentState, config: dict) -> dict:
    """
    Shared graph.ainvoke() wrapper for every endpoint that runs a turn
    (POST /chat, POST /chat/{thread_id}/retry, POST /chat/{thread_id}/edit)
    -- factored out so the 503-on-infra-failure translation below (Ollama
    down / the MCP server subprocess died) AND the timeout below are
    written, and can be fixed, in exactly one place rather than three
    copies quietly drifting apart.
    """
    try:
        return await asyncio.wait_for(
            app.state.graph.ainvoke(state, config=config), timeout=TURN_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        # A turn that's still running after TURN_TIMEOUT_SECONDS almost
        # always means something downstream is hung rather than merely
        # slow (Ollama loading a model cold, a stuck subprocess, a Groq
        # request that neither completed nor errored) -- the course's own
        # Part 10 guidance ("Timeouts on everything... Return a partial
        # answer or a clean error, never hang") applied here. Same 503 +
        # generic-detail treatment as the infra-failure branch below, for
        # the same reason: the real timeout value is server config, not
        # something a chat message should expose.
        print(
            f"[agents.api] /chat turn TIMED OUT after {TURN_TIMEOUT_SECONDS}s for "
            f"thread_id={state.get('thread_id')!r}",
            file=sys.stderr,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "This message is taking longer than expected to answer and was "
                "stopped. Please try again in a moment; if it keeps happening, "
                "let the person running this server know."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 -- see note below
        # The graph itself is designed never to raise for a normal bad
        # input (that's the whole point of supervisor.py's four safety
        # nets and the iteration cap) -- an exception surfacing here
        # almost always means an infrastructure problem outside the
        # graph's own control: Ollama not running, or the mcp_server/
        # server.py subprocess this process's shared MCP client talks to
        # having died. Surfaced as 503, not 500, to point at that.
        #
        # The full exception -- exactly what failed, in which module,
        # with what repr -- is deliberately printed to stderr (this
        # process's own command window) and NEVER put into the
        # HTTPException `detail` below. `detail` is what the caller's
        # HTTP client sees and what the frontend renders straight into
        # the chat as an error bubble; a Python traceback / exception
        # repr there is an internal implementation detail (module paths,
        # variable names, sometimes a fragment of a prompt or a stack
        # frame) that has no business being user-facing. A person using
        # the chat should see "this failed, try again" -- a developer
        # debugging it should see the real `exc!r` in the terminal that's
        # already running this server, which is exactly where this print
        # goes and exactly who is looking at that terminal.
        print(f"[agents.api] /chat turn failed for thread_id="
              f"{state.get('thread_id')!r}: {exc!r}", file=sys.stderr)
        raise HTTPException(
            status_code=503,
            detail=(
                "This message couldn't be answered right now due to a "
                "server-side issue. Please try again in a moment; if it "
                "keeps happening, let the person running this server know."
            ),
        ) from exc


def _chat_response_from_result(thread_id: str, result: dict) -> ChatResponse:
    """
    Shared response-shaping for every endpoint that runs a turn and hands
    back a ChatResponse-shaped result (POST /chat itself, plus retry/edit
    below, whose own response models embed one of these) -- see
    `_invoke_turn`'s own docstring for the same "one place, not three"
    reasoning.
    """
    last_message = result["messages"][-1]
    turn_messages = [
        TurnMessage(id=m.id, name=getattr(m, "name", None), content=_strip_internal_markup(m.content))
        for m in _messages_since_last_human(result["messages"])
        if isinstance(m, AIMessage)
    ]
    human_message = _find_last_human_message(result["messages"])

    return ChatResponse(
        thread_id=thread_id,
        answer=_strip_internal_markup(last_message.content),
        answered_by=getattr(last_message, "name", None),
        blocked=bool(result.get("blocked", False)),
        iteration_count=result.get("iteration_count", 0),
        human_message_id=human_message.id if human_message is not None else "",
        turn_messages=turn_messages,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Send one message and get one answer back. If `thread_id` is omitted,
    a new conversation is started and its id is returned in the response
    -- send that same id back on the next call to continue it.
    """
    if req.tool is not None and req.tool not in app.state.graph.known_specialist_names:
        # Rejected with a 400, not silently downgraded to the normal
        # supervisor-routed behavior the way an unrecognized forced_route
        # degrades INSIDE the graph (see graph.py's _resolve_forced_route
        # docstring for why THAT boundary prefers silent degradation over
        # raising) -- the two boundaries want different things on purpose.
        # Deep inside the graph, forced_route sits alongside `route` and
        # `iteration_count`, values a model itself can get wrong, so the
        # graph is built to degrade gracefully no matter what lands in
        # that field. Here at the API boundary, `tool` came from a human
        # or a UI dropdown typing/sending an exact string on purpose --
        # a typo'd tool name is far more likely to be "meant something
        # else" than "safe to silently reinterpret as the supervisor's
        # own choice," so this fails loudly, with the exact valid names,
        # rather than quietly running the default behavior instead of
        # the one the caller actually asked for.
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown tool {req.tool!r}. Valid values: "
                f"{sorted(app.state.graph.known_specialist_names)}. Omit `tool` "
                "for the default supervisor-routed behavior."
            ),
        )

    thread_id = req.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": _RECURSION_LIMIT}

    # See this function's own docstring for why this runs BEFORE the
    # graph turn -- it's a no-op (one cheap aget_state read, nothing
    # else) for the overwhelming majority of turns that carry no
    # `<attachment ...>` marker at all.
    await _retract_upload_preview_for_this_turn(thread_id, config, req.message)

    # result["messages"][-1] is reliably the user-facing answer for both
    # the refused path (input_guard's own AIMessage, name="input_guard")
    # and the normal path (a specialist's AIMessage, or the rare bare
    # "supervisor" fallback) -- see graph.py's own module docstring and
    # supervisor.py's _finalize_with_first_attempt docstring for why the
    # LAST message is always the one worth showing, never earlier ones.
    result = await _invoke_turn(_new_turn_state(req.message, thread_id, forced_route=req.tool), config)
    return _chat_response_from_result(thread_id, result)


@app.post("/chat/{thread_id}/retry", response_model=ChatResponse)
async def retry_message(thread_id: str, req: RetryRequest) -> ChatResponse:
    """
    Regenerate an answer for a prompt that's already in this thread --
    "resend this message and get a new answer back instead of the old
    one" (the whole point of a retry/regenerate button: the person
    should never end up looking at BOTH the old and the new answer to
    the same question stacked in the transcript).

    Implementation: find the target HumanMessage (req.message_id, or the
    last one in the thread if omitted -- see RetryRequest.message_id's
    own docstring), delete every message the checkpointer has stored
    AFTER it (that turn's old answer, plus -- if an earlier-than-last
    message was targeted -- every later turn too, since those turns'
    own answers were generated against a conversation that included the
    now-regenerated one) via LangGraph's own documented message-deletion
    pattern (RemoveMessage + aupdate_state, which runs it through
    state.py's add_messages reducer exactly like a node's return would),
    then re-invoke the graph with NO new HumanMessage (see
    `_retry_turn_state`'s own docstring) -- the target message is still
    the last one in the (now-trimmed) persisted history, so
    contextualize/every specialist picks it up exactly as if it had just
    arrived, and the graph produces a genuinely fresh answer rather than
    replaying a cached one.

    404s if `thread_id` has no messages at all, or if `message_id` was
    given but doesn't match any message in this thread, or matches a
    message that isn't a HumanMessage (only a person's own turn can be
    retried -- there's no such thing as "retry the assistant's answer"
    independent of the question it was answering).
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": _RECURSION_LIMIT}
    snapshot = await app.state.graph.aget_state(config)
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation found for thread_id {thread_id!r} -- nothing to retry.",
        )

    if req.message_id is not None:
        target_idx = _find_message_by_id(messages, req.message_id)
        if target_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"No message with id {req.message_id!r} found in thread_id {thread_id!r}.",
            )
        if not isinstance(messages[target_idx], HumanMessage):
            raise HTTPException(
                status_code=400,
                detail=f"message_id {req.message_id!r} is not a user message -- only a "
                "user's own prompt can be retried.",
            )
    else:
        target_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                target_idx = i
                break
        if target_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"thread_id {thread_id!r} has no user message to retry.",
            )

    trailing = messages[target_idx + 1 :]
    removals = [RemoveMessage(id=m.id) for m in trailing if getattr(m, "id", None)]
    if removals:
        await app.state.graph.aupdate_state(config, {"messages": removals})

    result = await _invoke_turn(_retry_turn_state(thread_id, forced_route=req.tool), config)
    return _chat_response_from_result(thread_id, result)


@app.post("/chat/{thread_id}/edit", response_model=EditResponse)
async def edit_message(thread_id: str, req: EditRequest) -> EditResponse:
    """
    Edit an earlier prompt and branch a new conversation from it --
    "change what I asked, and continue from there" without losing the
    original thread it was edited out of. This is deliberately the same
    "copy history onto a brand-new independent thread_id" shape POST
    /chat/{thread_id}/branch already uses (see that endpoint's own
    docstring), just truncated at the edited message and with its
    content replaced, then immediately run one turn on the new thread
    so the caller gets a real answer back, not just an empty new thread
    to POST /chat against separately.

    Implementation: read the source thread's messages, find `message_id`
    (must be a HumanMessage -- only a person's own prompt can be
    edited), keep everything STRICTLY BEFORE it (the prior conversation,
    unedited -- exactly what contextualize.py's own `_split_last_human`
    calls "prior" when it resolves a follow-up), seed a brand-new
    thread_id with that prefix via the graph's own aupdate_state() (the
    same "manually write a state update, as if it came from a node"
    call branch_thread() below already relies on), then run one normal
    turn on the new thread with the EDITED content as a fresh
    HumanMessage (`_new_turn_state`, unchanged) -- exactly as if the
    person had opened a new chat, pasted the prior conversation in, and
    then asked the edited question.

    The two threads share no further state after this call, same
    guarantee POST /chat/{thread_id}/branch already gives: continuing
    the original thread never touches the edit's branch, and vice
    versa.

    404s if `thread_id` has no messages, or if `message_id` doesn't
    match any message in it. 400s if `message_id` matches a message
    that isn't a HumanMessage.
    """
    source_config = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.state.graph.aget_state(source_config)
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation found for thread_id {thread_id!r} -- nothing to edit.",
        )

    target_idx = _find_message_by_id(messages, req.message_id)
    if target_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"No message with id {req.message_id!r} found in thread_id {thread_id!r}.",
        )
    if not isinstance(messages[target_idx], HumanMessage):
        raise HTTPException(
            status_code=400,
            detail=f"message_id {req.message_id!r} is not a user message -- only a "
            "user's own prompt can be edited.",
        )

    prior = messages[:target_idx]
    new_thread_id = str(uuid4())
    new_config = {"configurable": {"thread_id": new_thread_id}, "recursion_limit": _RECURSION_LIMIT}
    if prior:
        await app.state.graph.aupdate_state(
            {"configurable": {"thread_id": new_thread_id}}, {"messages": prior}
        )

    result = await _invoke_turn(
        _new_turn_state(req.content, new_thread_id, forced_route=req.tool), new_config
    )
    chat_response = _chat_response_from_result(new_thread_id, result)

    return EditResponse(
        thread_id=new_thread_id,
        branched_from=thread_id,
        edited_message_id=req.message_id,
        chat=chat_response,
    )


@app.get("/chat/{thread_id}/history", response_model=HistoryResponse)
async def get_history(thread_id: str) -> HistoryResponse:
    """
    Read back everything the checkpointer has stored for this thread_id.
    An unknown or never-used thread_id is not an error -- it just has no
    state yet, so this returns an empty message list rather than 404ing;
    a chatbot UI can call this on load without first checking whether
    the thread_id it generated client-side has ever been used.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.state.graph.aget_state(config)
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []

    history = [
        HistoryMessage(
            id=m.id,
            role="human" if isinstance(m, HumanMessage) else "ai",
            name=getattr(m, "name", None),
            content=_strip_internal_markup(m.content),
        )
        for m in messages
        if isinstance(m.content, str)
    ]
    return HistoryResponse(thread_id=thread_id, messages=history)


@app.get("/chats", response_model=ChatListResponse)
async def list_chats(limit: int = 30) -> ChatListResponse:
    """
    Browse past conversations -- every thread_id the checkpointer has
    ever stored a checkpoint for, most recently updated first, each with
    a short preview instead of its full message list (see
    GET /chat/{thread_id}/history for that).

    There's no "list every thread_id" method on LangGraph's own
    BaseCheckpointSaver interface (`alist`/`aget_tuple` are both scoped
    to ONE thread_id already known to the caller) -- AsyncSqliteSaver
    only guarantees a `checkpoints` table with a `thread_id` column
    exists (see its own `setup()`), so this queries that table directly
    via the same `.conn` (a plain aiosqlite.Connection) the checkpointer
    itself uses internally, rather than inventing a second, separately-
    maintained index of "every thread_id this process has seen." Reading
    straight from the checkpointer's own source of truth means a thread
    created by an earlier server run (same CHECKPOINT_DB_PATH, process
    restarted since) still shows up here correctly.

    `checkpoint_id` sorts lexicographically in creation order (LangGraph
    generates it from a time-ordered UUID), so MAX(checkpoint_id) per
    thread_id -- then ORDER BY that same value DESC -- gives "most
    recently active thread first" without needing to decode any
    checkpoint BLOB just to sort. Threads with zero real messages yet
    (a row exists but the graph never actually got a HumanMessage
    appended -- shouldn't normally happen, kept as a guard rather than
    assumed away) are silently skipped rather than shown with an empty
    title.
    """
    cursor = await app.state.checkpointer.conn.execute(
        "SELECT thread_id, MAX(checkpoint_id) AS latest "
        "FROM checkpoints WHERE checkpoint_ns = '' "
        "GROUP BY thread_id ORDER BY latest DESC LIMIT ?",
        (max(1, min(limit, 200)),),
    )
    rows = await cursor.fetchall()
    await cursor.close()

    chats: list[ChatSummary] = []
    for thread_id, _latest_checkpoint_id in rows:
        snapshot = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
        messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
        first_human = next(
            (m for m in messages if isinstance(m, HumanMessage) and isinstance(m.content, str)),
            None,
        )
        if first_human is None:
            continue
        title = first_human.content[:60]
        if len(first_human.content) > 60:
            title += "\u2026"
        chats.append(
            ChatSummary(
                thread_id=thread_id,
                title=title,
                updated_at=getattr(snapshot, "created_at", None),
                message_count=len(messages),
            )
        )
    return ChatListResponse(chats=chats)


@app.get("/tools", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    """
    The valid values for ChatRequest.tool -- built from
    `app.state.graph.known_specialist_names` (the ACTUAL set this
    running graph was compiled with, see graph.py's build_graph docstring
    for why that's stashed there) rather than SPECIALIST_PUBLIC_DESCRIPTIONS'
    own keys directly, so a specialist that exists in prompts.py but
    wasn't actually wired into this graph (or vice versa) can't silently
    appear here as choosable when GET /chat would 400 it anyway.

    Deliberately uses SPECIALIST_PUBLIC_DESCRIPTIONS, NOT the internal
    SPECIALIST_DESCRIPTIONS dict that's actually baked into
    SUPERVISOR_SYSTEM_PROMPT -- this is an unauthenticated, public
    endpoint, and SPECIALIST_DESCRIPTIONS' own wording is internal
    routing-strategy prose (confirmed misrouting cases, trigger phrasing
    being tuned against a live model) that has no business being handed
    out to anyone who calls this endpoint. See
    SPECIALIST_PUBLIC_DESCRIPTIONS' own docstring in prompts.py.
    """
    return ToolListResponse(
        tools=[
            ToolInfo(name=name, description=SPECIALIST_PUBLIC_DESCRIPTIONS.get(name, "(no description available)"))
            for name in app.state.graph.known_specialist_names
        ]
    )


@app.delete("/chat/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    """
    Forget a thread's entire history. Uses the checkpointer's own
    adelete_thread rather than hand-rolled SQL against its tables --
    langgraph-checkpoint-sqlite owns that schema, not this file.

    Also deletes every chunk this thread ever uploaded into the personal
    RAG's shared "temp" Chroma collection (personal_rag.delete_thread_data
    -- see that module's own docstring) -- a deleted conversation's
    uploaded images/PDFs/text files shouldn't silently keep living in that
    collection forever, especially once this thread_id itself is gone
    and nothing could ever reference them again anyway. Best-effort: a
    failure here is logged but does NOT stop the checkpointer delete
    above from completing -- the conversation itself is still gone
    either way, which is what DELETE means; a personal-RAG cleanup
    hiccup shouldn't leave the thread's chat history stuck undeletable.
    """
    await app.state.checkpointer.adelete_thread(thread_id)
    try:
        # Same reasoning as upload_document's own asyncio.to_thread call
        # -- delete_thread_data() is synchronous (a chromadb delete_where
        # plus a shutil.rmtree), and this endpoint shouldn't block the
        # whole event loop for its duration either, even though it's
        # normally fast.
        await asyncio.to_thread(personal_rag.delete_thread_data, thread_id)
    except Exception as exc:  # noqa: BLE001 -- see docstring above
        print(f"[agents.api] personal_rag cleanup failed for thread_id={thread_id!r}: {exc!r}")
    return {"thread_id": thread_id, "deleted": True}


class UploadResponse(BaseModel):
    thread_id: str
    filename: str
    n_chunks: int = Field(description="How many searchable chunks this upload produced. "
        "0 is a valid, non-error outcome (e.g. a PDF with no extractable text) -- see "
        "personal_rag.ingest_upload's own docstring.")
    modality: Literal["pdf", "text", "image"]
    n_images_shown: int = Field(
        default=0,
        description="How many of this upload's images were captioned and posted "
        "directly into the chat transcript as a normal assistant turn (see "
        "_post_captioned_images_to_chat). Always 0 for a PDF upload, or for an "
        "image upload the VLM couldn't caption at all.",
    )


async def _post_captioned_images_to_chat(thread_id: str, captioned_images: list[dict]) -> int:
    """
    Turn personal_rag.ingest_upload()'s "captioned_images" (see that
    function's own docstring for the exact shape) into real messages,
    appended straight onto this thread's persisted history via the
    graph's own aupdate_state() -- the same "manually write a state
    update, as if it came from a node" call POST .../retry and POST
    .../edit already rely on (see those endpoints' own docstrings) to
    inject a message without running any graph node at all.

    This is the fix for "the image sent should be captioned and shown in
    the chat, not have it retrieve a picture that resembles it": the
    caption AND the actual uploaded image now land in the conversation
    the moment the upload finishes -- deterministically, with zero
    LLM/routing involvement -- instead of only becoming visible later,
    and only if the supervisor happens to route a follow-up question to
    a specialist that goes looking for it (which, per a confirmed live
    run, it doesn't always do -- see agents/specialists.py's
    image_qa_node / personal_docs_node for the structural fix on THAT
    side of the same problem).

    One HumanMessage ("📎 Uploaded ...") + one AIMessage (the
    rendered image + its caption, name="personal_upload") per captioned
    image, so the transcript reads like a normal exchange on replay /
    GET .../history -- a turn the person took, and a turn the assistant
    took in response -- rather than a one-sided assistant message with
    no visible prompt behind it. If a real question about the SAME
    upload arrives via POST /chat shortly after (the common case: attach
    + type a question + one Send), this pair is retracted again right
    before that turn runs -- see `_retract_upload_preview_for_this_turn`
    below for why a redundant preview isn't worth keeping once a real,
    combined answer is about to supersede it.

    Best-effort and silent on failure: the upload itself has ALREADY
    succeeded by the time this is called (ingest_upload already ran, the
    file is already searchable), so a checkpointer hiccup here is logged
    to stderr and swallowed rather than raised -- a person should never
    have to re-upload a file just because posting its confirmation
    message failed. Returns how many images were actually posted (0 if
    `captioned_images` was empty, e.g. a PDF upload with no images at
    all, or every image failing to caption).
    """
    if not captioned_images:
        return 0

    messages: list[BaseMessage] = []
    for img in captioned_images:
        # CONFIRMED live-run bug this closes: an unescaped caption
        # interpolated straight into `![caption](...)` can prematurely
        # close the markdown image's own `![...]` alt-text span the
        # moment the (free-form, VLM-generated) caption itself contains
        # an unescaped "]" -- e.g. a caption describing a diagram's own
        # numbered/bracketed labels. Once that happens, everything after
        # it -- the entire `(data:image/...;base64,...)` destination --
        # falls through as plain paragraph text instead of being parsed
        # as an image, spilling a multi-hundred-KB raw base64 string
        # into the chat as visible text. specialists.py's own
        # `_escape_markdown_caption` already exists specifically for
        # this failure mode (see that function's own docstring) and is
        # already applied everywhere else an image gets rendered this
        # way (`_personal_image_display_block`, image_qa_node, ...) --
        # this was the one remaining call site still missing it.
        caption = specialists._escape_markdown_caption(img.get("caption")) or "(no caption available)"
        data_uri = img.get("data_uri")
        filename = img.get("filename") or "the image"
        # A short, clearly-attachment-shaped note rather than a bracketed
        # "[uploaded image: ...]" string -- both read to a person as
        # something they never actually said, but the bracket form in
        # particular was a confirmed source of confusion (a live report:
        # it looks like a raw internal marker leaking into the transcript,
        # not like a normal chat message). Still a HumanMessage (see this
        # function's own docstring for why -- the transcript should read
        # like a real exchange, a turn the person took and one the
        # assistant took in response), just phrased as a person would
        # actually narrate an attach action.
        messages.append(HumanMessage(content=f"📎 Uploaded {filename}"))
        if data_uri:
            # Image + alt-text caption ONLY -- no caption repeated again
            # as its own paragraph below. Mirrors agents/specialists.py's
            # _personal_image_display_block, which exists specifically
            # because a confirmed live report showed the doubled-up
            # version (this function's own previous behavior: the image,
            # then the same caption AGAIN as a separate italic line)
            # reads as noisy/duplicated -- see that function's own
            # docstring for the identical reasoning, now applied here too
            # so the upload-confirmation turn and the follow-up-question
            # turn render consistently instead of one of them still
            # showing the caption twice.
            body = f"![{caption}]({data_uri})"
        else:
            # Captioned successfully, but the bytes couldn't be
            # persisted/re-read/embedded afterward (see personal_rag.py's
            # _persist_personal_image / _image_to_data_uri -- both
            # degrade to None rather than raising). Show the caption
            # plainly rather than a broken image reference; the person
            # still gets "captioned", just not "shown" for this one file.
            body = (
                f"*{caption}*\n\n*(the image itself couldn't be displayed here, "
                "but this caption is saved and searchable)*"
            )
        messages.append(AIMessage(content=body, name="personal_upload"))

    try:
        await app.state.graph.aupdate_state(
            {"configurable": {"thread_id": thread_id}}, {"messages": messages}
        )
    except Exception as exc:  # noqa: BLE001 -- see docstring above
        print(f"[agents.api] could not post captioned image(s) to chat for "
              f"thread_id={thread_id!r}: {exc!r}", file=sys.stderr)
        return 0

    return len(captioned_images)


# Mirrors frontend/src/runtime.ts's own ATTACHMENT_MARKER_RE exactly --
# the `name=` group is the only thing this side needs out of it (the
# `status=`/`chunks=` parts are for the model to read, not for this
# server-side check). Kept in sync by hand with the frontend regex, the
# same "duplicated, not shared, across the agents/<->frontend boundary"
# tradeoff this project already accepts elsewhere for tightly-coupled
# string formats -- there's no shared build step between a Python
# backend and a TypeScript frontend to import a single definition from.
_ATTACHMENT_MARKER_RE = re.compile(r'<attachment name=(.*?) status="[^"]*"(?:\s+chunks=\d+)?>')


def _attachment_filenames_in_message(message: str) -> list[str]:
    """
    Every filename named in an `<attachment name=... status="...">`
    marker anywhere in `message` -- attachments.ts's own send() builds
    exactly this shape, and runtime.ts's onNew appends one such marker
    line per attachment onto whatever the person actually typed (see
    that file's own `messageForServer` construction) before this
    message ever reaches POST /chat. Used by `_retract_upload_preview_
    for_this_turn` below to find and retract the redundant upload-preview
    turn `_post_captioned_images_to_chat` already wrote for the SAME
    upload -- see that function's own docstring for the confirmed
    live-run duplication this closes.
    """
    return _ATTACHMENT_MARKER_RE.findall(message)


async def _retract_upload_preview_for_this_turn(thread_id: str, config: dict, message: str) -> None:
    """
    CONFIRMED live-run duplication this closes: attaching an image and
    asking a real question about it in the SAME send produced, in the
    PERSISTED transcript, four messages instead of two --
    `_post_captioned_images_to_chat` (called synchronously from POST
    .../upload, which always runs BEFORE this POST /chat call for the
    same compound send) already wrote its own "📎 Uploaded X" + caption
    pair straight into the checkpointer, and THIS turn's own real
    question + real answer land right after it. runtime.ts's own onNew
    already stops the LIVE view from re-displaying that pair a second
    time on the send that just happened (see that file's own comment on
    why it deliberately does NOT resync from the server after a
    successful send) -- but the pair was still genuinely PERSISTED as
    two extra turns, so it reappeared in full on any later GET
    /chat/{thread_id}/history: a page reload, reopening the thread, or
    hitting the API directly (e.g. via /docs), all of which read
    straight from the checkpointer with no client-side de-duplication
    to fall back on. This closes it at the source instead: the redundant
    pair is retracted from PERSISTED state itself, before this turn's
    own graph run, so every future read of this thread's history is
    clean too, not just the one live send that triggered it.

    The pair is redundant either way, not just duplicated: the real
    answer this turn's graph run is about to produce already shows the
    SAME image with its explanation combined (see specialists.py's
    retrieval_qa_node / personal_docs_node / image_qa_node -- all three
    render `image_block` alongside the real answer text on an
    image_hit), so nothing is lost by removing the upfront preview once
    a real question about that same upload has arrived.

    Only removes a pair whose filename EXACTLY matches one of THIS
    message's own `<attachment ...>` markers (see
    `_attachment_filenames_in_message`), and only the MOST RECENT such
    pair in the thread -- never an earlier upload of a same-named file
    from earlier in the same conversation, and never anything at all if
    this message carries no attachment marker (the common case, most
    turns never touch this function's body past the first check).
    Best-effort and silent on failure, same as
    `_post_captioned_images_to_chat` itself: this is tidying up a
    display redundancy, not something that should ever surface as a
    hard error on top of a person's real question.
    """
    filenames = _attachment_filenames_in_message(message)
    if not filenames:
        return

    try:
        snapshot = await app.state.graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001 -- best-effort, see docstring above
        print(f"[agents.api] could not read state to retract upload preview for "
              f"thread_id={thread_id!r}: {exc!r}", file=sys.stderr)
        return
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    if not messages:
        return

    removals: list[RemoveMessage] = []
    for filename in filenames:
        target_marker = f"📎 Uploaded {filename}"
        # Scan from the END: the pair THIS turn wants retracted is
        # always the MOST RECENT upload for this filename (the one that
        # just ran, synchronously, right before this /chat call) -- an
        # earlier upload of a same-named file earlier in the thread
        # should be left alone.
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, HumanMessage) and msg.content == target_marker:
                if getattr(msg, "id", None):
                    removals.append(RemoveMessage(id=msg.id))
                # The AIMessage(name="personal_upload") caption that
                # _post_captioned_images_to_chat always appends
                # immediately after its matching HumanMessage.
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if (
                        isinstance(next_msg, AIMessage)
                        and getattr(next_msg, "name", None) == "personal_upload"
                        and getattr(next_msg, "id", None)
                    ):
                        removals.append(RemoveMessage(id=next_msg.id))
                break

    if not removals:
        return

    try:
        await app.state.graph.aupdate_state(config, {"messages": removals})
    except Exception as exc:  # noqa: BLE001 -- best-effort, see docstring above
        print(f"[agents.api] could not retract upload preview for "
              f"thread_id={thread_id!r}: {exc!r}", file=sys.stderr)


@app.post("/chat/{thread_id}/upload", response_model=UploadResponse)
async def upload_document(thread_id: str, file: UploadFile = File(...)) -> UploadResponse:
    """
    Attach an image, PDF, or plain text file to this conversation: ingest it into
    personal_rag.py's "temp" Chroma collection, tagged with this
    thread_id, so the personal_docs specialist (agents/specialists.py)
    can answer questions about it for the rest of this conversation.

    For an IMAGE upload specifically (unlike a PDF or a .txt file), this
    endpoint also posts the caption -- and the image itself, embedded as
    a data: URI -- directly into the thread's chat transcript as a
    normal assistant turn, via _post_captioned_images_to_chat() above, so
    the person sees it right away without needing to ask a follow-up
    question first. A PDF or .txt upload's extracted text still only
    becomes retrievable (never posted as its own chat turn), via
    search_personal_documents (mcp_server/server.py) the next time the
    supervisor routes to personal_docs -- unchanged from before.

    Calls personal_rag.ingest_upload() directly, in-process -- never
    through the MCP server -- see personal_rag.py's own module docstring
    and this file's _find_pipeline_root() docstring for why ingestion is
    a direct pipeline call while SEARCHING what was ingested goes through
    MCP.

    thread_id does not need to already have any chat history -- the
    client is free to generate a thread_id up front (before the first
    POST /chat call) and upload against it immediately, exactly the way
    POST /chat itself accepts a caller-supplied thread_id for a brand-new
    conversation (see ChatRequest.thread_id's own docstring); the
    checkpointer simply has no messages for it yet until the first /chat
    call, same as today.

    400s for a file type personal_rag.py doesn't accept (see
    personal_rag.SUPPORTED_UPLOAD_EXTS -- PDF, plain text, or image files
    only). 422s if
    the file parses as empty (a corrupt/blank PDF, or an image the VLM
    couldn't caption with nothing else to fall back to) -- surfaced
    distinctly from a 400 so a caller can tell "wrong file type" apart
    from "right type, but this specific file had nothing extractable in
    it." 503s (with a deliberately generic detail -- see _invoke_turn's
    own docstring for the same reasoning) if ingestion itself raised,
    e.g. Ollama isn't running to caption an image; the real exception is
    printed to this process's own stderr, never returned to the caller.
    """
    dest_path = RAW_DOCS_DIR / f"{uuid4().hex}_{file.filename}"
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    try:
        # personal_rag.ingest_upload() is a plain synchronous function --
        # it does blocking I/O (VLM captioning over HTTP to Ollama, the
        # text embedder's own model inference, a synchronous chromadb
        # upsert) with no `await` anywhere in it. Calling it directly
        # from this async endpoint would run all of that ON the single
        # asyncio event loop thread this whole server shares -- for
        # however many seconds captioning takes, EVERY other request this
        # process is handling (a concurrent POST /chat, a GET
        # /chat/{id}/history poll, even an unrelated thread's traffic)
        # would simply hang until this one upload finished. Confirmed
        # live, not hypothetical: this is exactly the "chat seems
        # unresponsive, need to refresh the tab" symptom reported after
        # uploading images -- not a leaked resource, a blocked event
        # loop. asyncio.to_thread runs it on a worker thread instead, so
        # the event loop stays free to keep serving everyone else while
        # this upload's captioning is in flight.
        stats = await asyncio.to_thread(
            personal_rag.ingest_upload, thread_id, str(dest_path), file.filename
        )
    except ValueError as e:
        # The only ValueError personal_rag.ingest_upload() raises today
        # is an unsupported-extension one (see its own docstring) -- but
        # forwarding str(e) to the client would mean ANY future
        # ValueError it ever adds gets relayed verbatim, sight unseen,
        # by default. Same "never trust every future raise site to stay
        # benign forever" reasoning _invoke_turn's own docstring already
        # applies to every other exception in this file: the real text
        # goes to this process's own stderr (for whoever's running the
        # server), and the client gets a fixed message built from a
        # value THIS process controls (personal_rag.SUPPORTED_UPLOAD_EXTS)
        # instead of whatever text happened to be attached to the
        # exception object.
        print(f"[agents.api] upload rejected for thread_id={thread_id!r}, "
              f"file={file.filename!r}: {e!r}", file=sys.stderr)
        raise HTTPException(
            status_code=400,
            detail=(
                "This file type isn't supported for upload. Accepted types: "
                + ", ".join(personal_rag.SUPPORTED_UPLOAD_EXTS)
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- see _invoke_turn's own docstring for the same reasoning
        print(f"[agents.api] upload ingest failed for thread_id={thread_id!r}, "
              f"file={file.filename!r}: {exc!r}", file=sys.stderr)
        raise HTTPException(
            status_code=503,
            detail=(
                "This file couldn't be processed right now due to a server-side "
                "issue -- this often means a required local service (e.g. Ollama) "
                "isn't running. Please try again in a moment; if it keeps "
                "happening, let the person running this server know."
            ),
        ) from exc
    finally:
        # The staged copy under RAW_DOCS_DIR only ever existed to hand a
        # real filesystem path to ingest_path()/ingest_upload() -- for an
        # image, ingest_upload() has already copied its own persistent
        # copy out to PERSONAL_UPLOADS_DIR before returning (see
        # personal_rag._persist_personal_image), so nothing downstream
        # keeps a reference to THIS staged path either way. Always safe
        # to remove here regardless of whether ingestion succeeded,
        # failed, or raised.
        dest_path.unlink(missing_ok=True)

    if stats["n_chunks"] == 0:
        raise HTTPException(
            status_code=422,
            detail=f"{file.filename!r} was uploaded but nothing extractable was found in it "
            "(a blank/corrupt PDF, or an image that couldn't be captioned).",
        )

    captioned_images = stats.pop("captioned_images", [])
    n_images_shown = await _post_captioned_images_to_chat(thread_id, captioned_images)

    return UploadResponse(thread_id=thread_id, n_images_shown=n_images_shown, **stats)


@app.post("/chat/{thread_id}/branch", response_model=BranchResponse)
async def branch_thread(thread_id: str) -> BranchResponse:
    """
    Copy a thread's message history, as of right now, onto a brand-new
    independent thread_id -- "branch this conversation." The two threads
    share no further state after this call: continuing the ORIGINAL
    thread never touches the branch, and continuing the BRANCH never
    touches the original, which is the whole point of offering this
    (e.g. "try a different follow-up from this point without losing the
    conversation I already have").

    Implementation: rather than hand-copying rows out of the
    checkpointer's own SQLite tables (schema owned by
    langgraph-checkpoint-sqlite, not this file -- same reasoning
    DELETE /chat/{thread_id} above already gives for using
    adelete_thread instead of raw SQL there), this reads the source
    thread's current state via the graph's own aget_state(), then seeds
    the new thread_id with those exact messages via the graph's own
    aupdate_state() -- the same public, checkpointer-version-agnostic
    API LangGraph itself documents for manually writing a state update
    "as if it came from a node," without actually running any node.
    Confirmed empirically (a throwaway in-memory graph, same shape as
    this one) that this produces a real, independent first checkpoint
    for the new thread_id that GET /chat/{thread_id}/history can read
    back immediately and POST /chat can continue normally afterward,
    and that it does NOT mutate the source thread's own checkpoint chain
    in any way.

    404s if `thread_id` has no messages at all (nothing to branch from)
    -- covers both a thread_id that was never used and one that was
    already deleted, since neither has any state to copy.
    """
    source_config = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.state.graph.aget_state(source_config)
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation found for thread_id {thread_id!r} -- nothing to branch from.",
        )

    new_thread_id = str(uuid4())
    new_config = {"configurable": {"thread_id": new_thread_id}}
    await app.state.graph.aupdate_state(new_config, {"messages": messages})

    return BranchResponse(thread_id=new_thread_id, branched_from=thread_id, message_count=len(messages))


@app.get("/v1/usage")
async def usage() -> dict:
    """
    Groq free-tier rate-limit usage -- backs the small usage badge at the
    top of the chat UI (agents/static/chat.html and
    frontend/src/App.tsx/api.ts's fetchUsage()). Numbers come straight
    off Groq's own rate-limit response headers (see
    https://console.groq.com/docs/rate-limits#rate-limit-headers),
    recorded by local_rag/groq_client.py on every Groq call this process
    has made so far -- empty per-model entries just mean that model
    hasn't been called yet this run, not that something's wrong.

    Deliberately the ONLY usage-related thing exposed over HTTP: the
    dev-only cost log and request trace (local_rag/logs/cost_log.jsonl,
    local_rag/logs/request_trace.jsonl -- see
    local_rag/usage_tracker.py's own top docstring) are never read by
    this endpoint or any other -- those stay filesystem-only, visible
    only to whoever has access to the machine running this server.
    """
    return usage_tracker.get_usage_snapshot()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "iteration_cap": ITERATION_CAP,
        "route_format": ROUTE_FORMAT,
        "checkpoint_db": CHECKPOINT_DB_PATH,
    }


@app.get("/")
async def chat_ui() -> FileResponse:
    """Serve the built-in single-file browser chat UI (agents/static/chat.html)."""
    if not _CHAT_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Missing {_CHAT_HTML_PATH}")
    return FileResponse(_CHAT_HTML_PATH)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agents.api:app",
        host=os.environ.get("AGENT_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("AGENT_API_PORT", 8001)),
        reload=bool(os.environ.get("AGENT_API_RELOAD")),
    )

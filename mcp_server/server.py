"""
MCP server exposing the Production RAG System's retrieval pipeline as
tools and resources any MCP client can consume: Claude Code, Cursor,
OpenCode, or a LangGraph agent via langchain-mcp-adapters.

Wired directly against config.py / embeddings/hf_embedder.py /
vectorstore/chroma_store.py / retrieval/hybrid_retriever.py /
retrieval/reranker.py / generation/fallback_generator.py (Groq's hosted
free tier first, the original generation/ollama_generator.py's
OllamaGenerator automatically as a local fallback -- see
generation/fallback_generator.py's own module docstring).

Prerequisites (same as the rest of the project):
    - `ollama serve` running locally, with `ollama pull llama3.2` done.
    - A corpus already ingested via pipeline.py/stages.py into the
      Chroma collection defined by config.CHROMA_COLLECTION. If nothing's
      been ingested yet, retrieve() returns an empty list rather than
      erroring (see the corpus-snapshot note below).

Run from anywhere — sys.path resolution below finds config.py whether it
sits directly alongside mcp_server/ or nested under a local_rag/ folder:
    python mcp_server/server.py
    python /absolute/path/to/mcp_server/server.py
"""

import contextlib
import functools
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Force UTF-8 on every std stream, before anything else runs --------
# Confirmed cause of a class of "Connection closed" crashes reported
# against this server on Windows: this process's stdout/stderr are a pipe
# (spawned by agents/mcp_client.py's MultiServerMCPClient), not a real
# console, so Python falls back to `locale.getpreferredencoding()` for
# their text encoding -- on a typical Windows install that's cp1252, NOT
# UTF-8. Every `print()` anywhere in this process (this file's own `_log`,
# or any library several layers down) that ever has to write a non-Latin-1
# character -- Arabic text in a query being logged/echoed anywhere in the
# call chain, for one confirmed example -- raises UnicodeEncodeError right
# there. That's an unhandled exception in a plain synchronous print
# call, which is NOT something FastMCP's per-tool error handling catches
# (it isn't inside a tool call at all, half the time), so it kills this
# whole process outright -- which is exactly what the client sees as an
# opaque "Connection closed", with nothing in the client-side log to
# explain why. reconfigure() (Python 3.7+) forces UTF-8 regardless of the
# platform's locale, with errors="replace" so even a genuinely
# unencodable byte degrades to a replacement character instead of another
# crash. Doing this before ANY other import (including the _stdout_to_
# stderr helper below, which redirects INTO sys.stderr -- worthless if
# stderr itself still can't encode the message being redirected to it)
# is deliberate: nothing downstream should be able to print a single
# character before this is in effect.
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # AttributeError: some non-standard stream (e.g. under a test
        # runner) that doesn't support reconfigure at all. ValueError:
        # already detached/closed. Either way, degrade silently rather
        # than block startup over a best-effort hardening step.
        pass

# Several libraries in this pipeline print plain text to stdout on import or
# first use — chromadb's telemetry notice, huggingface_hub's download
# progress, tokenizer fork warnings, etc. MCP's stdio transport uses stdout
# as the literal wire for JSON-RPC messages, so even one stray printed line
# corrupts the handshake. The client sees this as an opaque "Connection
# closed" error with no indication anything was ever printed, since the
# process doesn't crash — it just talks over the wrong channel.
#
# Three layers of defense now: suppress the specific known-noisy env vars
# before their libraries get imported; redirect stdout to stderr around
# the component-build block below for anything that slips through anyway;
# and (see _tool_safety_net further down) redirect it AGAIN around every
# individual tool call for the life of this process, not just at startup
# -- a live tool call is exactly where the confirmed Arabic-text crash
# above actually happened, well after the component-build block's own
# protection had already ended.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb telemetry notice
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # HF tokenizers fork warning
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # huggingface_hub download bars


@contextlib.contextmanager
def _stdout_to_stderr():
    """
    Redirect stdout to stderr for the duration of the with-block. Used
    around anything that might print (library imports, model loading,
    telemetry) so stray text can't corrupt the MCP stdio protocol stream,
    which owns stdout from the moment mcp.run() is called onward.
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout


def _find_pipeline_root() -> Path:
    """
    Locate the directory that actually contains config.py — the root every
    other pipeline module imports against with bare imports like
    `from config import ...` or `from embeddings.hf_embedder import ...`.

    Checked in order, relative to this file's own location (mcp_server/):
      1. mcp_server/../config.py              (config.py directly alongside mcp_server/)
      2. mcp_server/../local_rag/config.py     (config.py nested under a local_rag/ folder)
      3. mcp_server/../../config.py            (mcp_server/ itself nested one level deeper)
      4. mcp_server/../../local_rag/config.py

    If none of these match, the error message below tells you exactly what
    was checked so you can add the right path.
    """
    here = Path(__file__).resolve().parent
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
        "Could not find config.py near mcp_server/. Checked:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nEdit _find_pipeline_root() in server.py to add your actual path."
    )


# MCP clients (Claude Code, Cursor, langchain-mcp-adapters) may spawn this
# process with a working directory we don't control, so rather than relying
# on whoever configures the client to set the right cwd, resolve the real
# pipeline root from this file's own location and put it on sys.path
# explicitly, before any of the pipeline's own bare imports run below.
_PIPELINE_ROOT = _find_pipeline_root()
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

# mcp_server/'s own directory -- needed explicitly (not just implied by
# "the script's own directory is on sys.path") because an MCP client may
# launch this file in a way that doesn't guarantee that (e.g. `python -m
# mcp_server.server` from the project root only puts the project root on
# sys.path, not mcp_server/ itself), and the new tool modules below
# (image_tools, invoice_tools, web_tools, color_tools) are imported as
# bare top-level names, siblings of this file, not as a sub-package.
_MCP_SERVER_DIR = Path(__file__).resolve().parent
if str(_MCP_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER_DIR))

with _stdout_to_stderr():
    from fastmcp import FastMCP  # noqa: E402

    from config import (  # noqa: E402
        CHROMA_CLIENT_MODE,
        CHROMA_COLLECTION,
        CHROMA_SERVER_HOST,
        CHROMA_SERVER_PORT,
        OLLAMA_GENERATION_MODELS,
    )
    from embeddings.hf_embedder import HFEmbedder  # noqa: E402
    from vectorstore.chroma_store import ChromaStore  # noqa: E402
    from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
    from retrieval.reranker import Reranker  # noqa: E402
    from generation.fallback_generator import FallbackGenerator  # noqa: E402
    from safety.domain_allowlist import ALLOWED_DOMAINS  # noqa: E402

def _log(msg: str) -> None:
    print(f"[server] {msg}", file=sys.stderr)


# Cap on how much text a single tool argument (a query, a question) is
# ever worth processing -- NOT a real-world question length, but cheap
# insurance against a caller accidentally passing something that was
# never meant to be a search query at all (e.g. a routing bug upstream
# handing an embedded base64 image data: URI, which can run to several
# megabytes of text, to retrieve()/generate_answer() as if it were a
# question). Embedding/reranking/prompting a multi-megabyte string is
# pure wasted latency at best; at worst it's the kind of oversized
# stdio message this transport has no graceful backpressure story for.
# Silently truncated (with a stderr note for whoever's running this
# server), never rejected outright -- a too-long real question should
# still get *an* answer, just from its first _MAX_TOOL_TEXT_CHARS.
_MAX_TOOL_TEXT_CHARS = 8_000


def _clip_text_arg(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) <= _MAX_TOOL_TEXT_CHARS:
        return value
    _log(f"argument {name!r} was {len(value)} chars -- clipped to "
         f"{_MAX_TOOL_TEXT_CHARS} before processing (looks more like an "
         f"accidentally-forwarded blob than a real query)")
    return value[:_MAX_TOOL_TEXT_CHARS]


def _tool_safety_net(func):
    """
    Defense-in-depth wrapper for every @mcp.tool() function below.
    Two things, for the life of THIS specific call, not just at server
    startup:

      1. Redirects stdout to stderr for the call's duration -- the same
         protection the component-build block above already gets (see
         this module's own header comment), extended to cover live tool
         calls too. That block's protection ends the moment component
         build finishes; a stray print from deep inside a library
         (sentence-transformers, chromadb, pytesseract, ...) triggered
         by a *specific* input during a *live* call -- not merely by
         importing or loading the library -- would otherwise still slip
         through onto the real stdout and corrupt the JSON-RPC stream,
         exactly the "Connection closed" failure mode this file's
         module docstring already describes.

      2. Catches any exception the wrapped tool itself doesn't already
         handle and turns it into a clearly-marked error result instead
         of letting it propagate. A tool function raising is supposed to
         be caught by FastMCP and turned into a normal MCP tool-error
         response the caller can recover from -- but if it happens
         somewhere this process's own stdio plumbing doesn't expect
         (mid-write, a background thread, etc.), the safer failure mode
         is a clean, textual error the specialist/react-agent calling
         this tool can read and route around, not a process crash that
         surfaces to the person as an opaque "Connection closed" 503.
         The real exception -- unabridged -- goes to stderr for whoever
         is running this server; NEVER into the return value itself,
         which is what a chat turn's answer can end up built from (see
         agents/specialists.py's own _looks_like_tool_error for the
         matching guard on the client side of this same problem).

    Applied as the INNERMOST decorator (`@mcp.tool()` stays outermost,
    directly above the def) -- functools.wraps below preserves the
    wrapped function's __name__/__doc__/__wrapped__/annotations, which
    is what lets @mcp.tool()'s own signature introspection (for building
    each tool's JSON schema) see straight through this wrapper to the
    real function underneath, unaffected by this wrapping.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _stdout_to_stderr():
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- see docstring above
                print(f"[server] tool {func.__name__!r} raised {exc!r} -- "
                      f"returning a safe error result instead of letting "
                      f"this crash the server process", file=sys.stderr)
                # dict/list-returning tools (retrieve, retrieve_images, ...)
                # get an empty collection back -- already this project's own
                # documented "no grounding available, not an error" contract
                # (see retrieve()'s own docstring) -- so a caller expecting
                # a list keeps working with zero special-casing. str-
                # returning tools (generate_answer) get a plain apology
                # string instead, since "" would read as a silently empty
                # answer rather than a visible failure.
                hint = str(getattr(func, "__annotations__", {}).get("return"))
                if hint.startswith("list") or hint.startswith("typing.List"):
                    return []
                if "Optional" in hint or hint.startswith("dict") or "None" in hint:
                    return None
                return (
                    "Sorry, this tool couldn't complete that request due to "
                    "a server-side issue. Please try again."
                )

    return wrapper


def _import_optional_tool_module(module_name: str):
    """
    Import one of the new-tool modules (image_tools, invoice_tools,
    web_tools) WITHOUT letting a failure in it take down the rest of
    this process.

    Before this, all three were plain top-level `import` statements --
    which means the connection between "does this one tool's module
    import cleanly" and "does the ENTIRE MCP server start at all" was a
    hard one: a single missing optional dependency (e.g. `ddgs` not
    installed, so web_tools.py's own internal `from ddgs import DDGS`
    -- itself already function-local and try/excepted, see that module's
    own docstring -- never even gets a chance to run) or any other
    import-time error in ANY ONE of these three would raise here, which
    is a plain, unhandled exception at module scope -- Python aborts
    loading server.py entirely, so retrieve()/generate_answer() and
    every other tool this server exposes go down too, even though
    nothing about the core RAG pipeline was actually broken. One
    offline/misconfigured tool taking every other tool down with it is
    exactly the failure mode this project's own "never raises, degrades
    instead" convention (see e.g. invoice_tools.build_invoice's own
    docstring, web_tools.py's module docstring) is built to avoid
    everywhere else -- this just extends that same convention to
    IMPORTING these modules, not only calling them.

    Returns the imported module, or None if it failed to import for any
    reason -- callers check for None and degrade that ONE tool's
    @mcp.tool() wrappers to a plain "unavailable" response (see e.g.
    retrieve_images() below) rather than raising AttributeError on a
    None module. Every other tool in this file, including the two other
    new-tool modules, is completely unaffected either way.
    """
    try:
        with _stdout_to_stderr():
            return __import__(module_name)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        _log(
            f"optional tool module {module_name!r} failed to import ({exc!r}) -- "
            f"that module's tool(s) will report themselves as unavailable instead "
            f"of crashing the server. Fix the underlying issue (missing dependency, "
            f"bad config, etc.) and restart this server to bring it back."
        )
        return None


# New-tool modules (web_tools, image_tools, invoice_tools, color_tools)
# live in mcp_server/ itself, not local_rag/ -- they're specific to
# *this* server's tool surface, not general-purpose pipeline components
# other scripts (pipeline.py, api.py) would ever import. sys.path already
# has _PIPELINE_ROOT on it from above (for their own `from config import
# ...` / `from safety... import ...` needs), and mcp_server/ itself is
# already importable as the package this very file lives in, so no
# further sys.path change is needed here.
#
# Each import is independent and guarded (_import_optional_tool_module
# above) -- these four modules are NOT connected to each other or to
# the core pipeline through a shared top-level `import` that would let
# one's failure cascade into the others. A module that fails to import
# becomes None; the @mcp.tool() wrappers further down check for that and
# degrade gracefully instead of crashing.
image_tools = _import_optional_tool_module("image_tools")
invoice_tools = _import_optional_tool_module("invoice_tools")
web_tools = _import_optional_tool_module("web_tools")
color_tools = _import_optional_tool_module("color_tools")

# framing_tools is the odd one out among these five: the other four
# wrap LOCAL logic (or a public, third-party API) that lives entirely
# inside this process. framing_tools.py's own request_quote() instead
# makes a network call to framing_agent/ -- a separate, independently
# deployed service (System B; see that package's own README.md) running
# in its own container. Importing framing_tools.py here still only
# imports the plain `requests`-based CLIENT code, never any part of
# System B itself -- there is no Python import across that boundary in
# either direction. Same guarded-import treatment as the other four
# either way: if framing_tools.py itself fails to import (e.g. the
# `requests` package genuinely missing, not just System B being
# unreachable at CALL time, which request_quote() already handles on
# its own), get_framing_quote() below degrades the same "unavailable"
# way image_tools/invoice_tools/etc. already do.
framing_tools = _import_optional_tool_module("framing_tools")

# personal_rag.py lives in local_rag/ itself (_PIPELINE_ROOT, already on
# sys.path above), not in mcp_server/ alongside the other four -- it's a
# pipeline module (agents/api.py's upload endpoint calls it directly too),
# not a tool module specific to this server's own surface. Imported the
# same defensive way as the four above anyway: it pulls in
# ingestion.loader (PyMuPDF for PDFs, an Ollama VLM for image captions),
# neither of which this server previously depended on, so one missing
# dependency there shouldn't be able to take retrieve()/generate_answer()
# down with it.
personal_rag = _import_optional_tool_module("personal_rag")


mcp = FastMCP("local-rag-server")


# ----------------------------------------------------------------------
# Pipeline components, built once at module load rather than per-call —
# HFEmbedder loads model weights, ChromaStore opens a persistent client,
# and FallbackGenerator opens a connection, none of which should happen
# on every tool invocation.
#
# HybridRetriever needs the full corpus client-side to build its BM25
# index (see retrieval/hybrid_retriever.py's docstring), so it's snapshotted
# once here via store.get_all() rather than re-fetched per call the way the
# hybrid_retrieve() convenience wrapper does — that wrapper is meant for
# one-off use, and rebuilding a BM25 index from the full corpus on every
# single tool call would add real latency.
#
# Known limitation worth stating plainly in the report: this snapshot goes
# stale if you ingest new documents while this server is running. Restart
# the server after re-ingesting. A live-refresh path (e.g. re-snapshotting
# on a timer, or exposing a "refresh_corpus" tool) is a reasonable Part-2
# extension but isn't implemented here.
#
# All wrapped in _stdout_to_stderr() too: HFEmbedder's first run downloads
# and loads model weights, and ChromaStore opens a persistent client — both
# real opportunities for a library to print something unexpected.
# ----------------------------------------------------------------------
with _stdout_to_stderr():
    # ChromaStore gets its own short retry here, separate from the
    # embedder/reranker handling below -- confirmed (real container
    # logs) this is a DIFFERENT failure shape, not the same "package
    # missing" story. docker-compose.yml's `depends_on: chroma-server:
    # condition: service_healthy` already waits for chroma-server's OWN
    # healthcheck before this container is even created -- but that
    # healthcheck hits chroma's /api/v2/heartbeat route specifically
    # (docker/chroma_server.Dockerfile), while chromadb.HttpClient()
    # eagerly calls get_user_identity() at CONSTRUCTION time, which
    # hits a DIFFERENT route, /auth/identity (confirmed straight from
    # the chromadb library's own traceback: client.py -> fastapi.py ->
    # "/auth/identity"). Chroma reporting its heartbeat healthy doesn't
    # guarantee every route on its API, /auth/identity included, is
    # already wired up and accepting connections at that exact instant
    # -- a real, narrow race between "container reports healthy" and
    # "the specific endpoint this client needs is live", which
    # depends_on reduces but can't fully close on its own. Five
    # attempts, 2s apart (~10s total) comfortably absorbs that gap
    # without masking a genuine, sustained chroma-server outage --
    # which still fails loudly below rather than hanging forever or
    # silently degrading, since so much of this server (not just
    # retrieve(), unlike the embedder/reranker case below) depends on
    # having a real corpus.
    _store = None
    _chroma_connect_error: Optional[Exception] = None
    for _attempt in range(1, 6):
        try:
            _store = ChromaStore(collection_name=CHROMA_COLLECTION)
            break
        except Exception as e:  # noqa: BLE001 -- any construction failure is worth one more try here
            _chroma_connect_error = e
            _log(f"chroma-server connection attempt {_attempt}/5 failed ({type(e).__name__}); "
                 f"retrying in 2s..." if _attempt < 5 else
                 f"chroma-server connection attempt {_attempt}/5 failed ({type(e).__name__}); giving up.")
            if _attempt < 5:
                time.sleep(2)
    if _store is None:
        raise RuntimeError(
            "Could not reach chroma-server after 5 attempts over ~10s -- this is past "
            "the transient startup race the retry above exists for; chroma-server "
            "itself is genuinely unreachable (wrong CHROMA_SERVER_HOST/PORT, network "
            "issue, or chroma-server actually down despite its own healthcheck)."
        ) from _chroma_connect_error

    _corpus = _store.get_all()

    # config.py's own import-time print already announced the resolved
    # CHROMA_CLIENT_MODE/target for this process (see that module's own
    # comment) -- this one confirms the CONNECTION actually succeeded
    # against that target and names the collection + chunk count, which
    # config.py can't know on its own (it doesn't construct the client
    # itself). Both together answer "which database, and does it
    # actually have anything in it" from this process's own startup log
    # alone, no live debugging session required.
    _log(f"chroma connection confirmed -- collection={CHROMA_COLLECTION!r}, {len(_corpus)} chunk(s)")

    _generator = FallbackGenerator(ollama_model=OLLAMA_GENERATION_MODELS[0])  # groq first, "llama3.2" fallback

    # HFEmbedder/Reranker are the only two components here that need
    # torch -- everything above (ChromaStore, FallbackGenerator) doesn't.
    # If this image was built WITHOUT torch/sentence-transformers (see
    # local_rag/requirements-docker.txt's header -- current Docker builds
    # reinstall them as a shared layer in docker/shared.Dockerfile's
    # `base` stage, but that layer is meant to come back out once this is
    # rewired properly), HFEmbedder()/Reranker() raise ImportError here. Letting
    # that propagate used to kill the WHOLE process before FastMCP even
    # bound a port -- every tool unavailable, including ones that never
    # touched embeddings at all (corpus_meta, invoice, framing, color,
    # web). Catching it here means the server still comes up and every
    # non-retrieval tool still works; only retrieve() (and generate_answer()
    # when called with no pre-supplied chunks) degrades, with a clear
    # reason logged once at startup instead of an opaque crash loop.
    #
    # Deliberately `except Exception`, not just `except ImportError` --
    # confirmed necessary the hard way: a torch/torchvision version
    # mismatch inside this same import chain raised a bare RuntimeError
    # ("operator torchvision::nms does not exist"), not an ImportError,
    # and an ImportError-only guard let that crash the whole server right
    # back through this same code path. This block's whole purpose is
    # "degrade instead of dying" for an optional component -- any failure
    # constructing it is reason enough to degrade, not just the specific
    # exception type this was first written against.
    try:
        _embedder = HFEmbedder()
        _reranker = Reranker()
    except Exception as e:
        _log(
            f"WARNING: embedder/reranker unavailable ({type(e).__name__}: {e}). "
            "retrieve() will return an empty list until this is fixed -- "
            "either rebuild the image with a working torch/sentence-transformers "
            "install, or rewire server.py to a non-torch embedder. Every "
            "other tool is unaffected."
        )
        _embedder = None
        _reranker = None

    # Deliberately requires BOTH a working embedder AND a non-empty
    # corpus -- HybridRetriever's dense-vector side has nothing to
    # encode queries with if _embedder is None, so there's no useful
    # partial mode (BM25-only) to fall back to here without changing
    # HybridRetriever itself, which is out of scope for this guard.
    _retriever = (
        HybridRetriever(embedder=_embedder, store=_store, corpus=_corpus)
        if _corpus and _embedder is not None
        else None
    )


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

@mcp.tool()
@_tool_safety_net
def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Retrieve the most relevant chunks from the local document corpus for a
    given natural-language query.

    Uses hybrid search (BM25 keyword matching fused with dense vector
    similarity via reciprocal rank fusion, see retrieval/hybrid_retriever.py)
    followed by cross-encoder reranking for precision (retrieval/reranker.py).

    Args:
        query: The natural-language question or search query.
        k: Number of chunks to return after reranking (default 5).

    Returns:
        A list of dicts, each with:
          - "text": the chunk's text content
          - "score": relevance score after reranking (higher is more relevant)
          - "metadata": dict with "filename", "page", and other source info
        Returns an empty list if the corpus is empty or nothing relevant is
        found — callers should treat an empty list as "no grounding
        available," not as an error.
    """
    if _retriever is None:
        return []

    query = _clip_text_arg("query", query)

    # Over-fetch a wider candidate pool before reranking, so the
    # cross-encoder has real work to do narrowing it down rather than just
    # re-sorting an already-narrow top-k. Diverges slightly from
    # pipeline.py's cmd_ask default (which reranks the same top_k it
    # retrieved) — a deliberate choice here, not an oversight; worth
    # comparing both against your labeled eval set if you want to confirm
    # it actually helps on your corpus.
    candidates = _retriever.retrieve(query, top_k=k * 3)
    reranked = _reranker.rerank(query, candidates, top_k=k)

    return [
        {
            "text": c["text"],
            "score": c.get("rerank_score", c.get("score", 0.0)),
            "metadata": c.get("metadata", {}),
        }
        for c in reranked
    ]


@mcp.tool()
@_tool_safety_net
def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate a grounded answer to a query using the provided context chunks.

    Does not retrieve anything itself — call retrieve() first and pass its
    output in as `chunks`. This separation lets a caller inspect or filter
    retrieved chunks before generation, or reuse the same chunks for
    multiple downstream questions without re-retrieving.

    Args:
        query: The question to answer.
        chunks: A list of chunk dicts as returned by retrieve(), each with
            at least "text" and "metadata" keys. Pass an empty list to get
            an answer with no grounding at all (not recommended for factual
            corpus questions — the underlying prompt template is instructed
            to say so explicitly rather than guess, but an empty context
            block gives it nothing to ground an answer OR a refusal in).

    Returns:
        The generated answer as plain text, with inline source citations
        (filename, page) per the RAG_SYSTEM_PROMPT template in
        generation/prompts.py.
    """
    query = _clip_text_arg("query", query)
    return _generator.generate(query, chunks)


@mcp.tool()
@_tool_safety_net
def search_personal_documents(thread_id: str, query: str, k: int = 5) -> list[dict]:
    """
    Search the images/PDFs/text files the user has personally uploaded INTO THIS
    CONVERSATION -- a separate, per-thread collection (local_rag/
    personal_rag.py's "temp" Chroma collection, filtered by thread_id),
    never the shared main corpus retrieve() searches.

    Args:
        thread_id: The current conversation's own thread id -- pass
            EXACTLY the thread_id this turn is running under (the
            personal_docs specialist in agents/specialists.py reads this
            straight from AgentState, never from the question text
            itself) so this can never search a different conversation's
            uploads.
        query: The natural-language question to search the person's
            uploaded documents for.
        k: Number of chunks to return (default 5).

    Returns:
        A list of dicts, each with "text", "score" (cosine similarity,
        higher is more relevant), and "metadata" (filename + whatever
        the source file contributed, e.g. "page" for a PDF) -- the exact
        same shape retrieve() returns, so generate_answer() can be
        called on this tool's output directly, unchanged.

        Returns an empty list if nothing has ever been uploaded into
        this thread_id, if nothing uploaded is relevant to `query`, or
        if personal_rag.py itself failed to import at server startup
        (see _import_optional_tool_module's docstring above) -- callers
        should treat an empty list as "no personal documents to ground
        an answer in," the same convention retrieve() already documents
        for an empty main corpus, not as an error.
    """
    if personal_rag is None:
        return []
    query = _clip_text_arg("query", query)
    return personal_rag.search_personal(thread_id, query, k=k)


@mcp.tool()
@_tool_safety_net
def latest_personal_image(thread_id: str) -> Optional[dict]:
    """
    The single most recently uploaded IMAGE in THIS CONVERSATION, chosen
    by upload recency, not by any similarity to the current question --
    see local_rag/personal_rag.py's latest_uploaded_image (this tool's
    thin wrapper target) for the full reasoning and the confirmed live
    scenario it exists for: search_personal_documents' own semantic
    ranking can rank an OLDER upload higher than the one just sent when
    two images in the same thread are visually/texturally similar,
    which is exactly the "shows a similar-but-wrong picture" failure
    mode this tool lets agents/specialists.py structurally rule out.

    Args:
        thread_id: The current conversation's own thread id -- same
            scoping rule search_personal_documents already documents:
            pass exactly this turn's own thread_id, never parsed from
            the question text, so this can never reach into a different
            conversation's uploads.

    Returns:
        A dict with "text" (the image's own VLM caption), "score"
        (always 1.0 -- there is no ranking happening here, see this
        tool's own docstring), and "metadata" (filename, image_path,
        uploaded_at, etc. -- the same shape search_personal_documents'
        hits already carry) -- OR None if this thread has never uploaded
        an image at all, or personal_rag.py itself failed to import at
        server startup. Callers should treat None as "nothing to fall
        back to," not an error, the same convention
        search_personal_documents already sets for an empty list.
    """
    if personal_rag is None:
        return None
    return personal_rag.latest_uploaded_image(thread_id)


# ----------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------

@mcp.resource("corpus://documents")
def list_documents() -> dict:
    """
    Lists all documents currently indexed in the corpus.

    Returns a dict with:
      - "documents": list of {"filename": str, "chunk_count": int}
      - "total_documents": int
      - "total_chunks": int

    Read this before answering questions about what the corpus contains,
    or to check whether a specific document has been ingested, rather than
    guessing from the query text alone. Reflects the live store, not the
    BM25 snapshot used by retrieve() — the two can briefly disagree if a
    document was ingested after this server started.
    """
    all_chunks = _store.get_all()

    by_file: dict[str, int] = {}
    for c in all_chunks:
        fname = c.get("metadata", {}).get("filename", "unknown")
        by_file[fname] = by_file.get(fname, 0) + 1

    return {
        "documents": [{"filename": f, "chunk_count": n} for f, n in by_file.items()],
        "total_documents": len(by_file),
        "total_chunks": len(all_chunks),
    }


# ----------------------------------------------------------------------
# New tools — image retrieval, internet search, and invoicing
# ----------------------------------------------------------------------
# All three are thin wrappers around image_tools.py / web_tools.py /
# invoice_tools.py — this file's job is only the @mcp.tool() docstring
# (the description every client reads, per this project's own "write
# real docstrings" convention) and the parameter/return shape MCP needs;
# the actual logic, error handling, and safety filtering live in those
# modules so they can be unit-tested without a running FastMCP server.


@mcp.tool()
@_tool_safety_net
def retrieve_images(query: str, k: int = 3) -> list[dict]:
    """
    Retrieve the most visually relevant images from the corpus for a
    natural-language query, each paired with a short caption.

    Uses CLIP cross-modal similarity (the query and every ingested image
    share one vector space, see embeddings/clip_embedder.py) — genuine
    visual retrieval, not a keyword match against filenames. Each result's
    caption is the one generated for it at ingest time
    (pipeline.py --multimodal); if a matched image somehow has none, a
    caption is generated live as a fallback so every result always has one.

    Args:
        query: Natural-language description of what to find (e.g.
            "an example of chiaroscuro", "a properly primed canvas").
        k: Number of images to return (default 3).

    Returns:
        A list of dicts, best match first, each with:
          - "image_path": local filesystem path to the image
          - "caption": short auto-generated description
          - "score": CLIP cosine similarity (higher is more relevant)
          - "metadata": dict with source filename, page, etc.
        Returns an empty list if no images have been ingested with
        --multimodal, or if the image/CLIP stack isn't installed —
        callers should treat an empty list as "no images available,"
        not as an error. Also returns an empty list (rather than
        raising) if image_tools.py itself failed to import at server
        startup -- see _import_optional_tool_module's docstring above.
    """
    if image_tools is None:
        return []
    query = _clip_text_arg("query", query)
    return image_tools.retrieve_images_with_captions(query, k=k)


@mcp.tool()
@_tool_safety_net
def retrieve_images_embedded(query: str, k: int = 3) -> list[dict]:
    """
    Same visual retrieval as retrieve_images() above, but with each
    image's actual bytes embedded in the response as base64 -- no
    filesystem path for the caller to separately resolve, no static
    file server needed to actually display anything. This tool is
    purely additive: retrieve_images() itself, and every existing
    caller of it, are untouched -- use this one instead only when the
    caller needs the pixels themselves, not just a pointer to them.

    Uses the exact same CLIP cross-modal search and ingest-time/live
    caption fallback as retrieve_images() (mcp_server/image_tools.py's
    retrieve_images_with_captions(), reused unchanged) -- this tool adds
    only the base64-encoding step on top (image_tools.py's
    retrieve_images_with_data()).

    Args:
        query: Natural-language description of what to find (e.g.
            "an example of chiaroscuro", "a properly primed canvas").
        k: Number of images to return (default 3).

    Returns:
        A list of dicts, best match first, each with:
          - "image_path": local filesystem path (kept for reference/
                debugging, same as retrieve_images())
          - "caption": short auto-generated description
          - "score": CLIP cosine similarity (higher is more relevant)
          - "metadata": dict with source filename, page, etc.
          - "image_base64": the image's raw bytes, base64-encoded, or
                None if the file is missing, unreadable, or larger than
                a 5 MB safety cap on this tool's own response payload
                (see image_tools.MAX_IMAGE_BYTES_FOR_B64)
          - "mime_type": e.g. "image/png", or None if not embedded
          - "data_uri": a ready-to-use "data:<mime_type>;base64,<data>"
                string -- usable directly as a markdown image
                (`![caption](data_uri)`) or an HTML/React `<img
                src=...>` -- or None if not embedded
          - "encoding_note": present only when image_base64 is None,
                explaining why (missing file / unreadable / over the
                size cap); the other fields stay valid even then
        Returns an empty list under the exact same conditions
        retrieve_images() does (no --multimodal corpus ingested, or the
        CLIP/image stack isn't installed), and also if image_tools.py
        itself failed to import at server startup -- never raises.
    """
    if image_tools is None:
        return []
    return image_tools.retrieve_images_with_data(query, k=k)


@mcp.tool()
@_tool_safety_net
def find_similar_images(thread_id: str, k: int = 3) -> list[dict]:
    """
    Find corpus images that VISUALLY RESEMBLE the most recently uploaded
    image in THIS CONVERSATION -- genuine image-to-image CLIP similarity
    search (image_tools.retrieve_similar_images_with_captions embeds the
    uploaded image itself with CLIP's image encoder and searches the
    same corpus image store retrieve_images() searches), not a text
    description matched against images the way retrieve_images() works.

    Use this for "find images like the one I sent" / "does the corpus
    have anything similar to my upload" / "show me paintings that look
    like this" -- never retrieve_images(), which has no picture to
    compare against, only a text query.

    Args:
        thread_id: The current conversation's own thread id -- same
            scoping rule search_personal_documents/latest_personal_image
            already document: pass exactly this turn's own thread_id,
            never parsed from the question text, so this can never
            reach into a different conversation's uploads.
        k: Number of similar corpus images to return (default 3).

    Returns:
        A list of dicts, best match first, in the exact same shape
        retrieve_images() returns ("image_path", "caption", "score",
        "metadata") -- the query image's own entry is excluded from
        these results if it happens to already exist in the corpus
        store itself (see retrieve_similar_images_with_captions's own
        docstring on `exclude_path`).

        Returns an empty list if this thread has never uploaded an
        image, if the uploaded image can't be read/embedded, if no
        corpus images have been ingested with --multimodal, or if
        personal_rag.py / image_tools.py failed to import at server
        startup -- callers should treat an empty list as "nothing to
        compare against or nothing similar found," not as an error.
    """
    if personal_rag is None or image_tools is None:
        return []
    upload = personal_rag.latest_uploaded_image(thread_id)
    if not upload:
        return []
    image_path = (upload.get("metadata") or {}).get("image_path")
    if not image_path:
        return []
    return image_tools.retrieve_similar_images_with_captions(image_path, k=k, exclude_path=image_path)


@mcp.tool()
@_tool_safety_net
def find_similar_images_embedded(thread_id: str, k: int = 3) -> list[dict]:
    """
    Same image-to-image search as find_similar_images() above, but with
    each result's actual bytes embedded as base64 -- the same additive
    "no filesystem path for the caller to resolve" relationship
    retrieve_images_embedded() already has to retrieve_images().

    Args/Returns: identical to find_similar_images(), with each result
    additionally carrying "image_base64" / "mime_type" / "data_uri" /
    (on failure) "encoding_note" -- exactly the fields
    retrieve_images_embedded() documents, same meaning here.
    """
    if personal_rag is None or image_tools is None:
        return []
    upload = personal_rag.latest_uploaded_image(thread_id)
    if not upload:
        return []
    image_path = (upload.get("metadata") or {}).get("image_path")
    if not image_path:
        return []
    return image_tools.retrieve_similar_images_with_data(image_path, k=k, exclude_path=image_path)


@mcp.tool()
@_tool_safety_net
def search_painting_online(painting_name: str) -> dict:
    """
    Look up a specific, named painting on the internet — Wikipedia first,
    supplemented by up to two more links from a general web search — and
    return a short summary plus source links.

    Use this for questions that name a specific well-known painting or
    artwork by title (e.g. "the Mona Lisa", "Starry Night", "Guernica"),
    as a source independent of (and complementary to) whatever the local
    corpus of painting/drawing treatises might separately say about
    technique — this tool answers "what is this painting / who painted
    it / what is it known for," which the corpus, being technique-focused
    treatises rather than an art encyclopedia, generally does not cover.

    Every returned URL is checked against a small, hand-curated allowlist
    of reputable museum/encyclopedic domains before being returned (see
    local_rag/safety/domain_allowlist.py) — a link this tool returns is
    never a raw, unfiltered search-engine result.

    Args:
        painting_name: The painting's title, as plainly as it's known
            (e.g. "The Starry Night", not a full sentence).

    Returns:
        {
          "query": the input painting_name,
          "summary": a short summary string, or None if nothing was
              found anywhere,
          "sources": list of {"title": str, "url": str}, possibly empty
        }
        Returns summary=None and sources=[] (never raises) if the
        internet is unreachable or nothing relevant was found — callers
        should say so plainly rather than guess from general knowledge.
        Also degrades the same way if web_tools.py itself failed to
        import at server startup, rather than crashing this tool call.
    """
    if web_tools is None:
        return {"query": painting_name, "summary": None, "sources": []}
    return web_tools.search_famous_painting(painting_name)


@mcp.tool()
@_tool_safety_net
def search_art_supplies(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the internet for real, currently-listed art supplies (brushes,
    canvases, paints, easels, etc.) on Amazon and eBay.

    ALWAYS use this for questions about buying, recommending, or comparing
    physical art supplies or tools — the local corpus contains historical
    painting *treatises*, not product listings or prices, and general
    knowledge goes stale immediately for anything involving current price
    or availability. Never answer a product question from the corpus or
    from memory; this tool is the only legitimate source for one.

    Over-fetches more candidates than `max_results` so the caller can rank
    them down (by price, listing quality, and any reputation cues visible
    in the snippet text) rather than just taking the first N as-is. The
    `product_search` agent specialist that calls this tool splits the
    results into a "beginner-friendly" and a "professional-grade" tier
    (up to 5 of each) using keyword cues in the title/snippet plus a
    price-relative-to-median tiebreak for anything unclassified — that
    split happens on the CALLER's side (agents/specialists.py), not here;
    this tool itself returns one flat, undifferentiated list, over-fetched
    generously enough (pass a higher `max_results`, e.g. 12, if you want
    two tiers with enough candidates to fill both) for that caller-side
    split to have real material to work with.

    Args:
        query: What to search for (e.g. "sable watercolor brush",
            "16x20 stretched canvas").
        max_results: Roughly how many final candidates the caller wants
            to choose from (default 5) — the actual list returned may be
            up to 2x this, before the caller's own ranking narrows it.

    Returns:
        A list of dicts, each with:
          - "title": listing title
          - "url": listing URL (always Amazon or eBay — checked against
                an allowlist, never an arbitrary domain)
          - "source": "amazon" or "ebay"
          - "price": float, best-effort extracted from the search
                snippet, or None if no price-shaped text was found —
                NOT guaranteed accurate or current; always treat as an
                estimate, not a quote
          - "snippet": the raw search-result text the price/quality
                signal was pulled from
        Returns an empty list if the internet is unreachable or the
        search backend isn't installed — never raises. Also returns an
        empty list if web_tools.py itself failed to import at server
        startup. Say so plainly to the user rather than inventing
        product data.
    """
    if web_tools is None:
        return []
    return web_tools.search_art_supplies(query, max_results=max_results)


@mcp.tool()
@_tool_safety_net
def generate_invoice(items: list[dict], customer_note: str = "") -> dict:
    """
    Build an itemized invoice (with a computed subtotal) from a list of
    priced items, and save it as a markdown file.

    All arithmetic (line totals, subtotal, item count) is computed here
    in plain Python — never delegated to an LLM — so the numbers on the
    invoice are exactly right given the input, with no risk of a model
    mis-adding a column. An item with a missing or non-numeric price is
    excluded from the subtotal and reported separately in `skipped`,
    rather than silently priced at $0.

    Every item's URL is re-checked against the same domain allowlist
    search_art_supplies already filters through, independent of that
    earlier check — an item reaching this tool came from a specialist's
    own message history, one hop removed from the original search result.

    Args:
        items: list of dicts, each with "name" (str), "price" (number),
            optionally "quantity" (int, default 1) and "url" (str).
        customer_note: optional free-text line shown on the invoice,
            purely cosmetic (never interpreted or evaluated).

    Returns:
        {
          "line_items": [...], "subtotal": float, "item_count": int,
          "skipped": [{"name", "reason"}, ...],
          "generated_at": ISO-8601 timestamp,
          "invoice_markdown": str,   # ready to show the user directly
          "file_path": str | None,  # where the .md file was saved
        }
        Never raises — a fully unpriceable `items` list comes back as an
        invoice with everything in `skipped` and a $0.00 subtotal, not a
        crash. If invoice_tools.py itself failed to import at server
        startup, this returns the same shape with every item in
        `skipped` and an explanatory note, rather than crashing.
    """
    if invoice_tools is None:
        return {
            "line_items": [],
            "subtotal": 0.0,
            "item_count": 0,
            "skipped": [
                {"name": it.get("name", "(unnamed item)"), "reason": "invoicing tool is currently unavailable"}
                for it in (items or [])
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "invoice_markdown": "_The invoice-generation tool isn't available right now._",
            "file_path": None,
        }
    return invoice_tools.build_invoice(items, customer_note=customer_note)


@mcp.tool()
@_tool_safety_net
def generate_color_palette(
    color: str = "",
    mood: str = "",
    scheme: str = "",
) -> dict:
    """
    Build a color palette, either from an explicit color or from a
    mood/feeling description, for the user's own painting -- NOT a
    lookup against corpus content. Every hex/rgb value and every
    scheme's hue math is plain, deterministic arithmetic (see
    color_tools.py's own module docstring) — never an LLM guess.

    Exactly one of `color` / `mood` should carry the real input; pass
    the other as an empty string. If both are non-empty, `color` wins.

    Args:
        color: A color as text -- a hex code ("#3f7cac"), an rgb triplet
            ("63, 124, 172" or "rgb(63, 124, 172)"), or a common color
            name ("cerulean", "forest green"). Empty string if the
            request is mood-based instead.
        mood: A mood or feeling description ("calm and peaceful",
            "bold and dramatic") to resolve to a representative color.
            Empty string if `color` is given instead.
        scheme: One of "monochromatic", "analogous", "complementary", or
            "triadic" to return only that scheme. Empty string (or any
            value that doesn't match one of the four) returns ALL FOUR.

    Returns:
        {
          "input_type": "color" | "mood" | None,
          "resolved_from_mood": [matched mood keyword, ...] | None,
          "base_color": {"hex", "rgb": {"r","g","b"}, "name", "family",
                          "feeling", "swatch"} | None,
          "schemes": {scheme_name: [ {same shape as base_color}, ... ] },
          "error": str | None,   # set, with the other fields empty/None,
                                   # if neither input resolved to a color
        }
        `swatch` is a self-contained `data:image/svg+xml;base64,...` URI
        -- embed it directly in markdown as `![...](swatch)`, no static
        file server needed. Never raises. If color_tools.py itself
        failed to import at server startup, returns the same shape with
        `error` explaining the tool is unavailable.
    """
    if color_tools is None:
        return {
            "input_type": None,
            "resolved_from_mood": None,
            "base_color": None,
            "schemes": {},
            "error": "The color-palette tool isn't available right now.",
        }
    return color_tools.generate_palette(color=color or None, mood=mood or None, scheme=scheme or None)


@mcp.tool()
@_tool_safety_net
def get_framing_quote(
    width_cm: float,
    height_cm: float,
    medium: str,
    destination_country: str,
    frame_style: str = "",
) -> dict:
    """
    Get a framing, glazing, and shipping cost estimate for ONE finished
    artwork, from System B -- an independent Google ADK + FastAPI
    service (framing_agent/, its own container) this tool reaches over
    plain HTTP, never as a Python import. See framing_tools.py's own
    module docstring for exactly why that boundary is drawn the way it
    is.

    Use this for a request about getting a finished piece FRAMED and/or
    SHIPPED -- e.g. "how much would it cost to frame and ship this
    16x20 oil painting to France," "what's a shipping estimate for a
    watercolor to Lebanon." This is a DIFFERENT question from
    search_art_supplies (raw materials -- brushes, canvas, paint -- not
    a finished-piece service) and generate_invoice (totals items
    already found via search_art_supplies) -- neither of those two
    tools has any framing or shipping pricing data at all.

    Args:
        width_cm: Artwork width in centimeters.
        height_cm: Artwork height in centimeters.
        medium: What the artwork is made of, e.g. "oil on canvas",
            "watercolor", "giclee print" -- used by System B to decide
            whether glazing is conventionally included.
        destination_country: Shipping destination, e.g. "Lebanon",
            "France". Matched against System B's own (small,
            illustrative) rate table -- an unrecognized country still
            gets an estimate, flagged as rougher than usual, never a
            hard failure.
        frame_style: Optional requested frame style ("basic wood",
            "modern metal", "classic ornate"). Leave empty for System
            B's own default.

    Returns:
        {
          "available": bool,   # False means System B itself couldn't
                                 # be reached or errored -- see "error"
          "quote": {...} | None,          # System B's full pricing
                                            # breakdown (frame/glazing/
                                            # shipping/subtotal), or
                                            # None if unavailable
          "explanation": str | None,      # ready-to-show paragraph
          "explanation_source": "groq" | "ollama" | "template" | None,
          "error": str | None,            # set only when available=False
        }
        Never raises: a stopped/unreachable/erroring framing-agent
        container degrades to available=False with a plain "error"
        string -- say so directly to the user rather than inventing a
        quote from general knowledge, which this tool has none of.
    """
    if framing_tools is None:
        return {
            "available": False, "quote": None, "explanation": None,
            "explanation_source": None,
            "error": "The framing & shipping quote tool isn't available right now.",
        }
    return framing_tools.request_quote(
        width_cm=width_cm,
        height_cm=height_cm,
        medium=medium,
        destination_country=destination_country,
        frame_style=frame_style,
    )


# ----------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------


@mcp.resource("policy://allowed-link-domains")
def allowed_link_domains() -> dict:
    """
    Lists every domain this server's internet-facing tools
    (search_painting_online, search_art_supplies) are allowed to return
    a link from, and that agents/guardrails.py's output_guard re-checks
    every outgoing link against on the way out.

    Not needed for retrieval or generation — exposed as a resource purely
    for transparency: a client (or a curious developer) can read exactly
    what's allowlisted without having to open
    local_rag/safety/domain_allowlist.py's source.

    Returns {"allowed_domains": sorted list of domain strings}.
    """
    return {"allowed_domains": sorted(ALLOWED_DOMAINS)}


@mcp.resource("policy://tool-status")
def tool_status() -> dict:
    """
    Server-wide health snapshot, in two layers:

    - The original, shallow layer (kept unchanged for backward
      compatibility -- see below): which optional new-tool modules
      (image_tools, invoice_tools, web_tools, color_tools, personal_rag)
      actually IMPORTED at server startup, same as before this resource
      was deepened.
    - `pipeline_components`, new: whether the actual heavyweight objects
      those modules depend on (the CLIP embedder, the text embedder/
      reranker, the Chroma stores) genuinely CONSTRUCTED and have real
      data, not just whether their containing module imported.

    CONFIRMED gap the second layer closes: `image_tools` importing
    successfully tells you NOTHING about whether `retrieve_images`
    actually works -- `image_tools.py` itself only tries `import
    open_clip`/`torch` lazily, inside `ClipEmbedder.__init__`, the first
    time a tool call needs it (see that module's own `_ensure_loaded`
    docstring). A server missing `open-clip-torch` entirely still
    reports `"image_tools": "available"` under the original shallow
    check alone -- exactly the gap that turned a real, live deployment
    bug into a several-turn debugging session before this existed.
    `pipeline_components.image_search` calls `image_tools.diagnostic_status()`
    (a real, if cheap, construction attempt) to catch this at the root
    instead.

    Exposed for the same transparency reason as policy://allowed-link-
    domains: a client (or a curious developer wondering why
    search_art_supplies -- or retrieve_images -- keeps coming back
    empty) can check this directly rather than guessing from tool output
    alone.
    """
    # framing_tools' own status is two independent questions, unlike
    # the other four: did the CLIENT module import (same "available"/
    # "unavailable (failed to import)" the other four report), and,
    # separately, is System B itself actually reachable RIGHT NOW over
    # the network (framing_agent_health()) -- a healthy client module
    # talking to a stopped container is a completely different failure
    # from the module itself missing, and this resource's whole point
    # is letting a developer tell those apart at a glance rather than
    # guessing from a tool call's own error text.
    framing_agent_status = "unavailable (failed to import)"
    if framing_tools is not None:
        health = framing_tools.framing_agent_health()
        framing_agent_status = "available" if health else "client loaded, but System B is unreachable"

    # Text-side retrieval stack: module-scope globals built once at
    # server startup (see the "Pipeline components" block above this
    # file's @mcp.tool()s) -- no lazy re-attempt needed, they're already
    # either real objects or None by the time any request reaches here.
    text_retrieval = {
        "ok": _retriever is not None,
        "embedder": "ok" if _embedder is not None else "unavailable (see startup logs -- likely missing torch/sentence-transformers)",
        "reranker": "ok" if _reranker is not None else "unavailable (see startup logs)",
        "chroma_chunks_indexed": len(_corpus) if _corpus else 0,
    }

    image_search = (
        image_tools.diagnostic_status()
        if image_tools is not None
        else {"ok": False, "detail": "image_tools module failed to import"}
    )

    return {
        "image_tools": "available" if image_tools is not None else "unavailable (failed to import)",
        "invoice_tools": "available" if invoice_tools is not None else "unavailable (failed to import)",
        "web_tools": "available" if web_tools is not None else "unavailable (failed to import)",
        "color_tools": "available" if color_tools is not None else "unavailable (failed to import)",
        "personal_rag": "available" if personal_rag is not None else "unavailable (failed to import)",
        "framing_tools (System B)": framing_agent_status,
        "pipeline_components": {
            "chroma_client_mode": CHROMA_CLIENT_MODE,
            "chroma_target": (
                f"http://{CHROMA_SERVER_HOST}:{CHROMA_SERVER_PORT}"
                if CHROMA_CLIENT_MODE == "http"
                else "local embedded PersistentClient"
            ),
            "text_retrieval": text_retrieval,
            "image_search": image_search,
        },
    }


# ----------------------------------------------------------------------
# Entry point — stdio by default, which is what both non-Docker consumers
# (Claude Code/Cursor/OpenCode, and agents/mcp_client.py's local dev path)
# still connect to. Set MCP_TRANSPORT=http (docker-compose.yml's
# mcp-server service does this) to instead serve over a real network
# port -- agents/mcp_client.py's build_client() switches to match via the
# same env var, so the two sides agree on transport without either one
# hardcoding a mode. MCP_SERVER_HOST/MCP_SERVER_PORT are only read in the
# http branch; stdio mode ignores them entirely, same as before.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    _transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if _transport in ("http", "streamable-http", "streamable_http"):
        _host = os.environ.get("MCP_SERVER_HOST", "0.0.0.0")
        _port = int(os.environ.get("MCP_SERVER_PORT", "8765"))
        _log(f"serving over HTTP at http://{_host}:{_port}/mcp (MCP_TRANSPORT={_transport!r})")
        mcp.run(transport="http", host=_host, port=_port)
    else:
        mcp.run()

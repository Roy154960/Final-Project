"""
Shared MCP-client plumbing for Phase 2 specialists.

Every specialist in this project talks to the retrieval pipeline through
the Phase 1 MCP server (mcp_server/server.py) -- never by importing
retrieval/hybrid_retriever.py, generation/ollama_generator.py, etc.
directly. That is the whole point of wrapping the pipeline as MCP first:
specialists (and later, the supervisor) are just MCP clients, so the same
server that Claude Code/Cursor/OpenCode talk to in Phase 1 is what the
agent graph talks to here too. One code path, not two that can drift.

This module only builds the client, loads tools/resources, and unwraps
their results -- it does not build the specialists themselves (see
specialists.py for that).
"""

import os
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# agents/ and mcp_server/ are both direct children of the project root, so
# server.py's location is derived from this file's own location rather
# than hand-edited -- same reasoning Phase 1's test_langgraph_client.py
# used, for the same reason: a wrong hand-edited path makes the spawned
# python.exe fail to even start, which surfaces client-side as an opaque
# "Connection closed" with no hint why.
#
# This path is only ever touched by the stdio branch of build_client()
# below -- it used to be validated unconditionally at import time, which
# meant importing this module at all required mcp_server/server.py to be
# present on disk even when MCP_TRANSPORT=http was going to be used and
# no local subprocess was ever going to be spawned. Docker's split-
# container layout is exactly that case: the backend image talks to a
# separate mcp-server container over the network and never needs
# mcp_server/'s source at all. Validation moved into the stdio branch
# itself so it only fires when it's actually about to matter.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT_PATH = _PROJECT_ROOT / "mcp_server" / "server.py"

# Which transport build_client() uses, and (for "http") where the server
# is. Both default to the original stdio-subprocess behavior so every
# existing non-Docker workflow (`python -3.12 -m agents.<module>`, the
# test suites, agent_mcp_server.py's own use of this same build_client())
# is completely unaffected unless these are explicitly set.
# docker-compose.yml sets MCP_TRANSPORT=http and MCP_SERVER_URL=http://
# mcp-server:8765 (the compose service name/port) for the backend
# container; mcp_server/server.py's own entry point reads the matching
# MCP_TRANSPORT/MCP_SERVER_HOST/MCP_SERVER_PORT env vars so both sides of
# the connection agree on transport from the same variable, rather than
# each hardcoding a mode that the other has to be manually kept in sync
# with.
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8765").rstrip("/")


def build_client() -> MultiServerMCPClient:
    """
    Build a MultiServerMCPClient pointed at the Phase 1 MCP server,
    either over stdio (spawning mcp_server/server.py as a local
    subprocess -- the original, still-default behavior) or over HTTP
    (connecting to an already-running server at MCP_SERVER_URL, e.g. a
    separate mcp-server container reachable by its compose service name),
    chosen by the MCP_TRANSPORT env var.

    stdio mode uses the exact interpreter running this process
    (sys.executable) rather than a bare "python" that may not resolve on
    PATH at all, or may resolve to an unrelated install -- same reasoning
    as Phase 1's test_langgraph_client.py.

    One client per graph run is the right granularity here either way:
    specialists share the same running server's BM25 snapshot and live
    Chroma store, which is what you want -- a retrieval-QA call and a
    multi-hop call in the same conversation should see the same corpus
    state, not two independently-snapshotted ones. Over HTTP that "same
    running server" is the long-lived mcp-server container rather than a
    subprocess spawned fresh per client, which is actually a stronger
    version of the same guarantee -- every graph run in every backend
    replica talks to the one server process, not one each.
    """
    if MCP_TRANSPORT in ("http", "streamable-http", "streamable_http"):
        return MultiServerMCPClient(
            {
                "local-rag": {
                    "url": f"{MCP_SERVER_URL}/mcp",
                    "transport": "streamable_http",
                }
            }
        )

    if not SERVER_SCRIPT_PATH.exists():
        raise SystemExit(
            f"Can't find mcp_server/server.py at {SERVER_SCRIPT_PATH}\n"
            "This module expects agents/ and mcp_server/ to both be direct "
            "children of the same project root. If your layout differs, edit "
            "SERVER_SCRIPT_PATH above. (If you meant to connect to a "
            "separately-running server instead of spawning one locally, "
            "set MCP_TRANSPORT=http and MCP_SERVER_URL.)"
        )
    return MultiServerMCPClient(
        {
            "local-rag": {
                "command": sys.executable,
                "args": [str(SERVER_SCRIPT_PATH)],
                "transport": "stdio",
            }
        }
    )


async def load_tools_by_name(client: MultiServerMCPClient) -> dict[str, BaseTool]:
    """
    Load every tool the server exposes and index by name, so callers can
    pick out exactly the small subset a given specialist is allowed to
    use (e.g. {"retrieve": ..., "generate_answer": ...}) instead of
    handing any specialist the full toolset -- the requirement that each
    specialist have its "own small tool set" is enforced here, at the
    point where tools get handed to a specialist builder, not by trusting
    a system prompt alone to keep an agent from reaching for a tool it
    technically has access to.
    """
    tools = await client.get_tools()
    return {t.name: t for t in tools}


async def fetch_corpus_documents(client: MultiServerMCPClient) -> dict:
    """
    Fetch and parse the corpus://documents resource.

    Returns the parsed dict straight from server.py's list_documents():
    {"documents": [...], "total_documents": int, "total_chunks": int}.
    Used to build the corpus-meta specialist, which is deliberately given
    this resource's *content*, baked into its prompt once, and nothing
    else -- see specialists.py for why that's a resource and not a tool.
    """
    blobs = await client.get_resources("local-rag", uris="corpus://documents")
    return json.loads(blobs[0].as_string())


async def fetch_tool_status(client: MultiServerMCPClient) -> dict:
    """
    Fetch and parse the policy://tool-status resource -- server.py's own
    real, construction-level health snapshot (which optional tool
    modules imported, AND whether the heavyweight components underneath
    them -- CLIP, the text embedder/reranker, Chroma -- actually
    initialized; see that resource's own docstring for exactly why the
    second half matters). Used by agents/api.py's `GET /diagnostics` to
    fold the mcp-server's own view of its health into one combined
    report, the same "one client per one-off need" pattern
    fetch_corpus_documents above already uses.
    """
    blobs = await client.get_resources("local-rag", uris="policy://tool-status")
    return json.loads(blobs[0].as_string())


def unwrap_tool_result(raw: Any) -> Any:
    """
    A LangChain tool call made through langchain-mcp-adapters comes back
    as [{"type": "text", "text": "...", "id": "..."}], not a plain Python
    value -- confirmed empirically in Phase 1's test_langgraph_client.py.
    That unwrapping is handled for you automatically when a tool is bound
    to a create_react_agent (its ToolNode does this internally), but any
    specialist here that calls a tool's .ainvoke() directly, outside of a
    ReAct loop -- the multi-hop specialist does this on purpose, to keep
    its two retrieval calls and one generation call explicit and countable
    rather than left to however many times a model decides to loop -- has
    to unwrap the result itself. This is that shared unwrapping logic,
    factored out so specialists.py doesn't duplicate it per call site.

    retrieve() returns a list[dict], so its "text" field is a JSON-encoded
    string needing json.loads(). generate_answer() returns a plain str, so
    its "text" field already IS the answer -- attempting to json.loads()
    an English sentence raises JSONDecodeError, which is caught and
    treated as "this was already plain text," not a real error.
    """
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        text = raw[0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return raw  # already unwrapped, or an unexpected shape -- surface as-is

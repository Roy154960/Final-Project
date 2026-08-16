"""
Throwaway sanity-check script for Phase 1, Step 9.

Confirms the second required consumer — a LangGraph-style client via
langchain-mcp-adapters — can connect to the same server process used by
Claude Code / Cursor / OpenCode, list the tools it exposes, and call one.

This isn't part of the agent system itself; it's just proof the server
round-trips correctly before specialist nodes get built around it in
Phase 2. Delete or move into a proper test suite once Phase 2 starts.

Run with whichever interpreter has fastmcp/langchain-mcp-adapters/your
pipeline deps installed — this spawns the server with that exact same
interpreter (via sys.executable) and finds server.py automatically as a
sibling file, so nothing needs hand-editing:
    py -3.12 mcp_server/test_langgraph_client.py
    python mcp_server/test_langgraph_client.py
"""

"""
Sanity-check script for Phase 1, Step 9 (and its follow-up: confirming
generate_answer and the corpus://documents resource work against your real
corpus and a real running Ollama, not just the fake stubs used earlier).

Confirms the second required consumer — a LangGraph-style client via
langchain-mcp-adapters — can connect to the same server process used by
Claude Code / Cursor / OpenCode, list what it exposes, and call each piece:
both tools (retrieve, generate_answer) and the one resource
(corpus://documents).

This isn't part of the agent system itself; it's just proof the server
round-trips correctly before specialist nodes get built around it in
Phase 2. Delete or move into a proper test suite once Phase 2 starts.

Prerequisites for this to fully pass (retrieve() alone doesn't need these,
but generate_answer() does):
    - ollama serve running, with `ollama pull llama3.2` done.
    - A real corpus already ingested (this test's queries assume painting/
      art-treatise content is in there — adjust QUERY below if yours differs).

Run with whichever interpreter has fastmcp/langchain-mcp-adapters/your
pipeline deps installed — this spawns the server with that exact same
interpreter (via sys.executable) and finds server.py automatically as a
sibling file, so nothing needs hand-editing:
    py -3.12 mcp_server/test_langgraph_client.py
    python mcp_server/test_langgraph_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

QUERY = "glazing technique"  # adjust to match your actual corpus content

# server.py is always a sibling of this file, so derive the path instead of
# requiring you to hand-edit a placeholder (a wrong/unedited path here makes
# python.exe fail to even start, which surfaces client-side as a generic,
# unhelpful "Connection closed" — this check turns that into a clear error).
SERVER_SCRIPT_PATH = Path(__file__).resolve().parent / "server.py"
if not SERVER_SCRIPT_PATH.exists():
    raise SystemExit(
        f"Can't find server.py at {SERVER_SCRIPT_PATH}\n"
        "This script expects server.py to sit right next to it in mcp_server/."
    )


def _unwrap_tool_result(raw):
    """
    A LangChain tool call through langchain-mcp-adapters comes back as
    [{"type": "text", "text": "...", "id": "..."}], not a plain Python
    value — this is the shape you'll need to unwrap the same way inside any
    Phase 2 specialist node that calls these tools.

    retrieve() returns a list[dict], so its "text" field is a JSON-encoded
    string that needs json.loads(). generate_answer() returns a plain str,
    so its "text" field already IS the answer — trying to json.loads() an
    English sentence raises JSONDecodeError, so that failure is caught and
    treated as "this was already plain text," not a real error.
    """
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        text = raw[0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return raw  # already unwrapped, or an unexpected shape — surface it as-is


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "local-rag": {
                # sys.executable is the exact interpreter running THIS script
                # (e.g. the one launched by `py -3.12 test_langgraph_client.py`)
                # rather than a bare "python" that may not resolve on PATH at
                # all on some Windows setups, or may resolve to a different,
                # unrelated Python install than the one you pip installed into.
                "command": sys.executable,
                "args": [str(SERVER_SCRIPT_PATH)],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s):")
    for t in tools:
        print(f"  - {t.name}: {t.description[:80]}...")

    # --- Tool 1: retrieve ---
    print(f"\n=== retrieve('{QUERY}', k=3) ===")
    retrieve_tool = next(t for t in tools if t.name == "retrieve")
    raw_chunks = await retrieve_tool.ainvoke({"query": QUERY, "k": 3})
    chunks = _unwrap_tool_result(raw_chunks)
    for c in chunks:
        print(f"  score={c['score']:.3f}  file={c['metadata'].get('filename')}  "
              f"text={c['text'][:70]}...")

    # --- Tool 2: generate_answer ---
    # This is the one that actually needs `ollama serve` running with
    # llama3.2 pulled — if it hangs or errors here, that's almost certainly
    # the cause, not the MCP wiring itself.
    print(f"\n=== generate_answer('{QUERY}', <chunks above>) ===")
    generate_tool = next(t for t in tools if t.name == "generate_answer")
    raw_answer = await generate_tool.ainvoke({"query": QUERY, "chunks": chunks})
    answer = _unwrap_tool_result(raw_answer)
    print(f"  {answer}")

    # --- Resource: corpus://documents ---
    # Resources use a different client method than tools — get_resources(),
    # not get_tools() — and come back as LangChain Blob objects, not dicts.
    print("\n=== corpus://documents resource ===")
    resources = await client.get_resources("local-rag", uris="corpus://documents")
    for blob in resources:
        doc_info = json.loads(blob.as_string())
        print(f"  {doc_info}")


if __name__ == "__main__":
    asyncio.run(main())


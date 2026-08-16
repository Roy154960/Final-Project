"""
Smoke test for agents/mcp_client.py's build_client() transport switch
(MCP_TRANSPORT / MCP_SERVER_URL env vars) -- the change that lets the
backend and mcp-server run as two separate containers talking over a
network port instead of the backend spawning mcp_server/server.py as a
stdio subprocess.

Three things checked:
  1. Unset MCP_TRANSPORT (the default) still builds the original stdio
     connection config, and still raises the original clear SystemExit
     if mcp_server/server.py isn't where it's expected -- a regression
     check for moving that existence check out of module-import time and
     into this function (see mcp_client.py's own comment on why: the
     check used to run unconditionally, which would break importing this
     module at all in a container that deliberately doesn't ship
     mcp_server/'s source in http mode).
  2. MCP_TRANSPORT=http builds a streamable_http connection config
     pointed at MCP_SERVER_URL, with NO filesystem check against
     mcp_server/server.py at all (proving the two branches are actually
     independent, not just cosmetically different).
  3. That http-mode config is not just shaped right but WORKS -- a real
     FastMCP dummy server is started as a subprocess (same mechanism
     mcp_server/server.py's own MCP_TRANSPORT=http branch uses:
     `mcp.run(transport="http", host=..., port=...)`), and
     build_client() is used, unmodified, to actually connect to it and
     call a tool. The dummy server stands in for the real
     mcp_server/server.py so this test doesn't need local_rag's full ML
     dependency stack (torch, sentence-transformers, ...) installed just
     to prove the transport mechanics -- server.py's own MCP_TRANSPORT=http
     branch is two lines of the identical `mcp.run(...)` call, exercised
     directly against a running container in docs/DOCKER.md's
     verification steps instead.

Run with:
    python agents/test_mcp_client_transport_smoke.py
    (or, from the project root: python -m agents.test_mcp_client_transport_smoke)
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_HTTP_TEST_PORT = 8798  # unlikely to collide with a real mcp-server (8765) or anything else running locally

_DUMMY_SERVER_SRC = '''
import os
from fastmcp import FastMCP

mcp = FastMCP("dummy-transport-smoke")

@mcp.tool()
def ping(name: str) -> str:
    """Say hello -- stands in for retrieve()/generate_answer() for this test's purposes."""
    return f"hello {name}"

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=int(os.environ["MCP_SERVER_PORT"]))
'''


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


def _clear_mcp_env():
    for var in ("MCP_TRANSPORT", "MCP_SERVER_URL"):
        os.environ.pop(var, None)


def _wait_for_port(port: int, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/mcp", timeout=1)
            return True  # any HTTP response (even 4xx) means something's listening
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------
# 1 & 2: build_client()'s config shape under each mode -- no network,
# no subprocess, just checking the branch produces the right connection
# dict.
# ---------------------------------------------------------------------


def test_default_mode_is_still_stdio():
    print("\n=== build_client(): unset MCP_TRANSPORT still means stdio (unchanged default) ===")
    _clear_mcp_env()
    import importlib
    import agents.mcp_client as mcp_client
    importlib.reload(mcp_client)

    _check("MCP_TRANSPORT defaults to 'stdio'", mcp_client.MCP_TRANSPORT == "stdio")

    client = mcp_client.build_client()
    conn = client.connections["local-rag"]
    _check("stdio mode's connection dict uses transport='stdio'", conn["transport"] == "stdio")
    _check("stdio mode's connection dict points at the real interpreter", conn["command"] == sys.executable)
    _check(
        "stdio mode's connection dict points at mcp_server/server.py",
        conn["args"][0] == str(mcp_client.SERVER_SCRIPT_PATH),
    )


def test_stdio_mode_existence_check_moved_not_removed():
    print("\n=== build_client(): stdio mode still validates mcp_server/server.py exists ===")
    _clear_mcp_env()
    import importlib
    import agents.mcp_client as mcp_client
    importlib.reload(mcp_client)

    real_path = mcp_client.SERVER_SCRIPT_PATH
    try:
        mcp_client.SERVER_SCRIPT_PATH = Path(tempfile.gettempdir()) / "definitely_not_here.py"
        raised = False
        try:
            mcp_client.build_client()
        except SystemExit:
            raised = True
        _check("stdio mode still raises SystemExit when the server script is missing", raised)
    finally:
        mcp_client.SERVER_SCRIPT_PATH = real_path


def test_http_mode_builds_streamable_http_config_with_no_filesystem_check():
    print("\n=== build_client(): MCP_TRANSPORT=http builds a streamable_http config ===")
    _clear_mcp_env()
    os.environ["MCP_TRANSPORT"] = "http"
    os.environ["MCP_SERVER_URL"] = "http://mcp-server:8765"
    try:
        import importlib
        import agents.mcp_client as mcp_client
        importlib.reload(mcp_client)

        # Deliberately break the stdio path's file check to prove http
        # mode never looks at it -- if this branch touched
        # SERVER_SCRIPT_PATH at all, this would raise SystemExit and
        # fail the test.
        mcp_client.SERVER_SCRIPT_PATH = Path(tempfile.gettempdir()) / "definitely_not_here.py"

        client = mcp_client.build_client()
        conn = client.connections["local-rag"]
        _check("http mode's connection dict uses transport='streamable_http'", conn["transport"] == "streamable_http")
        _check("http mode's connection dict uses MCP_SERVER_URL + /mcp", conn["url"] == "http://mcp-server:8765/mcp")
        _check("http mode's connection dict has no 'command'/'args' (never spawns anything)", "command" not in conn)
    finally:
        _clear_mcp_env()


# ---------------------------------------------------------------------
# 3: the http-mode config actually WORKS -- real dummy server, real
# socket, real tool call through build_client()'s unmodified output.
# ---------------------------------------------------------------------


def test_http_mode_actually_connects_and_calls_a_tool():
    print("\n=== build_client(): http mode round-trips a real tool call over a real socket ===")

    if shutil.which("python3") is None:
        print("  [SKIP] no python3 on PATH to spawn the dummy server")
        return

    tmp_dir = tempfile.mkdtemp(prefix="mcp_transport_smoke_")
    server_path = Path(tmp_dir) / "dummy_server.py"
    server_path.write_text(_DUMMY_SERVER_SRC)

    env = dict(os.environ)
    env["MCP_SERVER_PORT"] = str(_HTTP_TEST_PORT)
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        up = _wait_for_port(_HTTP_TEST_PORT)
        _check("dummy MCP-over-HTTP server came up", up)
        if not up:
            return

        _clear_mcp_env()
        os.environ["MCP_TRANSPORT"] = "http"
        os.environ["MCP_SERVER_URL"] = f"http://127.0.0.1:{_HTTP_TEST_PORT}"
        try:
            import importlib
            import agents.mcp_client as mcp_client
            importlib.reload(mcp_client)

            client = mcp_client.build_client()

            async def _run():
                tools = await mcp_client.load_tools_by_name(client)
                _check("real tool discovery over HTTP found the dummy tool", "ping" in tools)
                raw = await tools["ping"].ainvoke({"name": "Dominic"})
                result = mcp_client.unwrap_tool_result(raw)
                _check("real tool call over HTTP returned the expected text", result == "hello Dominic")

            asyncio.run(_run())
        finally:
            _clear_mcp_env()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_default_mode_is_still_stdio()
    test_stdio_mode_existence_check_moved_not_removed()
    test_http_mode_builds_streamable_http_config_with_no_filesystem_check()
    test_http_mode_actually_connects_and_calls_a_tool()
    print("\nAll mcp_client transport smoke tests passed.")

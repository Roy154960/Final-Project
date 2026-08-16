"""
Smoke test for vectorstore/chroma_store.py's CHROMA_CLIENT_MODE switch
(config.py's CHROMA_CLIENT_MODE / CHROMA_SERVER_HOST / CHROMA_SERVER_PORT
-- see that module's own docstring for why this exists: two containers
sharing one Chroma volume in "embedded" mode can hit SQLite's "database
is locked" under real concurrent writes, and "http" mode is the fix).

Unlike most of this project's smoke tests, the http-mode half of this one
is NOT mocked -- it starts a REAL `chroma run` server as a subprocess
(the exact CLI docker/chroma_server.Dockerfile's CMD uses) and round-
trips a real upsert/query through ChromaStore -> chromadb.HttpClient
against it, then tears the subprocess down. That's a deliberate choice:
the whole point of this change is "two processes, one server, over a
real socket" -- a mock HttpClient would only prove the branching logic
picks the right class, not that the actual wire protocol works.

Requires the `chroma` CLI on PATH (installed by `pip install chromadb`,
already in local_rag/requirements.txt) and a free TCP port
(_HTTP_TEST_PORT below) -- skips the http-mode tests with a clear message
rather than failing outright if `chroma` isn't found, so this doesn't
break a CI/dev environment that hasn't installed the full pipeline deps.

Run with:
    python local_rag/test_chroma_store_http_smoke.py
    (or, from the project root: python -m local_rag.test_chroma_store_http_smoke)
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_RAG_DIR = _PROJECT_ROOT / "local_rag"
for p in (_LOCAL_RAG_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

_HTTP_TEST_PORT = 8799  # unlikely to collide with a real chroma-server (8000) or anything else running locally


def _check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    assert condition, label


def _wait_for_heartbeat(port: int, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/api/v2/heartbeat"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------
# Embedded mode (original, default behavior) -- always runs, no
# subprocess, matches every non-Docker workflow exactly as before.
# ---------------------------------------------------------------------


def test_embedded_mode_is_still_the_default():
    print("\n=== ChromaStore: embedded mode is the unset-env-var default ===")
    import config
    _check("CHROMA_CLIENT_MODE defaults to 'embedded'", config.CHROMA_CLIENT_MODE == "embedded")

    from vectorstore.chroma_store import _build_chroma_client

    with tempfile.TemporaryDirectory() as tmp:
        client = _build_chroma_client(persist_dir=tmp)
        _check("embedded mode returns a usable client", hasattr(client, "get_or_create_collection"))
        coll = client.get_or_create_collection(name=f"smoke_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"})
        coll.upsert(ids=["x"], embeddings=[[0.1, 0.2, 0.3]], documents=["doc x"], metadatas=[{"k": 1}])
        _check("embedded mode upsert/count round-trips", coll.count() == 1)


# ---------------------------------------------------------------------
# HTTP mode -- real chroma server subprocess, real HttpClient round trip.
# ---------------------------------------------------------------------


def test_http_mode_round_trips_through_a_real_server():
    print("\n=== ChromaStore: http mode talks to a real chroma server over the network ===")

    if shutil.which("chroma") is None:
        print("  [SKIP] `chroma` CLI not on PATH (pip install chromadb) -- "
              "can't start a real server for this half of the test")
        return

    data_dir = tempfile.mkdtemp(prefix="chroma_http_smoke_")
    proc = subprocess.Popen(
        ["chroma", "run", "--host", "127.0.0.1", "--port", str(_HTTP_TEST_PORT), "--path", data_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        up = _wait_for_heartbeat(_HTTP_TEST_PORT)
        _check("test chroma server came up", up)
        if not up:
            return

        os.environ["CHROMA_CLIENT_MODE"] = "http"
        os.environ["CHROMA_SERVER_HOST"] = "127.0.0.1"
        os.environ["CHROMA_SERVER_PORT"] = str(_HTTP_TEST_PORT)
        try:
            # config.py reads these env vars at import time -- reload
            # rather than assume this is the first import in the
            # process, so this test is correct whether or not the
            # embedded-mode test above already imported config/chroma_store.
            import importlib
            import config
            importlib.reload(config)
            import vectorstore.chroma_store as chroma_store_module
            importlib.reload(chroma_store_module)
            from vectorstore.chroma_store import ChromaStore

            _check(
                "config picked up CHROMA_CLIENT_MODE=http from the environment",
                config.CHROMA_CLIENT_MODE == "http",
            )

            store = ChromaStore(collection_name=f"smoke_http_{uuid.uuid4().hex[:8]}")
            import numpy as np
            vecs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
            store.upsert(
                ids=["a", "b"],
                vectors=vecs,
                texts=["chunk A", "chunk B"],
                metadatas=[{"source": "smoke"}, {"source": "smoke"}],
            )
            _check("http-mode upsert/count round-trips through the real server", store.count() == 2)

            results = store.query(vecs[0], top_k=2)
            _check("http-mode query returns both rows", len(results) == 2)
            _check("http-mode query's closest match is the exact vector queried", results[0]["id"] == "a")
        finally:
            os.environ.pop("CHROMA_CLIENT_MODE", None)
            os.environ.pop("CHROMA_SERVER_HOST", None)
            os.environ.pop("CHROMA_SERVER_PORT", None)
            importlib.reload(config)
            importlib.reload(chroma_store_module)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    test_embedded_mode_is_still_the_default()
    test_http_mode_round_trips_through_a_real_server()
    print("\nAll chroma_store client-mode smoke tests passed.")

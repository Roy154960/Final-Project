# Chroma server image -- the piece that actually fixes the "database is
# locked" risk two containers (backend + mcp-server) sharing one Chroma
# volume can hit under real concurrent writes: instead of each container
# opening its own PersistentClient against a shared bind-mounted SQLite
# file, both talk to this ONE server process over HTTP
# (chromadb.HttpClient, via local_rag/vectorstore/chroma_store.py's
# CHROMA_CLIENT_MODE=http branch), and this process owns the single
# SQLite connection.
#
# Deliberately built from the plain `chromadb` pip package's own `chroma
# run` CLI rather than pulling Docker Hub's `chromadb/chroma` image --
# this is the exact command verified directly in a sandbox (`chroma run
# --path ... --host ... --port ...`, confirmed serving a real heartbeat
# and round-tripping an upsert/query through chromadb.HttpClient), so
# behavior here isn't a guess about what a third-party image's env vars
# or default CMD do.
#
# Build from the PROJECT ROOT:
#   docker build -f docker/chroma_server.Dockerfile -t inmind-chroma-server:latest .

FROM python:3.12-slim

RUN pip install --no-cache-dir chromadb

# Where data is persisted -- docker-compose.yml mounts a named volume
# here so the corpus/personal-upload embeddings survive a container
# restart.
RUN mkdir -p /chroma/data

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v2/heartbeat', timeout=3)" || exit 1

CMD ["chroma", "run", "--host", "0.0.0.0", "--port", "8000", "--path", "/chroma/data"]

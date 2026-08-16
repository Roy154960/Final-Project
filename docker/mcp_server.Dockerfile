# MCP server image: mcp_server/server.py, run with MCP_TRANSPORT=http so
# it listens on a real network port instead of talking stdio to a
# subprocess parent -- that's what lets this run as its OWN container,
# reachable from the backend container over docker-compose's network
# (see agents/mcp_client.py's build_client(), which switches to this same
# transport via the matching env var).
#
# local_rag/ is copied in because server.py imports directly from it
# (config, embeddings, vectorstore, retrieval, generation, safety) -- see
# that file's _find_pipeline_root(). Installs local_rag/requirements-
# docker.txt, a trimmed subset of local_rag/requirements.txt -- see that
# file's own header comment for exactly what's dropped (vllm,
# bitsandbytes, qdrant-client, ragas, arize-phoenix, unstructured, ...)
# and why each was confirmed unreachable from mcp_server/server.py's
# actual retrieve()/generate_answer()/tool code paths, traced by reading
# the real import chain and verified by actually importing
# mcp_server.server against only this trimmed set.
#
# CURRENT STATE: torch/sentence-transformers/open-clip-torch are also
# excluded entirely, on request, ahead of a planned switch to online/
# hosted models -- UNLIKE that verified-unreachable set above, these ARE
# currently imported by real code (server.py instantiates the embedder
# and reranker eagerly at module level), so THIS CONTAINER WILL FAIL TO
# START until that construction is rewired to a non-torch alternative.
# See local_rag/requirements-docker.txt's header for the full breakdown.
#
# Build from the PROJECT ROOT:
#   docker build -f docker/mcp_server.Dockerfile -t inmind-mcp-server:latest .

FROM python:3.12-slim

# poppler-utils deliberately dropped -- only backed pdf2image, which
# requirements-docker.txt confirmed is unused. See docker/backend.
# Dockerfile's identical apt-get comment for the rest.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY mcp_server/requirements.txt mcp_server/requirements.txt
COPY local_rag/requirements-docker.txt local_rag/requirements-docker.txt

# NOTE: no torch install step here (removed on request) -- see
# local_rag/requirements-docker.txt's header. UNLIKE backend, this means
# THIS CONTAINER WILL FAIL TO START until server.py's eager embedder/
# reranker construction is rewired -- see that same header for why.
RUN pip install --no-cache-dir -r local_rag/requirements-docker.txt \
    && pip install --no-cache-dir -r mcp_server/requirements.txt

# Same relative layout as the repo (mcp_server/ and local_rag/ as
# siblings under the project root) -- _find_pipeline_root()'s candidate
# list depends on this shape.
COPY mcp_server/ mcp_server/
COPY local_rag/ local_rag/

RUN mkdir -p /app/local_rag/data

# Same reasoning as docker/backend.Dockerfile's identical block: fix the
# Hugging Face cache to an explicit path (embeddings/hf_embedder.py's
# SentenceTransformer AND retrieval/reranker.py's CrossEncoder both use
# it, no cache_folder passed by either) so docker-compose.yml can mount a
# volume there and the ~90MB embedder + reranker weights survive a
# container recreation instead of re-downloading every time.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

ENV MCP_TRANSPORT=http \
    MCP_SERVER_HOST=0.0.0.0 \
    MCP_SERVER_PORT=8765 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

# No /health route on this server (it's the Phase 1 MCP server, not a
# REST API) -- curl the streamable-http endpoint itself. FastMCP replies
# 406 to a plain GET with no MCP session headers, which is still a
# "something is listening and answering" signal -- good enough for a
# liveness probe; a connection refused/timeout is the failure this is
# actually meant to catch.
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/mcp | grep -qE "^(200|406)$" || exit 1

CMD ["python", "mcp_server/server.py"]

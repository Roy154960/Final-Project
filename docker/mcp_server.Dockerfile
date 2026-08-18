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
# CURRENT STATE: local_rag/requirements-docker.txt itself still excludes
# torch/sentence-transformers/open-clip-torch, ahead of a planned switch
# to online/hosted models -- but server.py instantiates the embedder and
# reranker EAGERLY at module level, so without them THIS CONTAINER WOULD
# FAIL TO START. A dedicated stopgap layer further down in this file
# reinstalls just those three packages for this image only, until
# server.py's eager construction is rewired to a non-torch alternative
# (see that layer's own comment, and local_rag/requirements-docker.txt's
# header, for the full breakdown).
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

RUN pip install --no-cache-dir -r local_rag/requirements-docker.txt \
    && pip install --no-cache-dir -r mcp_server/requirements.txt

# --- STOPGAP: restore torch/sentence-transformers/open-clip-torch, for
# THIS image only. -----------------------------------------------------
# local_rag/requirements-docker.txt deliberately excludes these (see its
# own header) ahead of a planned switch to online/hosted models -- but
# server.py still constructs HFEmbedder()/Reranker() EAGERLY at module
# level (`_embedder = HFEmbedder()` / `_reranker = Reranker()`), so
# without them this container crashes on every single start, before
# FastMCP even binds a port. That rewiring (lazy construction + a
# non-torch fallback) hasn't landed yet, so this layer exists purely to
# keep the container alive until it does -- delete this whole block once
# server.py no longer needs it.
#
# Deliberately NOT added back to local_rag/requirements-docker.txt itself
# -- that file stays the shared, trimmed source for both backend AND
# mcp-server, and backend's own request path never touched these to
# begin with (personal_rag.py loads HFEmbedder lazily). Keeping this as
# mcp-server's own extra layer means backend's image stays exactly as
# lean as intended.
#
# Also matches what the corpus was actually embedded with: HFEmbedder's
# default model (all-MiniLM-L6-v2, 384-dim). Swapping to a different
# embedder (e.g. the non-torch embeddings/ollama_embedder.py) instead of
# reinstalling these would put query embeddings in a different vector
# space than what's already stored in Chroma -- silently wrong retrieval,
# not just a crash. Not doing that here.
#
# CPU-only wheel via PyTorch's own index -- the default PyPI `torch`
# resolves a much larger CUDA build this box will never use.
#
# open-clip-torch DELIBERATELY LEFT OUT, unlike the first version of this
# layer -- it pulls in torchvision as a hard dependency, installed from
# plain PyPI rather than the CPU-only index above, which produced a
# version/ABI mismatch against this torch build (confirmed via container
# logs: transformers, a sentence-transformers dependency, detects
# torchvision is present and tries to use it, hitting "RuntimeError:
# operator torchvision::nms does not exist" and taking the whole server
# down at import time -- worse than the original problem, since it
# doesn't even need vision support to happen). Without torchvision
# installed AT ALL, transformers' own is_torchvision_available() check
# correctly returns False and skips that code path entirely -- the
# normal, well-tested case for any text-only sentence-transformers setup.
# The only consumer of open-clip-torch is mcp_server/image_tools.py's
# ClipEmbedder, which already lazy-loads and degrades cleanly if it's
# missing (see that file's own `_embedder = None` comment) -- so
# find_similar_images becomes unavailable rather than crash-looping the
# server. Re-add open-clip-torch later only with torch/torchvision
# versions pinned together deliberately (e.g. both from the same
# https://download.pytorch.org/whl/cpu release), not as a bare name.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir sentence-transformers

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

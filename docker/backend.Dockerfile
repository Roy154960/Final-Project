# Backend image: agents/api.py (FastAPI chat API) + local_rag/ alongside it.
#
# local_rag/ is copied in (not just agents/) because agents/api.py's
# upload endpoint calls local_rag/personal_rag.py DIRECTLY, in-process --
# never through MCP -- see personal_rag.py's own module docstring for why.
# mcp_server/ is deliberately NOT copied into this image: with
# MCP_TRANSPORT=http (set below and in docker-compose.yml), agents/
# mcp_client.py's build_client() talks to the separate mcp-server
# container over the network and never touches mcp_server/server.py's
# source at all -- see that function's own docstring for the import-time
# existence check this used to require unconditionally, which this split
# specifically avoids needing.
#
# Build from the PROJECT ROOT (the directory containing agents/,
# local_rag/, mcp_server/, frontend/), not from docker/:
#   docker build -f docker/backend.Dockerfile -t inmind-backend:latest .
#
# Heads up on image size: this installs local_rag/requirements-docker.txt,
# a trimmed subset of local_rag/requirements.txt -- NOT the full research/
# benchmark dependency set (torch/vllm/bitsandbytes/qdrant-client/ragas/
# arize-phoenix/unstructured/... are NOT installed here). See that file's
# own header comment for exactly what's dropped and why each package was
# confirmed unreachable from agents/api.py's actual request path -- traced
# by reading the real import chain (including lazy/conditional imports),
# not guessed, and verified by actually importing agents.api against only
# this trimmed set. local_rag/requirements.txt itself is untouched and
# stays what the non-Docker workflow (benchmark scripts, the offline
# pipeline CLI, eval harness) installs.
#
# torch/sentence-transformers/open-clip-torch are currently EXCLUDED
# entirely (not just switched to a CPU-only build) -- see
# local_rag/requirements-docker.txt's header for exactly what that
# breaks in THIS image (personal_rag's upload/search fail at request
# time; the container itself still starts and stays healthy) pending a
# planned switch to online/hosted models.

FROM python:3.12-slim

# System libraries requirements-docker.txt's trimmed set needs at
# runtime, not just at pip-install time: tesseract-ocr backs pytesseract's
# OCR fallback for scanned PDFs (ingestion/ingest_pdf.py), libgl1/
# libglib2.0-0 satisfy Pillow/opencv-adjacent wheels, build-essential
# covers anything without a prebuilt wheel for this base image's
# platform. poppler-utils deliberately dropped -- it only backed
# pdf2image, which requirements-docker.txt confirmed has zero importers
# anywhere in either container's actual code path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements copied and installed before the rest of the source so
# `docker build` can cache this (heavy, slow) layer across source-only
# code changes -- only re-runs when a requirements file actually changes.
#
# Split into TWO separate COPY+RUN pairs, heaviest first, rather than one
# combined layer -- local_rag/requirements-docker.txt (torch +
# sentence-transformers + open-clip-torch, the genuinely slow part) gets
# its own cache layer that a change to agents/requirements.txt or
# agents/requirements-api.txt can no longer invalidate. One combined
# layer would mean editing either agents file (e.g. adding a package
# agents/mcp_client.py actually needs) busts the whole thing and
# re-installs the heavy stack too, for no reason tied to what actually
# changed.
COPY local_rag/requirements-docker.txt local_rag/requirements-docker.txt

# NOTE: no torch install step here (removed on request) -- see
# local_rag/requirements-docker.txt's header for exactly what this
# breaks (nothing at import time for this image specifically -- backend
# starts fine; personal_rag's upload/search fail at request time).
RUN pip install --no-cache-dir -r local_rag/requirements-docker.txt

COPY agents/requirements.txt agents/requirements.txt
COPY agents/requirements-api.txt agents/requirements-api.txt
RUN pip install --no-cache-dir -r agents/requirements.txt \
    && pip install --no-cache-dir -r agents/requirements-api.txt

# Same relative layout as the repo (agents/ and local_rag/ as siblings
# under the project root) -- both agents/api.py's own _find_pipeline_root
# and mcp_server-style path resolution elsewhere in this project depend
# on that shape, so it's preserved here rather than flattened.
COPY agents/ agents/
COPY local_rag/ local_rag/

# Writable location for the SQLite checkpoint DB and local_rag/data
# (personal_uploads/, raw staging) -- docker-compose.yml mounts a named
# volume here so conversation history and uploaded personal docs survive
# a container restart instead of vanishing with the container's own
# writable layer.
RUN mkdir -p /app/data

# Same reasoning, for the Hugging Face embedder weights
# (embeddings/hf_embedder.py's SentenceTransformer -- no cache_folder
# passed, so it uses whatever HF_HOME points at) -- fixed to an explicit
# path here rather than left at the default ~/.cache/huggingface so
# docker-compose.yml has something stable to mount a volume onto.
# Without this, a fresh container (every `docker compose down && up`,
# not just an image rebuild) would re-download the same ~90MB model from
# scratch every time.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

ENV AGENT_API_HOST=0.0.0.0 \
    AGENT_API_PORT=8001 \
    AGENT_API_DB_PATH=/app/data/chat_history.sqlite3 \
    PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "agents.api"]

# Shared multi-stage build for `backend` and `mcp-server` -- replaces
# the two independent docker/backend.Dockerfile and docker/mcp_server.Dockerfile
# files that used to hand-duplicate the exact same apt-get/pip install
# steps for the heavy ML dependency layer both containers need.
#
# CONFIRMED problem this exists to fix, structurally, not just once:
# those two files drifted from each other multiple times over this
# project's life --
#   - torch + sentence-transformers got added to mcp_server.Dockerfile
#     (fixing a real crash-loop) but never propagated to backend.Dockerfile,
#     so personal_rag.py's own HFEmbedder (backend's per-conversation
#     upload feature) failed at REQUEST time with
#     `ImportError('Run: pip install sentence-transformers')` -> 503.
#   - open-clip-torch never landed in EITHER file, despite three
#     separate comments across the codebase (mcp_server/requirements.txt,
#     local_rag/requirements-docker.txt) saying it was needed for
#     mcp-server's own retrieve_images -- ClipEmbedder() raised
#     ImportError on every call, silently caught and logged, so
#     retrieve_images returned [] with no visible error anywhere.
# Both were the SAME root cause: two independently hand-maintained files
# with no shared source of truth, so a fix (or a still-missing package)
# in one had no way to automatically apply to the other. A single file
# with one shared `base` stage makes that class of drift structurally
# impossible -- there is now exactly ONE place either container's ML
# dependency layer gets installed, and both images are built from it.
#
# docker-compose.yml points BOTH the `backend` and `mcp-server` services
# at THIS file, differing only in `target:` (`backend` / `mcp-server`).
# BuildKit (the default builder since Docker 23+) builds the `base` stage
# ONCE per `docker compose build` invocation and reuses its cached layers
# for both final targets -- so this doesn't just prevent drift, it also
# means the ~1-1.5GB shared ML layer (see this repo's own size notes) is
# only ever built and cached once, not twice.

FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY local_rag/requirements-docker.txt local_rag/requirements-docker.txt
RUN pip install --no-cache-dir -r local_rag/requirements-docker.txt

# The one shared ML dependency layer both containers need, unified into
# ONE install step instead of two independently-drifting ones:
#   - torch (CPU-only wheel) + sentence-transformers: backend's
#     personal_rag.py (HFEmbedder) AND mcp-server's own text
#     embedder/reranker (HFEmbedder, CrossEncoder) both need this.
#   - open-clip-torch: ONLY mcp-server's retrieve_images (ClipEmbedder)
#     actually uses this -- backend never touches CLIP directly.
#     Installed here anyway, in the SHARED stage, on purpose: the whole
#     point of this file is that "does every container that needs a
#     dependency actually have it" stops being a question anyone has to
#     re-audit by hand every time a feature is added to either service.
#     The cost is a modest amount of extra, unused size on the backend
#     image (see this project's own size notes for the actual number) --
#     a deliberate, small trade against reintroducing the exact
#     per-container drift this file exists to close. If that trade ever
#     stops being worth it, split this into two RUN steps in the
#     `backend`/`mcp-server` stages below instead of the `base` stage --
#     but that reopens the drift risk, so don't do it without a reason
#     better than "save a few hundred MB on one image."
#
# CONFIRMED problem this exact form fixes -- TWICE now, on two separate
# real `docker compose up` runs against two different versions of this
# file:
#
# Attempt 1 (torch alone from the CPU index, open-clip-torch afterward
# with no index override): open-clip-torch hard-requires `torchvision`
# (unpinned -- see its own metadata: `Requires-Dist: torchvision`) and
# pip resolved it from DEFAULT PyPI in that second command -- an
# ABI-incompatible build against the CPU-only torch already on disk.
#
# Attempt 2 (torch AND torchvision installed together, same command,
# same CPU index -- looked like it should be enough): still crashed the
# same way. Root cause: a matched pair installed in one step is not
# proof against a LATER step re-resolving one of them. Nothing here
# stops mcp_server/requirements.txt's own install (a separate `pip
# install -r ...` in the mcp-server stage below, extending this one)
# from silently pulling in a different torchvision if anything in that
# file's dependency graph needs one and doesn't pin a version -- pip has
# no memory of "these two were carefully matched" across separate
# invocations, only of what's already satisfied-or-not against whatever
# it's asked to resolve THIS time.
#
# Both attempts surfaced as the exact same runtime crash:
# `RuntimeError: operator torchvision::nms does not exist`, raised the
# moment anything imports torchvision (transformers -> torchvision.io,
# in this project's case) -- well before mcp-server even reaches FastMCP
# startup. A full crash-loop, strictly worse than the silent-degrade bug
# this dependency was added to fix in the first place.
#
# Considered and rejected: dropping the CPU-only index entirely and
# letting pip resolve torch+torchvision (and everything else) from
# plain PyPI in one pass. This WOULD guarantee a consistent resolve --
# there's no cross-index mismatch possible when there's only one index
# -- but it was measured, not guessed: plain `pip install torch
# torchvision` pulls roughly 2.2GB of NVIDIA CUDA runtime packages
# (nvidia-cublas, nvidia-cudnn, nvidia-cufft, triton, ...) as REQUIRED
# dependencies on Linux, none of which either container can ever use --
# there is no GPU here, this is CPU-only inference throughout. That
# reopens almost exactly the size regression the CPU-only index was
# introduced to close (this project's own README notes ~20GB -> ~10.6GB
# from that change), and since `backend` and `mcp-server` share this
# `base` stage, both images would carry the full 2.2GB, not just one.
#
# Actual fix: lock torch+torchvision's exact resolved versions with a
# pip CONSTRAINTS file (PIP_CONSTRAINT env var below), applied to EVERY
# subsequent `pip install` in this image -- not just the next line, but
# also mcp_server/requirements.txt's own install in the mcp-server stage
# further down, since ENV persists forward into any stage that does
# `FROM base`. If anything installed after this point would need a
# torchvision other than the one just resolved from the CPU index, pip
# now REFUSES and the BUILD fails immediately with a clear "not
# satisfiable" error -- a loud failure at build time, not a silent one
# discovered later via a crash-looping container. Costs nothing in image
# size (still the CPU-only wheels); costs a moment of build-time
# friction if a genuine incompatibility is ever introduced, which is
# exactly the point -- that friction is supposed to be visible.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip freeze | grep -iE '^(torch|torchvision)==' > /tmp/torch-constraints.txt \
    && cat /tmp/torch-constraints.txt

ENV PIP_CONSTRAINT=/tmp/torch-constraints.txt

RUN pip install --no-cache-dir sentence-transformers open-clip-torch

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface


# ---------------------------------------------------------------------
# backend -- agents/api.py
# ---------------------------------------------------------------------
FROM base AS backend

COPY agents/requirements.txt agents/requirements.txt
COPY agents/requirements-api.txt agents/requirements-api.txt
RUN pip install --no-cache-dir -r agents/requirements.txt \
    && pip install --no-cache-dir -r agents/requirements-api.txt

COPY agents/ agents/
COPY local_rag/ local_rag/

RUN mkdir -p /app/data

ENV AGENT_API_HOST=0.0.0.0 \
    AGENT_API_PORT=8001 \
    AGENT_API_DB_PATH=/app/data/chat_history.sqlite3 \
    PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "agents.api"]


# ---------------------------------------------------------------------
# mcp-server -- mcp_server/server.py
# ---------------------------------------------------------------------
FROM base AS mcp-server

COPY mcp_server/requirements.txt mcp_server/requirements.txt
RUN pip install --no-cache-dir -r mcp_server/requirements.txt

COPY mcp_server/ mcp_server/
COPY local_rag/ local_rag/

RUN mkdir -p /app/local_rag/data

ENV MCP_TRANSPORT=http \
    MCP_SERVER_HOST=0.0.0.0 \
    MCP_SERVER_PORT=8765 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/mcp | grep -qE "^(200|406)$" || exit 1

CMD ["python", "mcp_server/server.py"]

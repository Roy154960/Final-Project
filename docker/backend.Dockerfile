FROM python:3.12-slim-bookworm

# See mcp_server.Dockerfile for why -- deb.debian.org's plain-http://
# endpoint was getting 403'd from this network on both trixie and
# bookworm, so route apt over https instead. Covers both the classic
# sources.list and the newer deb822 sources.list.d/*.sources format.
RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do \
        [ -f "$f" ] && sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' "$f"; \
    done; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY local_rag/requirements-docker.txt local_rag/requirements-docker.txt


RUN pip install --no-cache-dir -r local_rag/requirements-docker.txt

COPY agents/requirements.txt agents/requirements.txt
COPY agents/requirements-api.txt agents/requirements-api.txt
RUN pip install --no-cache-dir -r agents/requirements.txt \
    && pip install --no-cache-dir -r agents/requirements-api.txt

COPY agents/ agents/
COPY local_rag/ local_rag/


RUN mkdir -p /app/data


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

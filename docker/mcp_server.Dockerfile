FROM python:3.12-slim

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

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir sentence-transformers

COPY mcp_server/ mcp_server/
COPY local_rag/ local_rag/

RUN mkdir -p /app/local_rag/data

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

ENV MCP_TRANSPORT=http \
    MCP_SERVER_HOST=0.0.0.0 \
    MCP_SERVER_PORT=8765 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/mcp | grep -qE "^(200|406)$" || exit 1

CMD ["python", "mcp_server/server.py"]

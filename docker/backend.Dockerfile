FROM python:3.12-slim

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

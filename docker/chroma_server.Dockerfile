FROM python:3.12-slim

RUN pip install --no-cache-dir chromadb

RUN mkdir -p /chroma/data

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v2/heartbeat', timeout=3)" || exit 1

CMD ["chroma", "run", "--host", "0.0.0.0", "--port", "8000", "--path", "/chroma/data"]

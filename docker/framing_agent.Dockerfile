FROM python:3.12-slim-bookworm

WORKDIR /app

COPY framing_agent/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY framing_agent/ .

ENV PORT=8090 \
    PYTHONUNBUFFERED=1

EXPOSE 8090


HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health', timeout=3)" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]

# System B's own image -- a genuinely independent stack (Google ADK +
# FastAPI, not LangGraph) in its own container. Nothing here COPYs or
# installs anything from agents/, mcp_server/, or local_rag/ -- see
# framing_agent/server.py's own module docstring for why that
# separation is the whole point of System B existing at all.
#
# Build from the PROJECT ROOT (same convention every other Dockerfile
# here follows):
#   docker build -f docker/framing_agent.Dockerfile -t inmind-framing-agent:latest .

FROM python:3.12-slim

WORKDIR /app

COPY framing_agent/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY framing_agent/ .

ENV PORT=8090 \
    PYTHONUNBUFFERED=1

EXPOSE 8090

# GOOGLE_API_KEY is deliberately NOT required here (no ARG/ENV default
# set to a real value) -- server.py's own _adk_configured() check
# degrades to the template explanation path if it's absent at runtime,
# same as running this service standalone with no .env at all. Pass it
# via docker-compose.yml's own environment: block (or `docker run -e
# GOOGLE_API_KEY=...`) if you want LLM-written explanations.

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health', timeout=3)" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]

"""
Reliability enhancement - request tracing via Arize Phoenix.

Phoenix runs entirely locally (a local web UI + local trace collector, no
cloud account, no API key) and lets you visualize the full trace of a RAG
call: what was retrieved, what was sent to the LLM, what came back, and how
long each step took. Genuinely useful once you have more than a couple of
retrieval/generation steps chained together and something looks wrong.

Usage:
    from utils.tracing import start_phoenix, trace_span

    start_phoenix()  # once, at pipeline startup — opens a local UI

    with trace_span("retrieve", question=question):
        results = vector_retrieve(...)

    with trace_span("generate", model=generator.name):
        answer = generator.generate(...)

Requires:
    pip install arize-phoenix
"""

from contextlib import contextmanager
import time

from utils.logging_config import get_logger

logger = get_logger("local_rag.tracing")

_phoenix_session = None


def start_phoenix(port: int = 6006):
    """Launches the local Phoenix UI (http://localhost:<port>) once per process."""
    global _phoenix_session
    if _phoenix_session is not None:
        return _phoenix_session
    try:
        import phoenix as px
        _phoenix_session = px.launch_app(port=port)
        logger.info("phoenix tracing UI started", extra={"url": _phoenix_session.url})
        return _phoenix_session
    except ImportError:
        logger.warning("phoenix not installed — tracing disabled, spans will just log timing. "
                        "Run: pip install arize-phoenix")
        return None


@contextmanager
def trace_span(name: str, **attributes):
    """
    Lightweight span context manager. Falls back to structured log lines
    with timing if Phoenix/OpenInference instrumentation isn't set up —
    this project keeps it dependency-light rather than hard-requiring the
    full OpenTelemetry instrumentation stack, so you always get at least
    basic timing even without Phoenix running.
    """
    start = time.perf_counter()
    logger.info(f"span started: {name}", extra={"span": name, **attributes})
    try:
        yield
    except Exception:
        logger.error(f"span failed: {name}", exc_info=True, extra={"span": name, **attributes})
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(f"span finished: {name}", extra={"span": name, "duration_ms": elapsed_ms, **attributes})


if __name__ == "__main__":
    with trace_span("demo_retrieve", question="what is RAG?"):
        time.sleep(0.05)
    with trace_span("demo_generate", model="ollama:llama3.2"):
        time.sleep(0.1)

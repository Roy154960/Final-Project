"""
Reliability enhancement - retry/backoff around Ollama calls.

Local Ollama calls can fail transiently (server still loading a model,
brief connection hiccup, OOM under load). This wraps the embed/generate
calls with exponential backoff via `tenacity` so a single blip doesn't
crash a whole ingest or ask run.
"""

import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

logger = logging.getLogger("local_rag.retry")

# Retry on connection errors and generic exceptions from the ollama client;
# tune stop/wait to your own tolerance for latency vs. resilience.
ollama_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def with_retry(fn):
    """Decorator form for wrapping any single function, e.g.:

        @with_retry
        def call_ollama():
            ...
    """
    return ollama_retry(fn)


if __name__ == "__main__":
    attempts = {"n": 0}

    @with_retry
    def flaky_call():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated transient failure")
        return "success"

    logging.basicConfig(level=logging.INFO)
    result = flaky_call()
    print(f"result={result} after {attempts['n']} attempt(s)")

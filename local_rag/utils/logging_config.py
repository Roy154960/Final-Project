"""
Reliability enhancement - structured logging.

Plain print() statements (used in the early benchmark scripts for quick
readability) don't carry timestamps, log levels, or module names, and
can't be filtered/shipped anywhere. This sets up JSON-structured logging
so every stage of the pipeline logs consistently and the output can be
piped into any log aggregator later if you deploy this.

Usage, from any module:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("ingested document", extra={"doc_id": doc.doc_id, "modality": doc.modality})
"""

import logging
import json
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via `extra={...}` gets merged in as structured fields
        standard_keys = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in standard_keys and key not in ("message", "asctime"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers if get_logger is called repeatedly
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


if __name__ == "__main__":
    logger = get_logger("local_rag.demo")
    logger.info("pipeline started", extra={"stage": "ingest", "n_files": 12})
    logger.warning("chunk near size limit", extra={"chunk_id": "abc123", "n_tokens": 8100})
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("something failed", exc_info=True, extra={"stage": "embed"})

"""Structured logging setup for pipeline/benchmark execution.

CLAUDE.md and instructions.md require structured logging instead of
`print` for anything that runs as part of a benchmark or pipeline stage,
so that pipeline stage, config identity, and run provenance are
machine-parseable in log output. This wraps the standard library's
`logging` module with a JSON formatter rather than adding a new
dependency (e.g. structlog) that isn't required by any documented spec.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)

        return json.dumps(payload)


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger configured with JSON structured output.

    Safe to call repeatedly (e.g. once per module) -- handlers are only
    attached once per logger name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

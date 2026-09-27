from __future__ import annotations

import builtins
import json
import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

_BUILTIN_EXCEPTIONS = {
    value: value.__name__
    for value in vars(builtins).values()
    if isinstance(value, type) and issubclass(value, BaseException)
}
_REQUEST_EVENTS = frozenset({"request_failed", "request_completed"})


class JsonFormatter(logging.Formatter):
    """Emit HTTP event metadata without exception values, source text or locals."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": (
                record.msg
                if isinstance(record.msg, str) and record.msg in _REQUEST_EVENTS
                else "application_event"
            ),
        }
        for field in ("request_id", "method", "path", "status_code", "duration_ms"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info and record.exc_info[1] is not None:
            frames: deque[dict[str, Any]] = deque(maxlen=20)
            trace = record.exc_info[2]
            while trace is not None:
                frame = trace.tb_frame
                frames.append({
                    "module": frame.f_globals.get("__name__", "<unknown>"),
                    "function": frame.f_code.co_name,
                    "line": trace.tb_lineno,
                })
                trace = trace.tb_next
            payload["exception"] = {
                "category": _BUILTIN_EXCEPTIONS.get(type(record.exc_info[1]), "Exception"),
                "frames": list(frames),
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_http_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("tramflow.http")
    logger.setLevel(level.upper())
    logger.propagate = False
    if not any(getattr(handler, "_tramflow_json", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._tramflow_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger

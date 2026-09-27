from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from sqlalchemy.exc import StatementError

from app.api.middleware.observability import ObservabilityMiddleware
from app.infrastructure.observability import JsonFormatter, configure_http_logger

_SECRET = "synthetic-password-UNIQUE-913"
_PASSENGER = "synthetic-passenger-UNIQUE-728"


def _record(error: BaseException) -> logging.LogRecord:
    return logging.LogRecord(
        "tramflow.http",
        logging.ERROR,
        __file__,
        1,
        "request_failed",
        (),
        (type(error), error, error.__traceback__),
    )


def _failure(kind: str) -> BaseException:
    try:
        try:
            raise StatementError(
                _SECRET,
                "SELECT passenger FROM validations WHERE token=:token",
                {"token": _PASSENGER},
                ValueError(_SECRET),
            )
        except StatementError as original:
            if kind == "cause":
                raise RuntimeError(_SECRET) from original
            if kind == "context":
                raise ValueError(_PASSENGER)  # noqa: B904 - exercise implicit exception context
            if kind == "group":
                raise ExceptionGroup(_SECRET, [original, ValueError(_PASSENGER)]) from original
            raise
    except Exception as error:
        error.add_note(_PASSENGER)
        return error


@pytest.mark.parametrize(
    ("kind", "category"),
    [
        ("cause", "RuntimeError"),
        ("context", "ValueError"),
        ("group", "ExceptionGroup"),
        ("sql", "Exception"),
    ],
)
def test_exception_diagnostics_exclude_all_exception_text(kind: str, category: str) -> None:
    record = _record(_failure(kind))
    record.exc_text = f"cached traceback {_SECRET}"
    record.stack_info = f"stack source {_PASSENGER}"
    serialized = JsonFormatter().format(record)
    payload = json.loads(serialized)

    for forbidden in (_SECRET, _PASSENGER, "SELECT passenger", "raise StatementError"):
        assert forbidden not in serialized
    diagnostic = payload["exception"]
    assert set(diagnostic) == {"category", "frames"}
    assert diagnostic["category"] == category
    assert diagnostic["frames"]
    for frame in diagnostic["frames"]:
        assert set(frame) == {"module", "function", "line"}
        assert isinstance(frame["module"], str)
        assert isinstance(frame["function"], str)
        assert isinstance(frame["line"], int) and frame["line"] > 0
    assert diagnostic["frames"][-1]["function"] == "_failure"


def test_exception_diagnostics_never_call_exception_str() -> None:
    class UnsafeException(Exception):
        def __str__(self) -> str:
            pytest.fail("Formatting invoked untrusted exception __str__")

    try:
        raise UnsafeException(_SECRET)
    except UnsafeException as error:
        payload = json.loads(JsonFormatter().format(_record(error)))

    assert payload["exception"]["category"] == "Exception"


def test_exception_class_names_are_not_treated_as_safe_text() -> None:
    custom = type(_SECRET, (RuntimeError,), {})
    serialized = JsonFormatter().format(_record(custom(_PASSENGER)))
    assert _SECRET not in serialized
    assert _PASSENGER not in serialized
    assert json.loads(serialized)["exception"] == {"category": "Exception", "frames": []}


def test_exception_frames_keep_only_last_twenty_locations() -> None:
    def descend(depth: int) -> None:
        if depth:
            descend(depth - 1)
        else:
            raise RuntimeError(_SECRET)

    try:
        descend(35)
    except RuntimeError as error:
        diagnostic = json.loads(JsonFormatter().format(_record(error)))["exception"]

    assert diagnostic["category"] == "RuntimeError"
    assert len(diagnostic["frames"]) == 20
    assert all(frame["function"] == "descend" for frame in diagnostic["frames"])


@pytest.fixture
def captured_http_log() -> Iterator[io.StringIO]:
    logger = logging.getLogger("tramflow.http")
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    configured = configure_http_logger("INFO")
    stream = io.StringIO()
    assert len(configured.handlers) == 1
    handler = configured.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert isinstance(handler.formatter, JsonFormatter)
    handler.setStream(stream)
    try:
        yield stream
    finally:
        logger.handlers, logger.level, logger.propagate = previous


@pytest.mark.parametrize("outcome", ["success", "failure", "unmatched"])
def test_http_logs_preserve_safe_context_and_error_contract(
    captured_http_log: io.StringIO,
    outcome: str,
) -> None:
    application = FastAPI()
    application.add_middleware(ObservabilityMiddleware)
    application.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"])

    @application.get("/passengers/{passenger_id}")
    def endpoint(passenger_id: str) -> dict[str, bool]:
        if outcome == "failure":
            raise _failure("cause")
        return {"ok": True}

    url = f"/passengers/{_PASSENGER}?token={_SECRET}"
    if outcome == "unmatched":
        url = f"/unknown/{_PASSENGER}?token={_SECRET}"
    request_id = "request-123.valid_ID"
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get(
            url,
            headers={
                "X-Request-ID": request_id,
                "Origin": "http://localhost:5173",
                "Authorization": f"Bearer {_SECRET}",
            },
        )

    expected_status = {"success": 200, "failure": 500, "unmatched": 404}[outcome]
    assert response.status_code == expected_status
    assert response.headers["X-Request-ID"] == request_id
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    if outcome == "failure":
        assert response.json() == {"detail": "Internal Server Error", "request_id": request_id}
    elif outcome == "success":
        assert response.json() == {"ok": True}
    serialized = captured_http_log.getvalue()
    for marker in (_SECRET, _PASSENGER):
        assert marker not in serialized
        assert marker not in response.text
    records: list[dict[str, Any]] = [json.loads(line) for line in serialized.splitlines()]
    assert [record["message"] for record in records] == (
        ["request_failed", "request_completed"] if outcome == "failure" else ["request_completed"]
    )
    for record in records:
        assert record["request_id"] == request_id
        assert record["method"] == "GET"
        assert record["path"] == (
            "<unmatched>" if outcome == "unmatched" else "/passengers/{passenger_id}"
        )
    assert records[-1]["status_code"] == expected_status
    assert records[-1]["duration_ms"] >= 0
    if outcome == "failure":
        assert records[0]["exception"]["category"] == "RuntimeError"
        assert any(frame["function"] == "endpoint" for frame in records[0]["exception"]["frames"])


@pytest.mark.parametrize("event", ["request_failed", "request_completed"])
def test_formatter_preserves_allowed_event_names(event: str) -> None:
    record = logging.LogRecord("tramflow.http", logging.INFO, __file__, 1, event, (), None)
    assert json.loads(JsonFormatter().format(record))["message"] == event


def test_unknown_log_messages_are_not_formatted() -> None:
    class UnsafeMessage:
        def __str__(self) -> str:
            pytest.fail("Formatting invoked untrusted message __str__")

    record = logging.LogRecord(
        "tramflow.http",
        logging.ERROR,
        __file__,
        1,
        UnsafeMessage(),
        (_SECRET,),
        None,
    )
    record.exc_text = _SECRET
    record.stack_info = _PASSENGER
    serialized = JsonFormatter().format(record)
    assert json.loads(serialized)["message"] == "application_event"
    assert "exception" not in json.loads(serialized)
    assert _SECRET not in serialized
    assert _PASSENGER not in serialized

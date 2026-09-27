"""Every failure must be JSON, whatever produced it -- and reach the browser.

A client that calls .json() on every response should never have to special-case a
content type. The 500 path is the one FastAPI does not cover by itself, and the
CORS test below is not decoration: an `Exception` handler registered on the app
runs outside CORSMiddleware, so its response has no Access-Control-Allow-Origin
and a browser rejects it before the body is ever read. TestClient does not enforce
CORS, so only an explicit header assertion catches that.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app

_BOOM = "/__test_boom"
_LEAKY_MESSAGE = "connection to 10.0.0.5:5432 failed for user tramflow_admin"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    @app.get(_BOOM)
    async def boom() -> dict[str, str]:  # pragma: no cover - always raises
        raise RuntimeError(_LEAKY_MESSAGE)

    # raise_server_exceptions=False makes TestClient behave like a real server
    # and return the handler's response instead of re-raising into the test.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

    app.router.routes = [
        route for route in app.router.routes if getattr(route, "path", None) != _BOOM
    ]


def test_unhandled_exception_returns_json(client: TestClient) -> None:
    response = client.get(_BOOM)

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == "Internal Server Error"


def test_unhandled_exception_hides_internal_detail(client: TestClient) -> None:
    response = client.get(_BOOM)

    assert _LEAKY_MESSAGE not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text


def test_unhandled_exception_is_traceable(client: TestClient) -> None:
    supplied = "trace-me-1"
    response = client.get(_BOOM, headers={"X-Request-ID": supplied})

    # Same id in the body and the header as the log line records, so a reported
    # 500 can be found in the logs.
    assert response.json()["request_id"] == supplied
    assert response.headers["X-Request-ID"] == supplied


def test_error_response_is_readable_cross_origin(client: TestClient) -> None:
    origin = "http://localhost:5173"
    response = client.get(_BOOM, headers={"Origin": origin})

    # Without this header the browser discards the 500 as a CORS failure and the
    # frontend reports a network error instead of the backend's message.
    assert response.headers.get("access-control-allow-origin") == origin


def test_unknown_route_is_json(client: TestClient) -> None:
    response = client.get("/api/v1/definitely-not-a-route")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_validation_error_is_json(client: TestClient) -> None:
    response = client.get("/api/v1/tram-graph/path", params={"from": "abc", "to": 1})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")

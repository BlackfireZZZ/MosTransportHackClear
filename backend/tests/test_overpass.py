"""Overpass adapter and endpoints.

The upstream instance is always faked: it is a shared box with a 4-query concurrency
ceiling, and a test suite must not spend slots on it or fail when it is down.

Every case below is one of the five response shapes in `docs/overpass-api.md`. The
two that took real debugging are the HTML error page arriving under HTTP 400, and the
error text straddling a `<strong>` boundary so that naive extraction yields the
useless string "Error".
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_overpass_query_enabled, get_overpass_service
from app.application.services.overpass import OverpassService
from app.core.config import Settings
from app.domain.overpass import OverpassError, OverpassQueryTooLongError
from app.infrastructure.overpass import (
    HttpOverpassGateway,
    OverpassTransportError,
    RawResponse,
    error_message,
)
from app.main import app

BASE = "/api/v1/overpass"

HTML_ERROR_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<html><body>
<p><strong style="color:#FF0000">Error</strong>: line 3: parse error: Unknown type "nodz" </p>
<p>runtime error: Query timed out in "query" at line 3 after 60 seconds.</p>
</body></html>"""

# The same page shape carrying only the <strong>-wrapped parse error: no "runtime
# error" line to fall back on.
HTML_PARSE_ERROR_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<html><body>
<p><strong style="color:#FF0000">Error</strong>: line 3: parse error: Unknown type "nodz" </p>
</body></html>"""

OSM3S_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="Overpass API"><node id="1"/></osm>"""


class FakeTransport:
    """Stands in for the network: replays one canned response, or raises."""

    def __init__(self, response: RawResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[tuple[str, bytes | None, float]] = []

    async def fetch(self, url: str, data: bytes | None, read_timeout: float) -> RawResponse:
        self.calls.append((url, data, read_timeout))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def gateway(transport: FakeTransport) -> HttpOverpassGateway:
    return HttpOverpassGateway(
        interpreter_url="http://overpass.test/api/interpreter",
        status_url="http://overpass.test/api/status",
        query_timeout=60,
        health_timeout=3,
        transport=transport,
    )


def service(transport: FakeTransport, max_query_chars: int = 8000) -> OverpassService:
    return OverpassService(gateway(transport), max_query_chars=max_query_chars)


@contextmanager
def api(transport: FakeTransport, max_query_chars: int = 8000) -> Iterator[TestClient]:
    """The real router, service and adapter, with only the network replaced."""
    app.dependency_overrides[get_overpass_service] = lambda: service(transport, max_query_chars)
    app.dependency_overrides[get_overpass_query_enabled] = lambda: True
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_overpass_service, None)
        app.dependency_overrides.pop(get_overpass_query_enabled, None)


def test_query_proxy_is_disabled_without_explicit_enablement() -> None:
    transport = FakeTransport(RawResponse(200, "{}"))
    app.dependency_overrides[get_overpass_service] = lambda: service(transport)
    try:
        with TestClient(app) as client:
            response = client.post(f"{BASE}/query", json={"query": "[out:json];node(1);out;"})
        assert response.status_code == 403
        assert response.json() == {"detail": "Overpass query proxy is disabled"}
        assert transport.calls == []
    finally:
        app.dependency_overrides.pop(get_overpass_service, None)


# --- the adapter ---------------------------------------------------------------


async def test_successful_query_is_form_encoded() -> None:
    transport = FakeTransport(RawResponse(200, '{"elements": [{"id": 1}]}'))

    result = await gateway(transport).run_query("[out:json];node(1);out;")

    assert result.payload == {"elements": [{"id": 1}]}
    assert result.remark is None
    url, data, read_timeout = transport.calls[0]
    assert url == "http://overpass.test/api/interpreter"
    assert data is not None and data.startswith(b"data=")
    assert read_timeout == 90  # the query's own claim plus headroom


async def test_remark_is_lifted_out_of_a_200_partial_result() -> None:
    body = '{"version": 0.6, "elements": [], "remark": "runtime error: Query timed out"}'
    transport = FakeTransport(RawResponse(200, body))

    result = await gateway(transport).run_query("[out:json];node(1);out;")

    assert result.remark == "runtime error: Query timed out"
    assert result.payload["elements"] == []


async def test_html_error_under_200_keeps_every_error_line() -> None:
    transport = FakeTransport(RawResponse(200, HTML_ERROR_PAGE))

    with pytest.raises(OverpassError) as raised:
        await gateway(transport).run_query("[out:json];nodz(1);out;")

    message = str(raised.value)
    assert "Query timed out" in message
    assert 'parse error: Unknown type "nodz"' in message


async def test_html_error_under_400_survives_the_status_code() -> None:
    # The real server answers a parse error with 400 and this page, so a client that
    # raises on the status code first loses the only description of the failure.
    transport = FakeTransport(RawResponse(400, HTML_ERROR_PAGE))

    with pytest.raises(OverpassError) as raised:
        await gateway(transport).run_query("[out:json];nodz(1);out;")

    message = str(raised.value)
    assert "HTTP 400" in message
    assert 'parse error: Unknown type "nodz"' in message


async def test_dispatcher_shed_under_504_is_reported_with_its_page() -> None:
    page = "<html><body><p><strong>Error</strong>: Dispatcher_Client::request_read_and_idx::"
    page += "timeout. Probably the server is overcrowded.</p></body></html>"
    transport = FakeTransport(RawResponse(504, page))

    with pytest.raises(OverpassError) as raised:
        await gateway(transport).run_query("[out:json][timeout:900];node(1);out;")

    assert "HTTP 504" in str(raised.value)
    assert "Dispatcher_Client" in str(raised.value)


async def test_xml_under_200_is_not_parsed_as_json() -> None:
    transport = FakeTransport(RawResponse(200, OSM3S_XML))

    with pytest.raises(OverpassError):
        await gateway(transport).run_query("node(1);out;")


async def test_unreachable_upstream_names_the_url() -> None:
    transport = FakeTransport(error=OverpassTransportError("ConnectTimeout: timed out"))

    with pytest.raises(OverpassError) as raised:
        await gateway(transport).run_query("[out:json];node(1);out;")

    assert "could not reach http://overpass.test/api/interpreter" in str(raised.value)


def test_error_message_does_not_collapse_to_the_strong_tag() -> None:
    # A page whose only message is the <strong>-wrapped parse error: matching around
    # the markup instead of stripping it yields the useless string "Error".
    message = error_message(HTML_PARSE_ERROR_PAGE)

    assert message != "Error"
    assert 'line 3: parse error: Unknown type "nodz"' in message
    assert "<" not in message and ">" not in message


async def test_status_probe_reports_a_dead_upstream_instead_of_raising() -> None:
    transport = FakeTransport(error=OverpassTransportError("ConnectError: connection refused"))

    status = await gateway(transport).check_status()

    assert status.reachable is False
    assert "ConnectError" in (status.detail or "")
    assert status.url == "http://overpass.test/api/status"


async def test_status_probe_uses_the_short_timeout() -> None:
    transport = FakeTransport(RawResponse(200, "Connected as: 1\n"))

    status = await gateway(transport).check_status()

    assert status.reachable is True
    assert status.detail is None
    assert transport.calls[0][1] is None  # GET, no form body
    assert transport.calls[0][2] == 3


# --- the use case --------------------------------------------------------------


async def test_oversized_query_never_reaches_the_shared_instance() -> None:
    transport = FakeTransport(RawResponse(200, "{}"))

    with pytest.raises(OverpassQueryTooLongError) as raised:
        await service(transport, max_query_chars=10).run_query("x" * 11)

    assert "limit is 10" in str(raised.value)
    assert transport.calls == []


# --- the endpoints -------------------------------------------------------------


def test_query_endpoint_passes_the_document_through() -> None:
    transport = FakeTransport(RawResponse(200, '{"elements": [{"id": 1}]}'))
    with api(transport) as client:
        response = client.post(f"{BASE}/query", json={"query": "[out:json];node(1);out;"})

    assert response.status_code == 200
    assert response.json() == {"result": {"elements": [{"id": 1}]}, "remark": None}


def test_query_endpoint_surfaces_a_partial_result() -> None:
    body = '{"elements": [], "remark": "runtime error: Query timed out"}'
    transport = FakeTransport(RawResponse(200, body))
    with api(transport) as client:
        response = client.post(f"{BASE}/query", json={"query": "[out:json];node(1);out;"})

    assert response.status_code == 200
    assert response.json()["remark"] == "runtime error: Query timed out"


def test_query_endpoint_maps_an_upstream_failure_to_502() -> None:
    transport = FakeTransport(RawResponse(400, HTML_ERROR_PAGE))
    with api(transport) as client:
        response = client.post(f"{BASE}/query", json={"query": "[out:json];nodz(1);out;"})

    assert response.status_code == 502
    assert 'parse error: Unknown type "nodz"' in response.json()["detail"]


def test_query_endpoint_maps_an_oversized_query_to_400() -> None:
    transport = FakeTransport(RawResponse(200, "{}"))
    with api(transport, max_query_chars=10) as client:
        response = client.post(f"{BASE}/query", json={"query": "x" * 11})

    assert response.status_code == 400
    assert "limit is 10" in response.json()["detail"]
    assert transport.calls == []


def test_status_endpoint_stays_200_when_the_instance_is_down() -> None:
    transport = FakeTransport(error=OverpassTransportError("ConnectError: connection refused"))
    with api(transport) as client:
        response = client.get(f"{BASE}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["reachable"] is False
    assert "ConnectError" in body["detail"]
    assert body["url"] == "http://overpass.test/api/status"


def test_too_long_query_is_not_an_upstream_error() -> None:
    """A query rejected before it is sent must not look like an upstream failure.

    While it subclassed OverpassError, any `except OverpassError` catch-all turned a
    local 400 into a 502, and swapping the two except clauses in the route would have
    done the same with every test still green.
    """
    assert not issubclass(OverpassQueryTooLongError, OverpassError)
    assert issubclass(OverpassQueryTooLongError, ValueError)


def test_interpreter_and_status_urls_are_siblings() -> None:
    """Both derive from one base, so they cannot drift apart.

    The previous shape chopped "/interpreter" off the configured URL and fell back to
    that URL unchanged when the suffix was absent — so a trailing slash silently
    pointed the health probe at the interpreter, which answers 400 without a `data`
    parameter and reports a live upstream as dead forever.
    """
    for base in ("http://example.test/api", "http://example.test/api/"):
        settings = Settings(overpass_base_url=base)
        assert settings.overpass_interpreter_url == "http://example.test/api/interpreter"
        assert settings.overpass_status_url == "http://example.test/api/status"

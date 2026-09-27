"""Outbound adapter for the self-hosted Overpass API.

Overpass has five response shapes and only one is the obvious one, so nothing here
calls `.json()` on a response and hopes:

    success                          200, application/json
    parse or static error            400, an HTML error page
    dispatcher shed / server busy    504, an HTML error page
    query hit its own [timeout:]     200, valid JSON carrying a "remark" key
    output not JSON                  200, application/osm3s+xml

Two traps, both documented in `docs/overpass-api.md` and both pinned by tests:

1. **The HTML error pages arrive under 400 and 504.** A client that raises on the
   status code before reading the body throws away the only description of what went
   wrong. `HttpxTransport` therefore returns an HTTP error response as an ordinary
   `RawResponse` instead of raising, which makes that mistake unrepresentable here.
2. **The message straddles a tag boundary.** The word "Error" sits inside a
   `<strong>` and the part that matters follows the closing tag, so a regex anchored
   on "error" and stopping at "<" collapses every failure to the useless string
   "Error". `error_message` strips the markup first and keeps *every* error line: one
   page can carry a parse error and a runtime error at once.

The `remark` shape is passed through rather than raised on -- a partial answer is
sometimes what the caller wants -- but it is lifted out of the payload so callers
cannot miss it.

Transport is `urllib.request` on a worker thread rather than an async HTTP client:
the backend package has no HTTP client among its production dependencies, and adding
one would add a production dependency. The seam below keeps the behaviour
testable without one.
"""

import asyncio
import html
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

from app.domain.overpass import OverpassError, OverpassResult, OverpassStatus

MAX_MESSAGE_CHARS = 500

_BLOCK_BREAK_RE = re.compile(r"</(?:p|div|li|tr|h[1-6]|pre)\s*>|<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]*>")
_ERROR_LINE_RE = re.compile(r"error", re.IGNORECASE)


def html_to_lines(body: str) -> list[str]:
    """The visible text of an HTML error page, one entry per block element."""
    text = _BLOCK_BREAK_RE.sub("\n", body)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def error_message(body: str) -> str:
    """Every distinct error line on the page, in order.

    A single page can carry a parse error and a runtime error at once -- keeping only
    the first one throws away half of what went wrong.
    """
    lines = html_to_lines(body)
    seen: list[str] = []
    for line in lines:
        if _ERROR_LINE_RE.search(line) and line not in seen:
            seen.append(line)
    message = "; ".join(seen) if seen else " ".join(" ".join(lines).split())
    return message[:MAX_MESSAGE_CHARS] if message else body.strip()[:MAX_MESSAGE_CHARS]


def parse_body(body: str) -> OverpassResult:
    """Turn a 200 body into a result, rejecting the HTML and XML shapes."""
    if not body.lstrip().startswith("{"):
        raise OverpassError(error_message(body))
    try:
        payload: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as error:
        raise OverpassError(f"upstream returned unparseable JSON: {error}") from error
    remark = payload.get("remark")
    return OverpassResult(payload=payload, remark=str(remark) if remark is not None else None)


class OverpassTransportError(RuntimeError):
    """The request never produced an HTTP response at all."""


@dataclass(frozen=True, slots=True)
class RawResponse:
    status: int
    body: str


class OverpassTransport(Protocol):
    async def fetch(self, url: str, data: bytes | None, read_timeout: float) -> RawResponse: ...


class HttpxTransport:
    async def fetch(self, url: str, data: bytes | None, read_timeout: float) -> RawResponse:
        try:
            async with httpx.AsyncClient(timeout=read_timeout) as client:
                response = (
                    await client.post(
                        url,
                        content=data,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                    )
                    if data is not None
                    else await client.get(url)
                )
        except httpx.HTTPError as error:
            raise OverpassTransportError(f"{type(error).__name__}: {error}") from error
        # Returned rather than raised: 400 and 504 carry the HTML error page, and that
        # page body is the only description of what went wrong.
        return RawResponse(response.status_code, response.text)


class HttpOverpassGateway:
    def __init__(
        self,
        interpreter_url: str,
        status_url: str,
        query_timeout: float,
        health_timeout: float,
        transport: OverpassTransport | None = None,
    ) -> None:
        self._interpreter_url = interpreter_url
        self._status_url = status_url
        self._query_timeout = query_timeout
        self._health_timeout = health_timeout
        self._transport = transport or HttpxTransport()

    async def run_query(self, query: str) -> OverpassResult:
        payload = urlencode({"data": query}).encode("utf-8")
        try:
            # The read budget is the query's own claim plus headroom: Overpass answers
            # a query that overran its [timeout:] rather than dropping the connection.
            response = await self._transport.fetch(
                self._interpreter_url, payload, self._query_timeout + 30
            )
        except OverpassTransportError as error:
            raise OverpassError(f"could not reach {self._interpreter_url}: {error}") from error

        if response.status != 200:
            detail = error_message(response.body)
            raise OverpassError(
                f"upstream returned HTTP {response.status}" + (f": {detail}" if detail else "")
            )
        # Megabyte answers: decoding on the event loop would stall every other request.
        return await asyncio.to_thread(parse_body, response.body)

    async def check_status(self) -> OverpassStatus:
        """Short-timeout probe of `/api/status`. Never raises."""
        try:
            response = await self._transport.fetch(self._status_url, None, self._health_timeout)
        except OverpassTransportError as error:
            return OverpassStatus(url=self._status_url, reachable=False, detail=str(error))
        except Exception as error:  # a broken env var or bad URL is still not a 500
            return OverpassStatus(
                url=self._status_url,
                reachable=False,
                detail=f"{type(error).__name__}: {error}",
            )
        if response.status != 200:
            return OverpassStatus(
                url=self._status_url,
                reachable=False,
                detail=f"HTTP {response.status}",
            )
        return OverpassStatus(url=self._status_url, reachable=True, detail=None)

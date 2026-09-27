"""Throttled, retrying HTTP access to the KudaGo public API, plus response validation.

The one-request-per-second single-worker throttle and the five-retry budget are this
collector's own policy. https://docs.kudago.com/api/ documents no rate limit, so the
only honest assumption is a small self-imposed one; a 429 with `Retry-After` is the
server correcting us and is honoured as given.

A request never resolves to "no data". It resolves to a body, or to a `FetchResult`
whose `error` says why there is none, so a caller can tell an empty catalogue from an
unfinished one.

The transport is a callable so the collector can be exercised without network I/O.
"""

import email.utils
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

from tramflow_ml.external.kudago.records import API_ROOT

USER_AGENT = "tramflow-ml/0.1 (+https://docs.kudago.com/api/)"
DEFAULT_TIMEOUT_SECONDS = 30.0
MIN_INTERVAL_SECONDS = 1.0
MAX_ATTEMPTS = 6
BACKOFF_SECONDS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0)
MAX_RETRY_AFTER_SECONDS = 300.0
JITTER_FRACTION = 0.25

_API_PARTS = urlsplit(API_ROOT)
API_HOST = _API_PARTS.netloc
API_PATH_PREFIX = _API_PARTS.path


class KudaGoError(Exception):
    """Anything that stops the collector from trusting what it received."""


class TransportError(KudaGoError):
    """The request produced no HTTP response: timeout, DNS or connection failure."""


class ResponseError(KudaGoError):
    """A response arrived and is not usable data."""


@dataclass(frozen=True, slots=True)
class RawResponse:
    """What a transport returns: an HTTP response, however unwelcome its status."""

    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def header(self, name: str) -> str | None:
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return None


class Transport(Protocol):
    """Performs one HTTP GET. Raises `TransportError` when no response arrives."""

    def __call__(self, url: str, *, timeout: float) -> RawResponse: ...


@dataclass(frozen=True, slots=True)
class Attempt:
    """One request attempt, successful or not.

    `body` is present whenever bytes came back, including for a 404 or an HTML error
    page: those bytes are evidence and the caller persists them.
    """

    url: str
    attempt: int
    requested_at: datetime
    fetched_at: datetime
    status: int | None
    body: bytes | None
    content_type: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Every attempt made for one URL, in order."""

    url: str
    attempts: tuple[Attempt, ...]

    @property
    def final(self) -> Attempt:
        return self.attempts[-1]

    @property
    def ok(self) -> bool:
        return self.final.ok

    @property
    def body(self) -> bytes | None:
        return self.final.body if self.final.ok else None

    @property
    def error(self) -> str | None:
        return self.final.error

    @property
    def retries_exhausted(self) -> bool:
        """True when the last attempt was retryable and the budget ran out.

        The caller reports this as an incomplete collection. Treating it as an empty
        result would turn a broken network into a claim about the catalogue.
        """
        return not self.ok and _is_retryable(self.final.status)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = MAX_ATTEMPTS
    backoff_seconds: tuple[float, ...] = BACKOFF_SECONDS
    max_retry_after_seconds: float = MAX_RETRY_AFTER_SECONDS


@dataclass(frozen=True, slots=True)
class Page:
    """A validated paged envelope from a list endpoint."""

    count: int
    next_url: str | None
    results: tuple[Any, ...]
    payload: dict[str, Any]


def _is_retryable(status: int | None) -> bool:
    if status is None:
        return True
    return status == 429 or status >= 500


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _default_jitter(base: float) -> float:
    return random.uniform(0.0, base * JITTER_FRACTION)


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Delta seconds for a `Retry-After` header given as seconds or as an HTTP date."""
    if value is None:
        return None
    text = value.strip()
    try:
        return max(0.0, float(int(text)))
    except ValueError:
        pass
    try:
        deadline = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return max(0.0, (deadline - now).total_seconds())


class HttpClient:
    """Serial, throttled GETs with a bounded retry budget.

    `on_attempt` is called with every attempt before the next one is made, so the
    caller can persist the body and journal the attempt in order.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        policy: RetryPolicy | None = None,
        min_interval_seconds: float = MIN_INTERVAL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _utcnow,
        jitter: Callable[[float], float] = _default_jitter,
    ) -> None:
        self._transport = transport
        self._policy = policy or RetryPolicy()
        self._min_interval = min_interval_seconds
        self._timeout = timeout_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self._jitter = jitter
        self._last_request: float | None = None
        self.slept_seconds = 0.0

    def fetch(
        self, url: str, *, on_attempt: Callable[[Attempt], None] | None = None
    ) -> FetchResult:
        attempts: list[Attempt] = []
        for index in range(1, self._policy.max_attempts + 1):
            self._throttle()
            attempt, retry_after = self._attempt(url, index)
            attempts.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt)
            if attempt.ok:
                break
            if not _is_retryable(attempt.status) or index == self._policy.max_attempts:
                break
            self._pause(self._delay(index, retry_after))
        return FetchResult(url=url, attempts=tuple(attempts))

    def _attempt(self, url: str, index: int) -> tuple[Attempt, float | None]:
        requested_at = self._now()
        try:
            response = self._transport(url, timeout=self._timeout)
        except TransportError as error:
            fetched_at = self._now()
            return (
                Attempt(
                    url=url,
                    attempt=index,
                    requested_at=requested_at,
                    fetched_at=fetched_at,
                    status=None,
                    body=None,
                    content_type=None,
                    error=f"transport_error: {error}",
                ),
                None,
            )
        fetched_at = self._now()
        ok = 200 <= response.status < 300
        retry_after = parse_retry_after(response.header("Retry-After"), fetched_at)
        return (
            Attempt(
                url=url,
                attempt=index,
                requested_at=requested_at,
                fetched_at=fetched_at,
                status=response.status,
                body=response.body,
                content_type=response.header("Content-Type"),
                error=None if ok else f"http_{response.status}",
            ),
            retry_after,
        )

    def _delay(self, index: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self._policy.max_retry_after_seconds)
        backoff = self._policy.backoff_seconds
        base = backoff[min(index, len(backoff)) - 1] if backoff else 0.0
        return base + self._jitter(base)

    def _pause(self, seconds: float) -> None:
        if seconds > 0:
            self.slept_seconds += seconds
            self._sleep(seconds)

    def _throttle(self) -> None:
        if self._last_request is not None:
            waiting = self._min_interval - (self._monotonic() - self._last_request)
            self._pause(waiting)
        self._last_request = self._monotonic()


class UrllibTransport:
    """The live transport. Stdlib only: the collector adds no HTTP dependency."""

    def __init__(self, *, user_agent: str = USER_AGENT) -> None:
        self._user_agent = user_agent

    def __call__(self, url: str, *, timeout: float) -> RawResponse:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return RawResponse(
                    status=int(response.status),
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as error:
            body = error.read()
            headers = dict(error.headers.items()) if error.headers is not None else {}
            return RawResponse(status=int(error.code), body=body, headers=headers)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TransportError(str(error)) from error


def looks_like_json(body: bytes) -> bool:
    stripped = body.lstrip()
    return stripped[:1] in (b"{", b"[")


def body_extension(body: bytes) -> str:
    """`.json` only for bytes that actually start a JSON document."""
    return ".json" if looks_like_json(body) else ".html"


def parse_json_object(body: bytes) -> dict[str, Any]:
    payload = _parse_json(body)
    if not isinstance(payload, dict):
        raise ResponseError(f"body is a JSON {type(payload).__name__}, not an object")
    return payload


def parse_json_array(body: bytes) -> list[Any]:
    payload = _parse_json(body)
    if not isinstance(payload, list):
        raise ResponseError(f"body is a JSON {type(payload).__name__}, not an array")
    return payload


def _parse_json(body: bytes) -> Any:
    if not looks_like_json(body):
        preview = body[:40].decode("utf-8", "replace")
        raise ResponseError(f"body is not JSON: starts with {preview!r}")
    try:
        return json.loads(body)
    except ValueError as error:
        raise ResponseError(f"body is not valid JSON: {error}") from error


def parse_page(body: bytes) -> Page:
    """Validate a paged envelope before any of it is treated as catalogue content."""
    payload = parse_json_object(body)
    count = payload.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ResponseError(f"count is not a non-negative integer: {count!r}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ResponseError(f"results is not an array: {type(results).__name__}")
    next_url = payload.get("next")
    if next_url is not None and not _is_url(next_url):
        raise ResponseError(f"next is neither a URL nor null: {next_url!r}")
    return Page(
        count=count,
        next_url=next_url,
        results=tuple(results),
        payload=payload,
    )


def _is_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = urlsplit(value)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def is_followable(url: str) -> bool:
    """A `next` link is followed only if it stays on the documented API root."""
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.netloc == API_HOST
        and parts.path.startswith(API_PATH_PREFIX)
    )

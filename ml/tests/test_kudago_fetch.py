"""Offline tests for the KudaGo fetch half. No test performs network I/O.

The transport is injected, so every HTTP behaviour the collector claims to handle —
throttling, retries, envelope validation, paging, resume — is exercised against
recorded bodies instead of the live API.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tramflow_ml.external.kudago.fetch import (
    FETCH_CHECKPOINT_NAME,
    FetchConfig,
    FetchSummary,
    KudaGoError,
    RequestJournal,
    collect,
    load_fetch_checkpoint,
    month_window,
)
from tramflow_ml.external.kudago.http import (
    Attempt,
    HttpClient,
    RawResponse,
    ResponseError,
    RetryPolicy,
    TransportError,
    is_followable,
    parse_page,
    parse_retry_after,
)
from tramflow_ml.external.kudago.records import RAW_DIR, REQUESTS_NAME

LOCATIONS = [
    {"slug": "msk", "name": "Москва"},
    {"slug": "spb", "name": "Санкт-Петербург"},
]
CATEGORIES = [
    {"id": 1, "slug": "concert", "name": "Концерты"},
    {"id": 9, "slug": "cinema", "name": "Кинопоказы"},
]
EVENT_DETAIL: dict[str, Any] = {
    "id": 3606,
    "publication_date": 1776357017,
    "title": "выставка «Механика чуда»",
    "slug": "vystavka-mehanika-chuda",
    "site_url": "https://kudago.com/msk/event/vystavka-mehanika-chuda/",
    "description": "Выставка старинных автоматов.",
    "body_text": "Экспозиция работает ежедневно.",
    "location": {"slug": "msk"},
    "categories": ["exhibition"],
    "tags": ["выставка"],
    "price": "300 рублей",
    "is_free": False,
    "age_restriction": "6+",
    "dates": [{"start": 1735689600, "end": 1735707600}],
    "place": {"id": 4242, "title": "Музей техники"},
}
PLACE_DETAIL: dict[str, Any] = {
    "id": 4242,
    "title": "Музей техники",
    "slug": "muzej-tehniki",
    "address": "ул. Образцовая, 1",
    "coords": {"lat": 55.75, "lon": 37.61},
    "subway": "Тульская",
    "timetable": "ежедневно с 10:00 до 20:00",
    "is_closed": False,
    "is_stub": False,
    "location": {"slug": "msk"},
    "site_url": "https://kudago.com/msk/place/muzej-tehniki/",
}
HTML_BODY = b"<!DOCTYPE html><html><head><title>502</title></head><body>oops</body></html>"

Handler = Callable[[str, int], RawResponse | BaseException]


class FakeTransport:
    """Answers each URL from a handler that also sees how many times it was asked."""

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self.calls: list[str] = []

    def __call__(self, url: str, *, timeout: float) -> RawResponse:
        self.calls.append(url)
        answer = self._handler(url, self.calls.count(url))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def calls_to(self, fragment: str) -> list[str]:
        return [url for url in self.calls if fragment in url]


class FakeClock:
    def __init__(self) -> None:
        self.monotonic_value = 0.0
        self.slept: list[float] = []
        self._wall = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.monotonic_value += seconds

    def monotonic(self) -> float:
        self.monotonic_value += 0.001
        return self.monotonic_value

    def now(self) -> datetime:
        self._wall += timedelta(seconds=1)
        return self._wall


def json_response(payload: Any, status: int = 200) -> RawResponse:
    return RawResponse(
        status=status,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def page(
    ids: list[int],
    *,
    count: int,
    next_url: str | None,
    place_ids: dict[int, int] | None = None,
) -> RawResponse:
    """An inventory page in the instruction's shape: id, dates, place, site_url."""
    places = place_ids or {}
    results = []
    for value in ids:
        entry: dict[str, Any] = {
            "id": value,
            "dates": [{"start_date": "2025-01-04", "start": 1735948800, "end": 1735966800}],
            "site_url": f"https://kudago.com/msk/event/{value}/",
        }
        if value in places:
            entry["place"] = {"id": places[value]}
        results.append(entry)
    return json_response({"count": count, "next": next_url, "previous": None, "results": results})


def client_for(transport: FakeTransport, clock: FakeClock) -> HttpClient:
    return HttpClient(
        transport,
        min_interval_seconds=1.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=clock.now,
        jitter=lambda base: 0.0,
    )


def list_only(out_dir: Path, **overrides: Any) -> FetchConfig:
    defaults: dict[str, Any] = {
        "months": ("2025-01",),
        "page_size": 2,
        "max_id_passes": 1,
        "fetch_details": False,
        "fetch_places": False,
    }
    defaults.update(overrides)
    return FetchConfig(out_dir=out_dir, **defaults)


def dictionaries_only(url: str) -> RawResponse | None:
    if "locations/" in url:
        return json_response(LOCATIONS)
    if "event-categories/" in url:
        return json_response(CATEGORIES)
    return None


def journal_lines(out_dir: Path) -> list[dict[str, Any]]:
    text = (out_dir / REQUESTS_NAME).read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def run(out_dir: Path, handler: Handler, config: FetchConfig) -> tuple[FetchSummary, FakeTransport]:
    transport = FakeTransport(handler)
    summary = collect(config, client=client_for(transport, FakeClock()), run_id="test-run")
    return summary, transport


def test_month_window_widens_by_one_day() -> None:
    assert month_window("2025-01", 1) == ("2024-12-31T00:00:00", "2025-02-02T00:00:00")
    assert month_window("2025-09", 1) == ("2025-08-31T00:00:00", "2025-10-02T00:00:00")
    assert month_window("2025-12", 0) == ("2025-12-01T00:00:00", "2026-01-01T00:00:00")


def test_retry_after_is_honoured_on_429() -> None:
    clock = FakeClock()
    transport = FakeTransport(
        lambda url, seen: RawResponse(429, b'{"detail":"slow down"}', {"Retry-After": "7"})
        if seen == 1
        else page([1], count=1, next_url=None)
    )
    result = client_for(transport, clock).fetch("https://kudago.com/public-api/v1.4/events/")

    assert [attempt.status for attempt in result.attempts] == [429, 200]
    assert result.ok
    assert 7.0 in clock.slept


def test_retry_after_accepts_an_http_date() -> None:
    now = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    assert parse_retry_after("Sun, 27 Sep 2026 12:00:30 GMT", now) == pytest.approx(30.0)
    assert parse_retry_after("nonsense", now) is None
    assert parse_retry_after(None, now) is None


def test_server_error_succeeds_on_retry_with_backoff() -> None:
    clock = FakeClock()
    transport = FakeTransport(
        lambda url, seen: RawResponse(500, b"upstream failed", {})
        if seen == 1
        else page([1, 2], count=2, next_url=None)
    )
    result = client_for(transport, clock).fetch("https://kudago.com/public-api/v1.4/events/")

    assert result.ok
    assert len(result.attempts) == 2
    assert clock.slept[0] == 2.0


def test_backoff_doubles_until_the_budget_is_spent() -> None:
    clock = FakeClock()
    transport = FakeTransport(lambda url, seen: TransportError("timed out"))
    result = client_for(transport, clock).fetch("https://kudago.com/public-api/v1.4/events/")

    assert len(result.attempts) == 6
    assert result.retries_exhausted
    assert [value for value in clock.slept if value >= 2.0] == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_client_throttles_to_one_request_per_second() -> None:
    clock = FakeClock()
    transport = FakeTransport(lambda url, seen: page([1], count=1, next_url=None))
    client = client_for(transport, clock)
    client.fetch("https://kudago.com/public-api/v1.4/locations/")
    client.fetch("https://kudago.com/public-api/v1.4/event-categories/")

    assert clock.slept and clock.slept[0] == pytest.approx(1.0, abs=0.01)


def test_page_envelope_is_validated() -> None:
    with pytest.raises(ResponseError, match="not JSON"):
        parse_page(HTML_BODY)
    with pytest.raises(ResponseError, match="count"):
        parse_page(b'{"count":-1,"next":null,"results":[]}')
    with pytest.raises(ResponseError, match="results"):
        parse_page(b'{"count":1,"next":null,"results":{}}')
    with pytest.raises(ResponseError, match="next"):
        parse_page(b'{"count":1,"next":"later","results":[]}')
    envelope = parse_page(b'{"count":1,"next":null,"results":[{"id":3606}]}')
    assert envelope.count == 1 and envelope.next_url is None


def test_followable_only_under_the_documented_api_root() -> None:
    assert is_followable("https://kudago.com/public-api/v1.4/events/?page=2")
    assert not is_followable("http://kudago.com/public-api/v1.4/events/?page=2")
    assert not is_followable("https://evil.example/public-api/v1.4/events/?page=2")
    assert not is_followable("https://kudago.com/public-api/v1.3/events/?page=2")


def test_retry_exhaustion_is_incomplete_not_empty(tmp_path: Path) -> None:
    second = "https://kudago.com/public-api/v1.4/events/?page=2"

    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        if "page=2" in url:
            return RawResponse(503, b"upstream unavailable", {})
        return page([11, 12], count=4, next_url=second)

    summary, transport = run(tmp_path, handler, list_only(tmp_path))

    month = summary.months[0]
    assert month.event_ids == (11, 12)
    assert month.pages_complete is False
    assert month.page_counts == (4,)
    assert summary.status == "partial"
    assert len(summary.failures) == 1
    failure = summary.failures[0]
    assert failure.kind == "list_page"
    assert "retries exhausted" in failure.error
    assert len(failure.request_ids) == 6
    assert summary.unresolved_request_ids == failure.request_ids
    assert len(transport.calls_to("page=2")) == 6


def test_html_under_http_200_is_rejected_and_kept_as_evidence(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        return RawResponse(200, HTML_BODY, {"Content-Type": "text/html"})

    summary, _ = run(tmp_path, handler, list_only(tmp_path))

    assert summary.status == "partial"
    assert summary.months[0].event_ids == ()
    assert "not JSON" in summary.failures[0].error
    saved = [record for record in journal_lines(tmp_path) if record["http_status"] == 200]
    html = [record for record in saved if str(record["raw_path"]).endswith(".html")]
    assert len(html) == 1
    assert (tmp_path / str(html[0]["raw_path"])).read_bytes() == HTML_BODY
    assert html[0]["error"] is None
    assert html[0]["count"] is None


def test_next_loop_is_refused(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        if "page=2" in url:
            return page([21], count=3, next_url=url)
        return page([11, 12], count=3, next_url="https://kudago.com/public-api/v1.4/events/?page=2")

    summary, transport = run(tmp_path, handler, list_only(tmp_path))

    assert summary.months[0].event_ids == (11, 12, 21)
    assert summary.months[0].pages_complete is False
    assert "revisits a URL" in summary.failures[0].error
    assert len(transport.calls_to("page=2")) == 1


def test_next_off_host_is_refused(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        return page([11], count=2, next_url="https://evil.example/public-api/v1.4/events/?page=2")

    summary, transport = run(tmp_path, handler, list_only(tmp_path))

    assert summary.months[0].event_ids == (11,)
    assert summary.months[0].pages_complete is False
    assert "leaves the API root" in summary.failures[0].error
    assert transport.calls_to("evil.example") == []


def test_resume_does_not_refetch_a_checkpointed_page(tmp_path: Path) -> None:
    second = "https://kudago.com/public-api/v1.4/events/?page=2"

    def failing(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        if "page=2" in url:
            return TransportError("connection reset")
        return page([11, 12], count=4, next_url=second)

    first_summary, _ = run(tmp_path, failing, list_only(tmp_path))
    assert first_summary.status == "partial"
    first_lines = journal_lines(tmp_path)

    def recovered(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        if "page=2" in url:
            return page([13, 14], count=4, next_url=None)
        raise AssertionError(f"page already checkpointed: {url}")

    second_summary, transport = run(tmp_path, recovered, list_only(tmp_path))

    assert transport.calls_to("locations/") == []
    assert second_summary.months[0].event_ids == (11, 12, 13, 14)
    assert second_summary.months[0].pages_complete is True
    resumed = journal_lines(tmp_path)
    assert resumed[: len(first_lines)] == first_lines
    checkpoint = load_fetch_checkpoint(tmp_path / FETCH_CHECKPOINT_NAME)
    assert checkpoint["run_id"] == "test-run"
    assert checkpoint["sequence"] == len(resumed)


def test_id_drift_reports_unstable_after_three_passes(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        return page([11, 20 + seen], count=2, next_url=None)

    summary, transport = run(tmp_path, handler, list_only(tmp_path, max_id_passes=3))

    assert summary.months[0].passes == 3
    assert summary.months[0].stability == "unstable"
    assert summary.status == "unstable"
    assert len(transport.calls_to("actual_since")) == 3


def test_stable_when_two_consecutive_passes_agree(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        return page([11, 12], count=2, next_url=None)

    summary, transport = run(tmp_path, handler, list_only(tmp_path, max_id_passes=3))

    assert summary.months[0].passes == 2
    assert summary.months[0].stability == "stable"
    assert summary.status == "complete"
    assert len(transport.calls_to("actual_since")) == 2


def test_full_shape_writes_dictionaries_details_and_places(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        if "/events/3606/" in url:
            return json_response(EVENT_DETAIL)
        if "/events/4040/" in url:
            return RawResponse(404, b'{"detail":"Not found."}', {})
        if "/places/4242/" in url:
            return json_response(PLACE_DETAIL)
        return page([3606, 4040], count=2, next_url=None)

    summary, transport = run(
        tmp_path,
        handler,
        FetchConfig(out_dir=tmp_path, months=("2025-01",), page_size=2, max_id_passes=2),
    )

    assert summary.dictionaries == ("locations", "event-categories")
    assert json.loads((tmp_path / "dictionaries" / "locations.json").read_bytes()) == LOCATIONS
    assert summary.event_count == 1
    assert summary.place_count == 1
    assert summary.status == "partial"
    assert [failure.object_id for failure in summary.failures] == ["4040"]
    assert summary.failures[0].error == "http_404"
    assert len(transport.calls_to("/events/4040/")) == 1

    detail_expected = "expand=dates%2Cplace"
    assert any(detail_expected in url for url in transport.calls_to("/events/3606/"))
    assert any("text_format=plain" in url for url in transport.calls_to("/places/4242/"))


def test_inventory_page_carries_dates_and_discovers_the_place(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        if "/events/3606/" in url:
            return RawResponse(500, b"upstream failed", {})
        if "/places/4242/" in url:
            return json_response(PLACE_DETAIL)
        return page([3606], count=1, next_url=None, place_ids={3606: 4242})

    summary, transport = run(
        tmp_path,
        handler,
        FetchConfig(out_dir=tmp_path, months=("2025-01",), page_size=2, max_id_passes=2),
    )

    inventory_calls = transport.calls_to("actual_since")
    assert any("expand=dates" in url for url in inventory_calls)
    assert any("fields=id%2Cdates%2Cplace%2Csite_url" in url for url in inventory_calls)
    assert summary.event_count == 0
    assert summary.place_count == 1
    assert [failure.object_id for failure in summary.failures] == ["3606"]
    inventory = json.loads((tmp_path / str(journal_lines(tmp_path)[2]["raw_path"])).read_bytes())
    assert inventory["results"][0]["dates"][0]["start_date"] == "2025-01-04"


def test_place_request_asks_for_the_whole_object(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        if "/places/4242/" in url:
            return json_response(PLACE_DETAIL)
        return page([3606], count=1, next_url=None, place_ids={3606: 4242})

    _, transport = run(
        tmp_path,
        handler,
        FetchConfig(
            out_dir=tmp_path, months=("2025-01",), page_size=2, max_id_passes=2, fetch_details=False
        ),
    )

    requested = transport.calls_to("/places/4242/")
    assert requested and all("fields=" not in url for url in requested)
    assert all("text_format=plain" in url for url in requested)


def test_unlimited_resume_completes_what_a_capped_run_left(tmp_path: Path) -> None:
    details = {3606: EVENT_DETAIL, 4040: {**EVENT_DETAIL, "id": 4040, "place": {"id": 4343}}}
    places = {4242: PLACE_DETAIL, 4343: {**PLACE_DETAIL, "id": 4343}}

    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        for event_id, payload in details.items():
            if f"/events/{event_id}/" in url:
                return json_response(payload)
        for place_id, payload in places.items():
            if f"/places/{place_id}/" in url:
                return json_response(payload)
        return page([3606, 4040], count=2, next_url=None)

    capped = FetchConfig(
        out_dir=tmp_path,
        months=("2025-01",),
        page_size=2,
        max_id_passes=2,
        detail_limit=1,
        place_limit=1,
    )
    first, _ = run(tmp_path, handler, capped)
    assert first.status == "partial"
    assert (first.event_count, first.place_count) == (1, 1)

    unlimited = FetchConfig(out_dir=tmp_path, months=("2025-01",), page_size=2, max_id_passes=2)
    second, transport = run(tmp_path, handler, unlimited)

    assert (second.event_count, second.place_count) == (2, 2)
    assert second.failures == ()
    assert second.status == "complete"
    assert transport.calls_to("/events/3606/") == []
    assert len(transport.calls_to("/events/4040/")) == 1


def test_every_journal_line_points_at_a_saved_body(tmp_path: Path) -> None:
    def handler(url: str, seen: int) -> RawResponse | BaseException:
        dictionary = dictionaries_only(url)
        if dictionary is not None:
            return dictionary
        return page([11], count=1, next_url=None)

    summary, _ = run(tmp_path, handler, list_only(tmp_path))

    records = journal_lines(tmp_path)
    assert len(records) == summary.request_count == 3
    paths = [str(record["raw_path"]) for record in records]
    assert len(set(paths)) == len(paths)
    for record in records:
        saved = tmp_path / str(record["raw_path"])
        assert saved.is_file()
        assert record["run_id"] == "test-run"
        assert record["attempt"] == 1
        assert record["body_sha256"] is not None
    assert len(list((tmp_path / RAW_DIR).iterdir())) == len(records)


def test_journal_refuses_to_rewrite_an_existing_raw_body(tmp_path: Path) -> None:
    journal = RequestJournal(tmp_path, "test-run")
    url = "https://kudago.com/public-api/v1.4/locations/?lang=ru"
    existing = tmp_path / RAW_DIR / f"{journal.request_id(1, url)}.json"
    existing.write_bytes(b"[]")
    clock = FakeClock()
    transport = FakeTransport(lambda called, seen: json_response(LOCATIONS))
    with pytest.raises(Exception, match="already exists"):
        client_for(transport, clock).fetch(url, on_attempt=journal.record)
    journal.close()
    assert existing.read_bytes() == b"[]"


def test_existing_journal_without_a_checkpoint_is_refused(tmp_path: Path) -> None:
    (tmp_path / REQUESTS_NAME).write_bytes(b"{}\n")
    transport = FakeTransport(lambda url, seen: json_response(LOCATIONS))
    with pytest.raises(Exception, match="without"):
        collect(list_only(tmp_path), client=client_for(transport, FakeClock()))


def test_retry_policy_can_be_shortened() -> None:
    clock = FakeClock()
    transport = FakeTransport(lambda url, seen: RawResponse(500, b"nope", {}))
    client = HttpClient(
        transport,
        policy=RetryPolicy(max_attempts=2, backoff_seconds=(1.0,)),
        min_interval_seconds=0.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=clock.now,
        jitter=lambda base: 0.0,
    )
    result = client.fetch("https://kudago.com/public-api/v1.4/events/")

    assert len(result.attempts) == 2
    assert result.body is None
    assert result.error == "http_500"


def _attempt(url: str, body: bytes) -> Attempt:
    moment = datetime(2026, 9, 27, 12, tzinfo=UTC)
    return Attempt(
        url=url,
        attempt=1,
        requested_at=moment,
        fetched_at=moment,
        status=200,
        body=body,
        content_type="application/json",
        error=None,
    )


def test_replaying_an_attempt_after_a_rolled_back_checkpoint_keeps_the_body(tmp_path):
    """A resume re-issues the same request id; identical bytes are that replay."""
    body = b'{"count": 0, "next": null, "results": []}'
    first = RequestJournal(tmp_path, "run-1")
    first.record(_attempt("https://example.test/a/", body))
    first.close()

    second = RequestJournal(tmp_path, "run-1")
    record = second.record(_attempt("https://example.test/a/", body))
    second.close()

    assert record["raw_path"] is not None
    assert (tmp_path / record["raw_path"]).read_bytes() == body


def test_a_different_body_under_the_same_request_id_stops_the_run(tmp_path):
    first = RequestJournal(tmp_path, "run-1")
    first.record(_attempt("https://example.test/a/", b'{"count": 0}'))
    first.close()

    second = RequestJournal(tmp_path, "run-1")
    with pytest.raises(KudaGoError, match="different content"):
        second.record(_attempt("https://example.test/a/", b'{"count": 1}'))
    second.close()

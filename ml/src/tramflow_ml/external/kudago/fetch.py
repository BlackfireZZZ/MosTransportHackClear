"""The fetch half of the KudaGo collector: live API to `raw/` plus `requests.jsonl`.

This module decides *what* to request and *what to keep as evidence*. It does not
interpret content: the only fields it reads out of a payload are the ones it needs to
issue the next request (`next`, `results[].id`, `place.id`). Everything else is the
normaliser's business and reaches it as bytes on disk.

The inventory request asks for `dates` and `place` because the instruction's list
shape does: a list page is the record of a card's repeat structure, so the repeat
history survives in `raw/` even for cards whose detail request never succeeds.

Output layout under `config.out_dir`:

    raw/<request_id>.json      one saved body per attempt (`.html` when not JSON)
    requests.jsonl             one `RequestRecord` per attempt, body written first
    dictionaries/*.json        the two reference lists, validated
    fetch_checkpoint.json      resume state, committed after every page and object

The `actual_since`/`actual_until` window is a narrowing of the fetch, not evidence of
a complete date intersection: it is recorded in `MonthReport` as the window that was
requested and nowhere as a coverage claim.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from urllib.parse import urlencode, urljoin
from uuid import uuid4

from tramflow_ml.external.kudago.http import (
    DEFAULT_TIMEOUT_SECONDS,
    Attempt,
    HttpClient,
    KudaGoError,
    Page,
    ResponseError,
    UrllibTransport,
    body_extension,
    is_followable,
    parse_json_array,
    parse_json_object,
    parse_page,
)
from tramflow_ml.external.kudago.records import (
    API_ROOT,
    DICTIONARIES_DIR,
    MOSCOW,
    RAW_DIR,
    REQUESTS_NAME,
    RequestRecord,
    RunStatus,
)
from tramflow_ml.ingestion.checkpoint import OutputWriter, write_atomic
from tramflow_ml.ingestion.records import FileInventory, encode

FETCH_CHECKPOINT_NAME = "fetch_checkpoint.json"
FETCH_CHECKPOINT_SCHEMA = "kudago-fetch-checkpoint.v2"

LOCATIONS_PATH = "locations/"
EVENT_CATEGORIES_PATH = "event-categories/"
EVENTS_PATH = "events/"
PLACES_PATH = "places/"

DICTIONARY_PATHS: tuple[tuple[str, str], ...] = (
    ("locations", LOCATIONS_PATH),
    ("event-categories", EVENT_CATEGORIES_PATH),
)

# The two request shapes are given verbatim by the team instruction. The inventory
# shape carries the repeat structure and the place reference, so a list page is a
# record of a card's dates in its own right, not just a source of ids.
LIST_FIELDS: tuple[str, ...] = ("id", "dates", "place", "site_url")
LIST_EXPAND = "dates"
DETAIL_FIELDS: tuple[str, ...] = (
    "id",
    "publication_date",
    "title",
    "slug",
    "dates",
    "place",
    "location",
    "categories",
    "tags",
    "description",
    "body_text",
    "price",
    "is_free",
    "age_restriction",
    "site_url",
)
DETAIL_EXPAND = "dates,place"
TEXT_FORMAT = "plain"

FailureKind = Literal["dictionary", "list_page", "event", "place"]
Stability = Literal["stable", "unstable", "unknown"]


class PageState(TypedDict):
    ids: list[int]
    place_ids: list[str]
    next: str | None
    count: int
    request_id: str


class DetailState(TypedDict):
    request_id: str
    place_id: str | None


class FailureState(TypedDict):
    kind: str
    object_id: str | None
    url: str
    request_ids: list[str]
    error: str


class FetchCheckpoint(TypedDict):
    schema_version: str
    run_id: str
    sequence: int
    journal: FileInventory
    dictionaries: dict[str, str]
    pages: dict[str, PageState]
    details: dict[str, DetailState]
    places: dict[str, str]
    failures: list[FailureState]


@dataclass(frozen=True, slots=True)
class FetchConfig:
    """Pilot parameters. The three limits exist to keep a smoke run tiny.

    Any limit that actually bites makes the run `partial`: a capped run has not seen
    the catalogue and must never be reported as if it had.

    `detail_limit` and `place_limit` count objects fetched *by this run*, not objects
    already in the output directory: a resumed run skips what the checkpoint holds and
    then fetches up to the limit again, so that resuming makes progress. `page_limit`
    counts pages walked per id pass, cached pages included.
    """

    out_dir: Path
    months: tuple[str, ...]
    location: str = "msk"
    lang: str = "ru"
    order_by: str = "id"
    page_size: int = 100
    overlap_days: int = 1
    max_id_passes: int = 3
    page_limit: int | None = None
    detail_limit: int | None = None
    place_limit: int | None = None
    fetch_details: bool = True
    fetch_places: bool = True


@dataclass(frozen=True, slots=True)
class FailedFetch:
    """A request the run could not resolve. Quarantine input for the normaliser.

    The collector deliberately does not write `quarantine.jsonl`; that file belongs to
    the normaliser, which is the half that knows what a rejected object means.
    """

    kind: FailureKind
    object_id: str | None
    url: str
    request_ids: tuple[str, ...]
    error: str


@dataclass(frozen=True, slots=True)
class MonthReport:
    """What the run observed for one month window.

    `stability` compares consecutive full id passes over the same window. It measures
    the catalogue as served during this run and says nothing about cards deleted
    before it started. `pages_complete` false means pages are missing, so the id set
    is a floor, not the month's contents.
    """

    month: str
    actual_since: str
    actual_until: str
    passes: int
    stability: Stability
    pages_complete: bool
    page_counts: tuple[int, ...]
    event_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FetchSummary:
    run_id: str
    out_dir: str
    status: RunStatus
    months: tuple[MonthReport, ...]
    dictionaries: tuple[str, ...]
    event_count: int
    place_count: int
    request_count: int
    failures: tuple[FailedFetch, ...]
    unresolved_request_ids: tuple[str, ...]


def month_window(month: str, overlap_days: int) -> tuple[str, str]:
    """Moscow-local `actual_since`/`actual_until` for `YYYY-MM`, widened by the overlap."""
    try:
        first = datetime.strptime(month, "%Y-%m").replace(tzinfo=MOSCOW)
    except ValueError as error:
        raise KudaGoError(f"month must be YYYY-MM, got {month!r}") from error
    if len(month) != 7:
        raise KudaGoError(f"month must be YYYY-MM, got {month!r}")
    if first.month == 12:
        following = first.replace(year=first.year + 1, month=1)
    else:
        following = first.replace(month=first.month + 1)
    overlap = timedelta(days=overlap_days)
    since = first - overlap
    until = following + overlap
    return since.strftime("%Y-%m-%dT%H:%M:%S"), until.strftime("%Y-%m-%dT%H:%M:%S")


def build_url(path: str, params: Mapping[str, str | int]) -> str:
    query = urlencode(sorted((key, str(value)) for key, value in params.items()))
    return f"{urljoin(API_ROOT, path)}?{query}"


class RequestJournal:
    """`raw/` plus `requests.jsonl`, in that order.

    The body is on disk before the line that names it, so a journal entry never points
    at a file that does not exist. A request id is a function of the run, the sequence
    and the url, so a resume that rolls the journal back to the last checkpoint replays
    the same ids: an existing raw file holding the same bytes is that replay and is left
    alone. Different bytes under the same id mean the run state is wrong, and the run
    stops rather than overwrite evidence.
    """

    def __init__(
        self,
        out_dir: Path,
        run_id: str,
        *,
        resume: FileInventory | None = None,
        sequence: int = 0,
    ) -> None:
        self.out_dir = out_dir
        self.run_id = run_id
        self.raw_dir = out_dir / RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = sequence
        self._writer = OutputWriter(out_dir / REQUESTS_NAME, resume)
        self.records = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    def request_id(self, sequence: int, url: str) -> str:
        digest = hashlib.sha256(f"{self.run_id}:{sequence}:{url}".encode())
        return digest.hexdigest()[:32]

    def record(self, attempt: Attempt) -> RequestRecord:
        self._sequence += 1
        request_id = self.request_id(self._sequence, attempt.url)
        raw_path: str | None = None
        body_sha256: str | None = None
        count: int | None = None
        next_url: str | None = None
        if attempt.body is not None:
            name = f"{request_id}{body_extension(attempt.body)}"
            target = self.raw_dir / name
            if target.exists() and target.read_bytes() != attempt.body:
                raise KudaGoError(
                    f"raw body already exists with different content: {target}"
                )
            write_atomic(target, attempt.body)
            raw_path = f"{RAW_DIR}/{name}"
            body_sha256 = hashlib.sha256(attempt.body).hexdigest()
            count, next_url = _envelope_hints(attempt.body)
        record: RequestRecord = {
            "request_id": request_id,
            "run_id": self.run_id,
            "url": attempt.url,
            "requested_at": attempt.requested_at.isoformat(),
            "fetched_at": attempt.fetched_at.isoformat(),
            "http_status": attempt.status,
            "attempt": attempt.attempt,
            "raw_path": raw_path,
            "body_sha256": body_sha256,
            "error": attempt.error,
            "count": count,
            "next_url": next_url,
        }
        self._writer.write(encode(record))
        self.records += 1
        return record

    def flush(self) -> FileInventory:
        self._writer.flush()
        return self._writer.inventory()

    def close(self) -> None:
        self._writer.close()


def _envelope_hints(body: bytes) -> tuple[int | None, str | None]:
    try:
        page = parse_page(body)
    except ResponseError:
        return None, None
    return page.count, page.next_url


def new_checkpoint(run_id: str) -> FetchCheckpoint:
    return {
        "schema_version": FETCH_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "sequence": 0,
        "journal": {"sha256": hashlib.sha256().hexdigest(), "bytes": 0},
        "dictionaries": {},
        "pages": {},
        "details": {},
        "places": {},
        "failures": [],
    }


def load_fetch_checkpoint(path: Path) -> FetchCheckpoint:
    try:
        payload = json.loads(path.read_bytes())
    except ValueError as error:
        raise KudaGoError(f"{path.name} is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != FETCH_CHECKPOINT_SCHEMA:
        raise KudaGoError(f"{path.name} is not a {FETCH_CHECKPOINT_SCHEMA} checkpoint")
    expected: dict[str, type] = {
        "run_id": str,
        "sequence": int,
        "journal": dict,
        "dictionaries": dict,
        "pages": dict,
        "details": dict,
        "places": dict,
        "failures": list,
    }
    malformed = sorted(
        key
        for key, kind in expected.items()
        if not isinstance(payload.get(key), kind) or isinstance(payload.get(key), bool)
    )
    journal = payload.get("journal")
    if isinstance(journal, dict) and not (
        isinstance(journal.get("sha256"), str) and isinstance(journal.get("bytes"), int)
    ):
        malformed.append("journal")
    if malformed:
        raise KudaGoError(f"{path.name} is missing or mistyped: {sorted(set(malformed))}")
    return cast(FetchCheckpoint, payload)


@dataclass(frozen=True, slots=True)
class _PassResult:
    ids: tuple[int, ...]
    counts: tuple[int, ...]
    complete: bool


class Collector:
    """One fetch run over one output directory."""

    def __init__(
        self,
        config: FetchConfig,
        client: HttpClient,
        *,
        run_id: str | None = None,
    ) -> None:
        self._list_place_ids: list[str] = []
        self.config = config
        self.client = client
        self.out_dir = config.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self.out_dir / FETCH_CHECKPOINT_NAME
        self.checkpoint = self._open_checkpoint(run_id)
        self.run_id = self.checkpoint["run_id"]
        resume = self.checkpoint["journal"] if self.checkpoint["sequence"] else None
        self.journal = RequestJournal(
            self.out_dir,
            self.run_id,
            resume=resume,
            sequence=self.checkpoint["sequence"],
        )
        self._capped = False

    def _open_checkpoint(self, run_id: str | None) -> FetchCheckpoint:
        if self._checkpoint_path.is_file():
            checkpoint = load_fetch_checkpoint(self._checkpoint_path)
            if run_id is not None and run_id != checkpoint["run_id"]:
                raise KudaGoError(
                    f"{FETCH_CHECKPOINT_NAME} belongs to run {checkpoint['run_id']!r}"
                )
            return checkpoint
        journal = self.out_dir / REQUESTS_NAME
        if journal.exists():
            raise KudaGoError(f"{REQUESTS_NAME} exists without {FETCH_CHECKPOINT_NAME}")
        return new_checkpoint(run_id or _new_run_id())

    def commit(self) -> None:
        self.checkpoint["sequence"] = self.journal.sequence
        self.checkpoint["journal"] = self.journal.flush()
        write_atomic(self._checkpoint_path, encode(self.checkpoint))

    def run(self) -> FetchSummary:
        try:
            dictionaries = self._dictionaries()
            months = tuple(self._month(month) for month in self.config.months)
            event_ids = _ordered_unique(_chain_ids(months))
            details = self._details(event_ids)
            self._places(details)
            self.commit()
        finally:
            self.journal.close()
        return self._summary(months=months, dictionaries=dictionaries)

    def _summary(
        self, *, months: tuple[MonthReport, ...], dictionaries: tuple[str, ...]
    ) -> FetchSummary:
        failures = tuple(
            FailedFetch(
                kind=cast(FailureKind, state["kind"]),
                object_id=state["object_id"],
                url=state["url"],
                request_ids=tuple(state["request_ids"]),
                error=state["error"],
            )
            for state in self.checkpoint["failures"]
        )
        unresolved = tuple(
            request_id for failure in failures for request_id in failure.request_ids
        )
        return FetchSummary(
            run_id=self.run_id,
            out_dir=str(self.out_dir),
            status=self._status(months, failures),
            months=months,
            dictionaries=dictionaries,
            event_count=len(self.checkpoint["details"]),
            place_count=len(self.checkpoint["places"]),
            request_count=self.checkpoint["sequence"],
            failures=failures,
            unresolved_request_ids=unresolved,
        )

    def _status(
        self, months: tuple[MonthReport, ...], failures: tuple[FailedFetch, ...]
    ) -> RunStatus:
        if any(month.stability == "unstable" for month in months):
            return "unstable"
        incomplete = (
            self._capped
            or bool(failures)
            or any(not month.pages_complete for month in months)
            or any(month.stability != "stable" for month in months)
        )
        return "partial" if incomplete else "complete"

    def _fail(
        self,
        kind: FailureKind,
        url: str,
        error: str,
        request_ids: Iterable[str],
        *,
        object_id: str | None = None,
    ) -> None:
        self.checkpoint["failures"].append(
            {
                "kind": kind,
                "object_id": object_id,
                "url": url,
                "request_ids": list(request_ids),
                "error": error,
            }
        )

    def _get(self, url: str) -> tuple[bytes | None, str | None, tuple[str, ...]]:
        """Fetch one URL. Returns the body, the terminal error and the attempt ids."""
        recorded: list[RequestRecord] = []

        def journal(attempt: Attempt) -> None:
            recorded.append(self.journal.record(attempt))

        result = self.client.fetch(url, on_attempt=journal)
        ids = tuple(record["request_id"] for record in recorded)
        if result.ok:
            return result.body, None, ids
        error = result.error or "unknown_error"
        if result.retries_exhausted:
            error = f"{error} (retries exhausted after {len(result.attempts)} attempts)"
        return None, error, ids

    def _dictionaries(self) -> tuple[str, ...]:
        target_dir = self.out_dir / DICTIONARIES_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for name, path in DICTIONARY_PATHS:
            if name in self.checkpoint["dictionaries"]:
                saved.append(name)
                continue
            url = build_url(path, {"lang": self.config.lang})
            body, error, request_ids = self._get(url)
            if body is None:
                self._fail("dictionary", url, error or "unknown_error", request_ids, object_id=name)
                continue
            try:
                entries = parse_json_array(body)
            except ResponseError as failure:
                self._fail("dictionary", url, str(failure), request_ids, object_id=name)
                continue
            if not all(isinstance(entry, dict) for entry in entries):
                self._fail(
                    "dictionary", url, "entries are not objects", request_ids, object_id=name
                )
                continue
            write_atomic(target_dir / f"{name}.json", body)
            self.checkpoint["dictionaries"][name] = request_ids[-1]
            saved.append(name)
            self.commit()
        return tuple(saved)

    def _month(self, month: str) -> MonthReport:
        since, until = month_window(month, self.config.overlap_days)
        passes: list[_PassResult] = []
        stability: Stability = "unknown"
        for index in range(1, max(1, self.config.max_id_passes) + 1):
            current = self._walk(month, index, since, until)
            passes.append(current)
            if not current.complete:
                break
            if len(passes) >= 2 and set(passes[-2].ids) == set(current.ids):
                stability = "stable"
                break
            if len(passes) >= 2:
                stability = "unstable"
        last = passes[-1]
        return MonthReport(
            month=month,
            actual_since=since,
            actual_until=until,
            passes=len(passes),
            stability="unknown" if not last.complete else stability,
            pages_complete=last.complete,
            page_counts=last.counts,
            event_ids=last.ids,
        )

    def _walk(self, month: str, pass_index: int, since: str, until: str) -> _PassResult:
        url: str | None = build_url(
            EVENTS_PATH,
            {
                "lang": self.config.lang,
                "location": self.config.location,
                "order_by": self.config.order_by,
                "page_size": self.config.page_size,
                "fields": ",".join(LIST_FIELDS),
                "expand": LIST_EXPAND,
                "actual_since": since,
                "actual_until": until,
            },
        )
        seen: set[str] = set()
        ids: list[int] = []
        counts: list[int] = []
        complete = True
        pages = 0
        while url is not None:
            if url in seen:
                self._fail("list_page", url, "next revisits a URL already seen in this pass", ())
                complete = False
                break
            seen.add(url)
            if self.config.page_limit is not None and pages >= self.config.page_limit:
                self._capped = True
                complete = False
                break
            key = f"{month}#{pass_index}#{url}"
            cached = self.checkpoint["pages"].get(key)
            if cached is not None:
                ids.extend(cached["ids"])
                self._list_place_ids.extend(cached["place_ids"])
                counts.append(cached["count"])
                pages += 1
                url = cached["next"]
                continue
            body, error, request_ids = self._get(url)
            if body is None:
                self._fail("list_page", url, error or "unknown_error", request_ids)
                complete = False
                break
            try:
                page = parse_page(body)
                page_ids, page_place_ids = _result_references(page)
            except ResponseError as failure:
                self._fail("list_page", url, str(failure), request_ids)
                complete = False
                break
            following = page.next_url
            if following is not None and not is_followable(following):
                self._fail("list_page", url, f"next leaves the API root: {following}", request_ids)
                complete = False
                following = None
            ids.extend(page_ids)
            self._list_place_ids.extend(page_place_ids)
            counts.append(page.count)
            pages += 1
            self.checkpoint["pages"][key] = {
                "ids": list(page_ids),
                "place_ids": list(page_place_ids),
                "next": following,
                "count": page.count,
                "request_id": request_ids[-1],
            }
            self.commit()
            url = following
        return _PassResult(ids=tuple(ids), counts=tuple(counts), complete=complete)

    def _details(self, event_ids: Sequence[int]) -> dict[str, DetailState]:
        if not self.config.fetch_details:
            return self.checkpoint["details"]
        fetched = 0
        for event_id in event_ids:
            key = str(event_id)
            if key in self.checkpoint["details"]:
                continue
            if self.config.detail_limit is not None and fetched >= self.config.detail_limit:
                self._capped = True
                break
            url = build_url(
                f"{EVENTS_PATH}{event_id}/",
                {
                    "lang": self.config.lang,
                    "fields": ",".join(DETAIL_FIELDS),
                    "expand": DETAIL_EXPAND,
                    "text_format": TEXT_FORMAT,
                },
            )
            payload, request_ids = self._object(url, "event", key)
            fetched += 1
            if payload is None:
                continue
            self.checkpoint["details"][key] = {
                "request_id": request_ids[-1],
                "place_id": _place_id(payload),
            }
            self.commit()
        return self.checkpoint["details"]

    def _places(self, details: Mapping[str, DetailState]) -> None:
        """Every place the run saw, whether it came from an inventory page or a card.

        A card whose detail request failed still names its place on the list page, so
        sourcing place ids from both keeps one broken card from hiding a whole venue.
        """
        if not self.config.fetch_places:
            return
        fetched = 0
        discovered = [
            *self._list_place_ids,
            *(state["place_id"] for state in details.values() if state["place_id"] is not None),
        ]
        for place_id in _ordered_unique(discovered):
            if place_id in self.checkpoint["places"]:
                continue
            if self.config.place_limit is not None and fetched >= self.config.place_limit:
                self._capped = True
                break
            url = build_url(
                f"{PLACES_PATH}{place_id}/",
                {"lang": self.config.lang, "text_format": TEXT_FORMAT},
            )
            payload, request_ids = self._object(url, "place", place_id)
            fetched += 1
            if payload is None:
                continue
            self.checkpoint["places"][place_id] = request_ids[-1]
            self.commit()

    def _object(
        self, url: str, kind: FailureKind, object_id: str
    ) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
        body, error, request_ids = self._get(url)
        if body is None:
            self._fail(kind, url, error or "unknown_error", request_ids, object_id=object_id)
            return None, request_ids
        try:
            return parse_json_object(body), request_ids
        except ResponseError as failure:
            self._fail(kind, url, str(failure), request_ids, object_id=object_id)
            return None, request_ids


def _result_references(page: Page) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """The card ids and the place ids an inventory page points at."""
    ids: list[int] = []
    place_ids: list[str] = []
    for entry in page.results:
        if not isinstance(entry, dict):
            raise ResponseError(f"results entry is a {type(entry).__name__}, not an object")
        value = entry.get("id")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ResponseError(f"results entry has no integer id: {value!r}")
        ids.append(value)
        place_id = _place_id(entry)
        if place_id is not None:
            place_ids.append(place_id)
    return tuple(ids), tuple(place_ids)


def _place_id(payload: Mapping[str, Any]) -> str | None:
    place = payload.get("place")
    if not isinstance(place, dict):
        return None
    value = place.get("id")
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    return str(value)


def _chain_ids(months: Iterable[MonthReport]) -> list[int]:
    return [event_id for month in months for event_id in month.event_ids]


def _ordered_unique[T](values: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    ordered: list[T] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def live_client(*, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> HttpClient:
    """The throttled client used against the real API."""
    return HttpClient(UrllibTransport(), timeout_seconds=timeout_seconds)


def collect(
    config: FetchConfig,
    *,
    client: HttpClient | None = None,
    run_id: str | None = None,
) -> FetchSummary:
    """Run or resume a fetch over `config.out_dir` and report what was observed."""
    return Collector(config, client or live_client(), run_id=run_id).run()


def collect_months(
    out_dir: Path,
    months: Sequence[str],
    *,
    client: HttpClient | None = None,
    run_id: str | None = None,
    **overrides: Any,
) -> FetchSummary:
    config = FetchConfig(out_dir=out_dir, months=tuple(months), **overrides)
    return collect(config, client=client, run_id=run_id)

"""The normalisation half of the KudaGo collector: `raw/` bytes to the delivery files.

This module performs no network I/O. It reads the evidence the fetcher left behind --
`requests.jsonl`, one saved body per attempt under `raw/`, and the reference lists
under `dictionaries/` -- and writes `events.jsonl`, `places.jsonl`,
`occurrences.jsonl`, `observations.jsonl` and `quarantine.jsonl`.

Contracts that shape the code:

* Determinism. Every value comes from the request journal, the saved bodies or the
  configuration; nothing comes from the clock or from directory iteration order, so
  two runs over the same inputs produce byte-identical files.
* Availability. A body fetched in 2026 is evidence about 2026: `available_at` is the
  journal's `fetched_at` and `availability_basis` is `observed_snapshot`. A card's own
  `publication_date` never moves that earlier, and because the API serves no version
  history every row carries `historical_snapshot_unavailable`.
* Observations. Every sighting is recorded, keyed by `(request_id, object_type, id)`,
  even when the content repeats. A card going A -> B -> A leaves three observations
  over two snapshots; keeping only distinct content would erase the reverted
  postponement.
* Shapes. The API returns one entity in three shapes: a projection on a list page
  (`fields=id,dates,place,site_url`, no title by construction), an object inlined by
  `expand`, and the detail response. They are not three states of the entity, so what
  an object *is* comes from the request that produced it, never from its content.
  Only a detail response -- `events/{id}/`, `places/{id}/` -- becomes a delivery row,
  an observation or a required-field check. Projections are discovery: they are
  counted in the report and left in `raw/`, and they enter no delivery file.
* Identity. `event_id`, `title` and `source_url` are required: an empty or absent
  value quarantines the card together with its date records, because the contract
  types them as `str` and carrying `""` would be an empty string standing in for
  absence. The check applies to a detail response alone: a projection that omits a
  field is not a card that lacks a name. A venue keeps a flag instead -- quarantining
  it would strip coordinates from every valid card that points at it.
* Evidence. Presence of a card does not prove the event happened, so `status` is
  `unknown` and anything else would need `status_evidence`. Coordinates come from
  `place.coords` alone -- `location.coords` is the city centre and is never read.
* Bad data is quarantined with a reason and a `raw_ref`, never repaired in place.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from tramflow_ml.external.kudago.records import (
    DICTIONARIES_DIR,
    EVENTS_NAME,
    MOSCOW,
    OBSERVATIONS_NAME,
    OCCURRENCES_NAME,
    PLACES_NAME,
    QUARANTINE_NAME,
    REQUESTS_NAME,
    SCHEMA_VERSION,
    SOURCE,
    CoordinatePrecision,
    CoordinateSource,
    EventRecord,
    ObjectType,
    ObservationRecord,
    OccurrenceRecord,
    PlaceRecord,
    QuarantineRecord,
    canonical_bytes,
    derivation_id,
    occurrence_id,
    snapshot_id,
)
from tramflow_ml.external.kudago.timerules import (
    TIMERULES_VERSION,
    Session,
    TimeRuleError,
    Window,
    parse_windows,
    resolve_date_record,
)
from tramflow_ml.ingestion.checkpoint import write_atomic

NORMALIZER_VERSION = "kudago-normalize.v1"

CATEGORIES_DICTIONARY = "event-categories"
LOCATIONS_DICTIONARY = "locations"

MOSCOW_LAT_RANGE = (54.2, 56.9)
MOSCOW_LON_RANGE = (35.1, 40.2)
PLAUSIBLE_FIRST_YEAR = 2000
PLAUSIBLE_LAST_YEAR = 2100

FLAG_HISTORICAL_SNAPSHOT_UNAVAILABLE = "historical_snapshot_unavailable"
FLAG_TITLE_MISSING = "title_missing"
FLAG_SOURCE_URL_MISSING = "source_url_missing"
FLAG_PLACE_MISSING = "place_missing"
FLAG_PLACE_DETAIL_MISSING = "place_detail_missing"
FLAG_DATES_MISSING = "dates_missing"
FLAG_NO_DATES = "no_dates"
FLAG_PUBLICATION_DATE_IMPLAUSIBLE = "publication_date_implausible"
FLAG_UNKNOWN_CATEGORY = "unknown_category"
FLAG_UNKNOWN_LOCATION = "unknown_location"
FLAG_COORDINATES_MISSING = "coordinates_missing"
FLAG_COORDINATES_MALFORMED = "coordinates_malformed"
FLAG_COORDINATES_OUT_OF_RANGE = "coordinates_out_of_range"
FLAG_COORDINATES_OUTSIDE_MOSCOW = "coordinates_outside_moscow"
FLAG_COORDINATES_POSSIBLY_TRANSPOSED = "coordinates_possibly_transposed"

REASON_INVALID_JOURNAL_LINE = "invalid_request_record"
REASON_RAW_BODY_MISSING = "raw_body_missing"
REASON_BODY_HASH_MISMATCH = "body_hash_mismatch"
REASON_BODY_NOT_JSON = "body_not_json"
REASON_BODY_NOT_AN_OBJECT = "body_not_an_object"
REASON_MISSING_IDENTIFIER = "missing_identifier"
REASON_MISSING_TITLE = "missing_title"
REASON_MISSING_SOURCE_URL = "missing_source_url"
REASON_CONFLICTING_SNAPSHOT = "conflicting_snapshot_in_request"
REASON_INVALID_DATES_FIELD = "invalid_dates_field"
REASON_SESSION_INVARIANT = "session_invariant_violated"

_NON_FINITE = ("NaN", "Infinity", "-Infinity")


class NormalizeError(Exception):
    """The input directory cannot be read as a collector run."""


@dataclass(frozen=True, slots=True)
class NormalizeConfig:
    """Everything that decides the output, plus where to read it and write it.

    `fingerprint` deliberately excludes the paths: the same `raw/` normalised into
    two different directories must produce the same derivation ids, or the
    reproducibility check would compare two different derivations.
    """

    input_dir: Path
    output_dir: Path
    windows: tuple[tuple[str, str], ...]
    normalizer_version: str = NORMALIZER_VERSION

    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_bytes(
                {
                    "windows": [list(span) for span in self.windows],
                    "normalizer_version": self.normalizer_version,
                    "schema_version": SCHEMA_VERSION,
                    "timerules_version": TIMERULES_VERSION,
                }
            )
        ).hexdigest()


class NormalizeReport(TypedDict):
    """Counts for the manifest and the quality report, all derived from the journal."""

    schema_version: str
    normalizer_version: str
    timerules_version: str
    normalizer_config_sha256: str
    windows: list[list[str]]
    run_ids: list[str]
    first_fetched_at: str | None
    last_fetched_at: str | None
    requests_total: int
    requests_read: int
    requests_skipped: int
    events: int
    places: int
    event_projections: int
    place_projections: int
    events_without_detail: int
    places_without_detail: int
    observations: int
    occurrences: int
    quarantined: int
    date_records: int
    date_records_resolved: int
    date_records_quarantined: int
    reconciled: bool
    quarantine_reasons: dict[str, int]
    quality_flags: dict[str, int]
    time_precision: dict[str, int]
    schedule_basis: dict[str, int]


@dataclass(frozen=True, slots=True)
class _Context:
    """Where a payload came from, so every row can point back at its bytes."""

    request_id: str
    run_id: str
    url: str
    fetched_at: str
    raw_path: str
    pointer: str

    def ref(self, suffix: str = "") -> str:
        return f"{self.raw_path}#{self.pointer}{suffix}"

    def at(self, pointer: str) -> "_Context":
        return _Context(
            request_id=self.request_id,
            run_id=self.run_id,
            url=self.url,
            fetched_at=self.fetched_at,
            raw_path=self.raw_path,
            pointer=pointer,
        )


@dataclass(frozen=True, slots=True)
class _Pending:
    """A delivered card waiting for the venue detail its occurrences will point at."""

    event: EventRecord
    place_id: str | None
    context: _Context


@dataclass(slots=True)
class _Run:
    config: NormalizeConfig
    window: Window
    dictionaries: Mapping[str, frozenset[str]]
    fingerprint: str
    events: list[EventRecord] = field(default_factory=list)
    places: list[PlaceRecord] = field(default_factory=list)
    observations: list[ObservationRecord] = field(default_factory=list)
    occurrences: list[OccurrenceRecord] = field(default_factory=list)
    quarantine: list[QuarantineRecord] = field(default_factory=list)
    seen_snapshots: set[tuple[str, str, str]] = field(default_factory=set)
    seen_observations: dict[tuple[str, str, str], str] = field(default_factory=dict)
    pending: list[_Pending] = field(default_factory=list)
    place_details: dict[str, str] = field(default_factory=dict)
    delivered_events: set[str] = field(default_factory=set)
    events_listed: set[str] = field(default_factory=set)
    places_referenced: set[str] = field(default_factory=set)
    event_projections: int = 0
    place_projections: int = 0
    run_ids: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    requests_read: int = 0
    requests_skipped: int = 0
    date_records: int = 0
    date_records_resolved: int = 0
    date_records_quarantined: int = 0

    def reject(
        self,
        *,
        reason: str,
        detail: str,
        request_id: str | None = None,
        event_id: str | None = None,
        place_id: str | None = None,
        date_index: int | None = None,
        raw_ref: str | None = None,
    ) -> None:
        self.quarantine.append(
            {
                "request_id": request_id,
                "event_id": event_id,
                "place_id": place_id,
                "date_index": date_index,
                "reason": reason,
                "detail": detail,
                "raw_ref": raw_ref,
            }
        )


def normalize(config: NormalizeConfig) -> NormalizeReport:
    """Read one fetched run and write the five normalised files. No network, no clock."""
    window = parse_windows(config.windows)
    journal = _read_journal(config.input_dir)
    run = _Run(
        config=config,
        window=window,
        dictionaries=_read_dictionaries(config.input_dir),
        fingerprint=config.fingerprint(),
    )
    for line_number, line in journal:
        _read_request(run, line_number, line)
    for pending in run.pending:
        _expand_pending(run, pending)
    _write_outputs(run)
    return _report(run, len(journal))


def _read_journal(input_dir: Path) -> list[tuple[int, bytes]]:
    path = input_dir / REQUESTS_NAME
    if not path.is_file():
        raise NormalizeError(f"{path} is missing: this is not a collector run directory")
    return [
        (number, line)
        for number, line in enumerate(path.read_bytes().splitlines(), start=1)
        if line.strip()
    ]


def _read_dictionaries(input_dir: Path) -> dict[str, frozenset[str]]:
    """Slug sets used to flag unknown references; an absent list validates nothing."""
    directory = input_dir / DICTIONARIES_DIR
    if not directory.is_dir():
        return {}
    known: dict[str, frozenset[str]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = _load_json(path.read_bytes())
        if not isinstance(payload, list):
            continue
        slugs = {
            entry["slug"]
            for entry in payload
            if isinstance(entry, dict) and isinstance(entry.get("slug"), str)
        }
        if slugs:
            known[path.stem] = frozenset(slugs)
    return known


def _load_json(body: bytes) -> Any:
    """Reject the JSON extensions the contract forbids instead of carrying them through.

    `canonical_bytes` refuses NaN and Infinity, so a body containing them has to fail
    here, where the request can be quarantined with its own reason.
    """
    try:
        return json.loads(body, parse_constant=_reject_constant)
    except ValueError:
        return None


def _reject_constant(name: str) -> float:
    raise ValueError(f"body contains the non-finite literal {name}")


def _read_request(run: _Run, line_number: int, line: bytes) -> None:
    record = _load_json(line)
    if not isinstance(record, dict):
        run.reject(
            reason=REASON_INVALID_JOURNAL_LINE,
            detail="line is not a JSON object",
            raw_ref=f"{REQUESTS_NAME}#L{line_number}",
        )
        return
    request_id = record.get("request_id")
    raw_path = record.get("raw_path")
    fetched_at = record.get("fetched_at")
    run_id = record.get("run_id")
    url = record.get("url")
    if not isinstance(request_id, str) or not isinstance(url, str):
        run.reject(
            reason=REASON_INVALID_JOURNAL_LINE,
            detail="request_id and url must be strings",
            raw_ref=f"{REQUESTS_NAME}#L{line_number}",
        )
        return
    if not isinstance(fetched_at, str) or _parse_iso(fetched_at) is None:
        run.reject(
            reason=REASON_INVALID_JOURNAL_LINE,
            detail="fetched_at is not an ISO 8601 timestamp",
            request_id=request_id,
            raw_ref=f"{REQUESTS_NAME}#L{line_number}",
        )
        return
    if not isinstance(raw_path, str):
        run.requests_skipped += 1
        return
    context = _Context(
        request_id=request_id,
        run_id=run_id if isinstance(run_id, str) else "",
        url=url,
        fetched_at=fetched_at,
        raw_path=raw_path,
        pointer="",
    )
    body = _read_body(run, record, context)
    if body is None:
        return
    payload = _load_json(body)
    if not isinstance(payload, dict):
        run.requests_skipped += 1
        if payload is None:
            run.reject(
                reason=REASON_BODY_NOT_JSON,
                detail="saved body is not valid JSON, or holds a non-finite number",
                request_id=request_id,
                raw_ref=context.ref(),
            )
        elif not isinstance(payload, list):
            run.reject(
                reason=REASON_BODY_NOT_AN_OBJECT,
                detail=f"saved body is a JSON {type(payload).__name__}, not an object",
                request_id=request_id,
                raw_ref=context.ref(),
            )
        return
    run.requests_read += 1
    if context.run_id and context.run_id not in run.run_ids:
        run.run_ids.append(context.run_id)
    run.fetched.append(fetched_at)
    for pointer, obj in _objects(payload):
        _read_object(run, obj, context.at(pointer))


def _read_body(run: _Run, record: Mapping[str, Any], context: _Context) -> bytes | None:
    path = context.raw_path
    target = run.config.input_dir / path
    if not target.is_file():
        run.reject(
            reason=REASON_RAW_BODY_MISSING,
            detail=f"the journal records {path} but the file is absent",
            request_id=context.request_id,
            raw_ref=context.ref(),
        )
        return None
    body = target.read_bytes()
    expected = record.get("body_sha256")
    if isinstance(expected, str) and hashlib.sha256(body).hexdigest() != expected:
        run.reject(
            reason=REASON_BODY_HASH_MISMATCH,
            detail="the saved body does not match body_sha256 in the journal",
            request_id=context.request_id,
            raw_ref=context.ref(),
        )
        return None
    if not path.endswith(".json"):
        run.reject(
            reason=REASON_BODY_NOT_JSON,
            detail=f"the saved body {path} is not JSON",
            request_id=context.request_id,
            raw_ref=context.ref(),
        )
        return None
    return body


def _objects(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    results = payload.get("results")
    if isinstance(results, list):
        return [
            (f"/results/{index}", entry)
            for index, entry in enumerate(results)
            if isinstance(entry, dict)
        ]
    return [("", dict(payload))]


def _request_role(url: str) -> tuple[ObjectType, bool] | None:
    """What the request asked for: the object type and whether it is a detail response.

    Classification comes from the path alone. The same entity arrives as a list-page
    projection, as an object inlined by `expand` and as a detail response, and only
    the request distinguishes them -- a projection is missing fields by construction,
    so its content cannot be trusted to say what it is.
    """
    segments = [part for part in url.split("?", 1)[0].split("#", 1)[0].split("/") if part]
    for index in range(len(segments) - 1, -1, -1):
        if segments[index] == "events":
            return "event", index + 1 < len(segments)
        if segments[index] == "places":
            return "place", index + 1 < len(segments)
    return None


def _read_object(run: _Run, payload: dict[str, Any], context: _Context) -> None:
    role = _request_role(context.url)
    if role is None:
        return
    kind, is_detail = role
    if not is_detail:
        _note_projection(run, kind, payload, context)
    elif kind == "event":
        _read_event(run, payload, context)
    else:
        _read_place(run, payload, context)


def _note_projection(
    run: _Run, kind: ObjectType, payload: Mapping[str, Any], context: _Context
) -> None:
    """Record that a partial view of an object exists, and deliver nothing.

    An inventory row carries the ids the fetcher walks and nothing that can be
    published: `{"id": 2033}` is a reference to a venue, not a venue without a name.
    """
    object_id = _identifier(payload, "id")
    if object_id is None:
        run.reject(
            reason=REASON_MISSING_IDENTIFIER,
            detail=f"{kind} projection has no usable id",
            request_id=context.request_id,
            raw_ref=context.ref("/id"),
        )
        return
    if kind == "event":
        run.event_projections += 1
        run.events_listed.add(object_id)
        place = payload.get("place")
        if isinstance(place, dict):
            _note_projection(run, "place", place, context.at(f"{context.pointer}/place"))
        return
    run.place_projections += 1
    run.places_referenced.add(object_id)


def _read_event(run: _Run, payload: dict[str, Any], context: _Context) -> None:
    event_id = _identifier(payload, "id")
    if event_id is None:
        run.reject(
            reason=REASON_MISSING_IDENTIFIER,
            detail="event payload has no usable id",
            request_id=context.request_id,
            raw_ref=context.ref("/id"),
        )
        return
    place_payload = payload.get("place")
    place_id: str | None = None
    if isinstance(place_payload, dict):
        place_id = _identifier(place_payload, "id")
        if place_id is not None:
            run.place_projections += 1
            run.places_referenced.add(place_id)
    snapshot = snapshot_id(payload)
    title = _text(payload, "title")
    source_url = _text(payload, "site_url")
    if title is None or source_url is None:
        _reject_card(run, payload, event_id, snapshot, context)
        return
    if not _record_observation(run, "event", event_id, snapshot, context):
        return
    if ("event", event_id, snapshot) in run.seen_snapshots:
        return
    run.seen_snapshots.add(("event", event_id, snapshot))
    flags = {FLAG_HISTORICAL_SNAPSHOT_UNAVAILABLE}
    if not isinstance(place_payload, dict):
        flags.add(FLAG_PLACE_MISSING)
    dates_raw = _dates(run, payload, event_id, context, flags)
    categories = _slugs(payload, "categories")
    known_categories = run.dictionaries.get(CATEGORIES_DICTIONARY)
    if known_categories is not None and not set(categories) <= known_categories:
        flags.add(FLAG_UNKNOWN_CATEGORY)
    location_slug = _location_slug(payload.get("location"))
    known_locations = run.dictionaries.get(LOCATIONS_DICTIONARY)
    if location_slug is not None and known_locations is not None:
        if location_slug not in known_locations:
            flags.add(FLAG_UNKNOWN_LOCATION)
    published_at = _moment(payload.get("publication_date"), flags)
    event: EventRecord = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "run_id": context.run_id,
        "request_id": context.request_id,
        "snapshot_id": snapshot,
        "fetched_at": context.fetched_at,
        "available_at": context.fetched_at,
        "availability_basis": "observed_snapshot",
        "availability_evidence": f"{REQUESTS_NAME}#{context.request_id}",
        "quality_flags": sorted(flags),
        "payload": payload,
        "event_id": event_id,
        "title": title,
        "source_url": source_url,
        "published_at": published_at,
        "location_slug": location_slug,
        "place_id": place_id,
        "categories": categories,
        "tags": _slugs(payload, "tags"),
        "description": _text(payload, "description"),
        "body_text": _text(payload, "body_text"),
        "price_text": _text(payload, "price"),
        "age_restriction": _age(payload.get("age_restriction")),
        "is_free": _boolean(payload.get("is_free")),
        "dates_raw": dates_raw,
        "status": "unknown",
        "status_evidence": None,
    }
    run.events.append(event)
    run.delivered_events.add(event_id)
    run.pending.append(_Pending(event=event, place_id=place_id, context=context))


def _expand_pending(run: _Run, pending: _Pending) -> None:
    """Join the card to the venue detail delivered for it, then expand its dates.

    The venue is provenance, not date evidence: the sessions come from the card alone,
    so `available_at` stays the card's. A card whose venue was never fetched as a
    detail keeps a null `place_snapshot_id` and says so with a flag, rather than
    pointing at an inlined projection that no row in `places.jsonl` matches.
    """
    place_snapshot = (
        run.place_details.get(pending.place_id) if pending.place_id is not None else None
    )
    if pending.place_id is not None and place_snapshot is None:
        pending.event["quality_flags"] = sorted(
            set(pending.event["quality_flags"]) | {FLAG_PLACE_DETAIL_MISSING}
        )
    _expand(run, pending.event, pending.place_id, place_snapshot, pending.context)


def _reject_card(
    run: _Run, payload: Mapping[str, Any], event_id: str, snapshot: str, context: _Context
) -> None:
    """Quarantine a card whose required identity fields are empty, dates included.

    The date records are counted against this row rather than left out of the totals:
    a reader of the report must be able to see how many sessions the rejection cost.
    """
    if ("event", event_id, snapshot) in run.seen_snapshots:
        return
    run.seen_snapshots.add(("event", event_id, snapshot))
    field_name, reason = (
        ("title", REASON_MISSING_TITLE)
        if _text(payload, "title") is None
        else ("site_url", REASON_MISSING_SOURCE_URL)
    )
    dates = payload.get("dates")
    lost = len(dates) if isinstance(dates, list) else 0
    run.date_records += lost
    run.date_records_quarantined += lost
    run.reject(
        reason=reason,
        detail=f"{field_name} is empty or absent; {lost} date records fall with the card",
        request_id=context.request_id,
        event_id=event_id,
        raw_ref=context.ref(f"/{field_name}"),
    )


def _read_place(run: _Run, payload: dict[str, Any], context: _Context) -> None:
    place_id = _identifier(payload, "id")
    if place_id is None:
        run.reject(
            reason=REASON_MISSING_IDENTIFIER,
            detail="place payload has no usable id",
            request_id=context.request_id,
            raw_ref=context.ref("/id"),
        )
        return
    snapshot = snapshot_id(payload)
    if not _record_observation(run, "place", place_id, snapshot, context):
        return
    if ("place", place_id, snapshot) in run.seen_snapshots:
        return
    run.seen_snapshots.add(("place", place_id, snapshot))
    run.place_details[place_id] = snapshot
    flags = {FLAG_HISTORICAL_SNAPSHOT_UNAVAILABLE}
    title = _text(payload, "title")
    if title is None:
        flags.add(FLAG_TITLE_MISSING)
    source_url = _text(payload, "site_url")
    if source_url is None:
        flags.add(FLAG_SOURCE_URL_MISSING)
    lat, lon, source, precision, evidence = _coordinates(payload, flags, context)

    place: PlaceRecord = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "run_id": context.run_id,
        "request_id": context.request_id,
        "snapshot_id": snapshot,
        "fetched_at": context.fetched_at,
        "available_at": context.fetched_at,
        "availability_basis": "observed_snapshot",
        "availability_evidence": f"{REQUESTS_NAME}#{context.request_id}",
        "quality_flags": sorted(flags),
        "payload": payload,
        "place_id": place_id,
        "title": title or "",
        "source_url": source_url or "",
        "address": _text(payload, "address"),
        "location_slug": _location_slug(payload.get("location")),
        "subway_text": _text(payload, "subway"),
        "timetable_text": _text(payload, "timetable"),
        "lat": lat,
        "lon": lon,
        "coordinate_source": source,
        "coordinate_evidence": evidence,
        "coordinate_precision": precision,
        "is_closed": _boolean(payload.get("is_closed")),
        "is_stub": _boolean(payload.get("is_stub")),
    }
    run.places.append(place)


def _record_observation(
    run: _Run, kind: ObjectType, object_id: str, snapshot: str, context: _Context
) -> bool:
    """One sighting per `(request_id, object_type, id)`, repeated content included.

    The same venue embedded in two cards of one page is one sighting; the same id
    arriving twice in one request with different content is a source contradiction
    and is quarantined rather than resolved by preferring one of them.
    """
    key = (context.request_id, kind, object_id)
    previous = run.seen_observations.get(key)
    if previous == snapshot:
        return False
    if previous is not None:
        run.reject(
            reason=REASON_CONFLICTING_SNAPSHOT,
            detail=f"{kind} {object_id} appears twice in one request with different content",
            request_id=context.request_id,
            event_id=object_id if kind == "event" else None,
            place_id=object_id if kind == "place" else None,
            raw_ref=context.ref(),
        )
        return False
    run.seen_observations[key] = snapshot
    run.observations.append(
        {
            "object_type": kind,
            "event_or_place_id": object_id,
            "snapshot_id": snapshot,
            "request_id": context.request_id,
            "fetched_at": context.fetched_at,
            "available_at": context.fetched_at,
            "availability_basis": "observed_snapshot",
            "availability_evidence": f"{REQUESTS_NAME}#{context.request_id}",
        }
    )
    return True


def _dates(
    run: _Run,
    payload: Mapping[str, Any],
    event_id: str,
    context: _Context,
    flags: set[str],
) -> list[dict[str, Any]]:
    raw = payload.get("dates")
    if raw is None:
        flags.add(FLAG_DATES_MISSING)
        return []
    if not isinstance(raw, list):
        flags.add(FLAG_DATES_MISSING)
        run.reject(
            reason=REASON_INVALID_DATES_FIELD,
            detail="dates is present but is not a list",
            request_id=context.request_id,
            event_id=event_id,
            raw_ref=context.ref("/dates"),
        )
        return []
    if not raw:
        flags.add(FLAG_NO_DATES)
    return list(raw)


def _expand(
    run: _Run,
    event: EventRecord,
    place_id: str | None,
    place_snapshot: str | None,
    context: _Context,
) -> None:
    """Every raw date record leaves either at least one occurrence or a quarantine row."""
    derivation = derivation_id(
        event_snapshot_id=event["snapshot_id"],
        place_snapshot_id=place_snapshot,
        normalizer_version=run.config.normalizer_version,
        normalizer_config_sha256=run.fingerprint,
    )
    for index, raw_date in enumerate(event["dates_raw"]):
        run.date_records += 1
        pointer = context.ref(f"/dates/{index}")
        if not isinstance(raw_date, dict):
            run.date_records_quarantined += 1
            run.reject(
                reason=REASON_INVALID_DATES_FIELD,
                detail="date record is not an object",
                request_id=context.request_id,
                event_id=event["event_id"],
                date_index=index,
                raw_ref=pointer,
            )
            continue
        try:
            resolution = resolve_date_record(raw_date, run.window)
        except TimeRuleError as error:
            run.date_records_quarantined += 1
            run.reject(
                reason=REASON_SESSION_INVARIANT,
                detail=str(error),
                request_id=context.request_id,
                event_id=event["event_id"],
                date_index=index,
                raw_ref=pointer,
            )
            continue
        if resolution.rejection is not None:
            run.date_records_quarantined += 1
            run.reject(
                reason=resolution.rejection.reason,
                detail=resolution.rejection.detail,
                request_id=context.request_id,
                event_id=event["event_id"],
                date_index=index,
                raw_ref=pointer,
            )
            continue
        run.date_records_resolved += 1
        for expansion, session in enumerate(resolution.sessions):
            run.occurrences.append(
                _occurrence(event, place_id, place_snapshot, derivation, index, expansion, session)
            )


def _occurrence(
    event: EventRecord,
    place_id: str | None,
    place_snapshot: str | None,
    derivation: str,
    date_index: int,
    expansion_index: int,
    session: Session,
) -> OccurrenceRecord:
    raw_date = event["dates_raw"][date_index]
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event["event_id"],
        "event_snapshot_id": event["snapshot_id"],
        "derivation_id": derivation,
        "occurrence_id": occurrence_id(
            derivation=derivation, date_index=date_index, expansion_index=expansion_index
        ),
        "date_index": date_index,
        "expansion_index": expansion_index,
        "place_id": place_id,
        "place_snapshot_id": place_snapshot,
        "raw_date": raw_date,
        "start_at": session.start_at.isoformat() if session.start_at is not None else None,
        "end_at": session.end_at.isoformat() if session.end_at is not None else None,
        "start_date": session.start_date,
        "end_date": session.end_date,
        "time_precision": session.time_precision,
        "schedule_basis": session.schedule_basis,
        "available_at": event["available_at"],
        "quality_flags": list(session.quality_flags),
    }


def _identifier(payload: Mapping[str, Any], key: str) -> str | None:
    """Ids stay exactly as the source wrote them; a JSON number keeps its digits."""
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    return value or None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _age(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    return value if isinstance(value, str) and value else None


def _slugs(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, str) and entry]


def _location_slug(value: Any) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        slug = value.get("slug")
        return slug if isinstance(slug, str) and slug else None
    return None


def _moment(value: Any, flags: set[str]) -> str | None:
    """A Unix publication date in Moscow time; it never becomes `available_at`."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        flags.add(FLAG_PUBLICATION_DATE_IMPLAUSIBLE)
        return None
    try:
        moment = datetime.fromtimestamp(value, MOSCOW)
    except (OSError, OverflowError, ValueError):
        flags.add(FLAG_PUBLICATION_DATE_IMPLAUSIBLE)
        return None
    if not PLAUSIBLE_FIRST_YEAR <= moment.year <= PLAUSIBLE_LAST_YEAR:
        flags.add(FLAG_PUBLICATION_DATE_IMPLAUSIBLE)
        return None
    return moment.isoformat()


def _coordinates(
    payload: Mapping[str, Any], flags: set[str], context: _Context
) -> tuple[float | None, float | None, CoordinateSource, CoordinatePrecision, str | None]:
    """`place.coords` is the only source.

    `location.coords` is the centre of the city, not of the venue, so a card without
    venue coordinates stays without coordinates. A pair that fails validation is
    reported as missing and flagged while the payload keeps the original values; a
    pair that would only be plausible the other way round is flagged, never swapped,
    because the source has been observed to ship both orders.
    """
    coords = payload.get("coords")
    if not isinstance(coords, dict):
        flags.add(FLAG_COORDINATES_MISSING)
        return None, None, "missing", "unknown", None
    evidence = context.ref("/coords")
    lat = _number(coords.get("lat"))
    lon = _number(coords.get("lon"))
    if lat is None or lon is None:
        absent = coords.get("lat") is None and coords.get("lon") is None
        flags.add(FLAG_COORDINATES_MISSING if absent else FLAG_COORDINATES_MALFORMED)
        return None, None, "missing", "unknown", evidence
    if abs(lat) > 90 or abs(lon) > 180:
        flags.add(FLAG_COORDINATES_OUT_OF_RANGE)
        return None, None, "missing", "unknown", evidence
    if not _in_moscow(lat, lon):
        flags.add(FLAG_COORDINATES_OUTSIDE_MOSCOW)
        if _in_moscow(lon, lat):
            flags.add(FLAG_COORDINATES_POSSIBLY_TRANSPOSED)
        return None, None, "missing", "unknown", evidence
    return lat, lon, "place_api", "venue_area", evidence


def _in_moscow(lat: float, lon: float) -> bool:
    return (
        MOSCOW_LAT_RANGE[0] <= lat <= MOSCOW_LAT_RANGE[1]
        and MOSCOW_LON_RANGE[0] <= lon <= MOSCOW_LON_RANGE[1]
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parse_iso(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _write_outputs(run: _Run) -> None:
    run.config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        (EVENTS_NAME, run.events),
        (PLACES_NAME, run.places),
        (OCCURRENCES_NAME, run.occurrences),
        (OBSERVATIONS_NAME, run.observations),
        (QUARANTINE_NAME, run.quarantine),
    ):
        payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        write_atomic(run.config.output_dir / name, payload)


def _histogram(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _report(run: _Run, requests_total: int) -> NormalizeReport:
    """Run metadata comes from the journal, so a rerun reports the same numbers."""
    stamps = sorted(run.fetched, key=lambda text: (_parse_iso(text) or datetime.min, text))
    flags = [flag for row in run.occurrences for flag in row["quality_flags"]]
    flags += [flag for row in run.events for flag in row["quality_flags"]]
    flags += [flag for row in run.places for flag in row["quality_flags"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "normalizer_version": run.config.normalizer_version,
        "timerules_version": TIMERULES_VERSION,
        "normalizer_config_sha256": run.fingerprint,
        "windows": [list(span) for span in run.config.windows],
        "run_ids": sorted(run.run_ids),
        "first_fetched_at": stamps[0] if stamps else None,
        "last_fetched_at": stamps[-1] if stamps else None,
        "requests_total": requests_total,
        "requests_read": run.requests_read,
        "requests_skipped": run.requests_skipped,
        "events": len(run.events),
        "places": len(run.places),
        "event_projections": run.event_projections,
        "place_projections": run.place_projections,
        "events_without_detail": len(run.events_listed - run.delivered_events),
        "places_without_detail": len(run.places_referenced - set(run.place_details)),
        "observations": len(run.observations),
        "occurrences": len(run.occurrences),
        "quarantined": len(run.quarantine),
        "date_records": run.date_records,
        "date_records_resolved": run.date_records_resolved,
        "date_records_quarantined": run.date_records_quarantined,
        "reconciled": run.date_records
        == run.date_records_resolved + run.date_records_quarantined,
        "quarantine_reasons": _histogram([row["reason"] for row in run.quarantine]),
        "quality_flags": _histogram(flags),
        "time_precision": _histogram([row["time_precision"] for row in run.occurrences]),
        "schedule_basis": _histogram([row["schedule_basis"] for row in run.occurrences]),
    }

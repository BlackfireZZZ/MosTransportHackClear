"""Types, constants and canonicalisation shared by the KudaGo collector.

The contract is `external-events.v1`, specified in
`docs/external-events-kudago.md`.
Both halves of the collector — fetching and normalisation — depend only on this
module, so the delivery format has one definition.
"""

import hashlib
import json
from typing import Any, Literal, TypedDict
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "external-events.v1"
SOURCE = "kudago"
API_VERSION = "v1.4"
API_ROOT = "https://kudago.com/public-api/v1.4/"
DOCS_URL = "https://docs.kudago.com/api/"
MOSCOW = ZoneInfo("Europe/Moscow")

REQUESTS_NAME = "requests.jsonl"
EVENTS_NAME = "events.jsonl"
PLACES_NAME = "places.jsonl"
OCCURRENCES_NAME = "occurrences.jsonl"
OBSERVATIONS_NAME = "observations.jsonl"
QUARANTINE_NAME = "quarantine.jsonl"
MANIFEST_NAME = "manifest.json"
QUALITY_REPORT_NAME = "quality_report.json"
RAW_DIR = "raw"
DICTIONARIES_DIR = "dictionaries"

AvailabilityBasis = Literal["observed_snapshot", "archived_snapshot"]
ObjectType = Literal["event", "place"]
EventStatus = Literal["unknown", "scheduled", "cancelled", "postponed"]
CoordinateSource = Literal["place_api", "geocoded", "missing"]
CoordinatePrecision = Literal["entrance", "building", "venue_area", "unknown"]
TimePrecision = Literal["exact_interval", "start_only", "date_only", "unresolved"]
ScheduleBasis = Literal["explicit", "event_schedule", "place_schedule", "unresolved"]
RunStatus = Literal["complete", "partial", "unstable"]


class RequestRecord(TypedDict):
    """One attempt, successful or not. `request_id` is unique per attempt."""

    request_id: str
    run_id: str
    url: str
    requested_at: str
    fetched_at: str
    http_status: int | None
    attempt: int
    raw_path: str | None
    body_sha256: str | None
    error: str | None
    count: int | None
    next_url: str | None


class SnapshotCommon(TypedDict):
    schema_version: str
    source: str
    run_id: str
    request_id: str
    snapshot_id: str
    fetched_at: str
    available_at: str
    availability_basis: AvailabilityBasis
    availability_evidence: str | None
    quality_flags: list[str]
    payload: dict[str, Any]


class EventRecord(SnapshotCommon):
    event_id: str
    title: str
    source_url: str
    published_at: str | None
    location_slug: str | None
    place_id: str | None
    categories: list[str]
    tags: list[str]
    description: str | None
    body_text: str | None
    price_text: str | None
    age_restriction: str | None
    is_free: bool | None
    dates_raw: list[dict[str, Any]]
    status: EventStatus
    status_evidence: str | None


class PlaceRecord(SnapshotCommon):
    place_id: str
    title: str
    source_url: str
    address: str | None
    location_slug: str | None
    subway_text: str | None
    timetable_text: str | None
    lat: float | None
    lon: float | None
    coordinate_source: CoordinateSource
    coordinate_evidence: str | None
    coordinate_precision: CoordinatePrecision
    is_closed: bool | None
    is_stub: bool | None


class ObservationRecord(TypedDict):
    """One sighting of an object's state, even when the payload repeats.

    A card going A -> B -> A leaves three observations over two snapshots. Keeping
    only distinct content hashes would erase the return to A, and with it the fact
    that a postponement was reverted.
    """

    object_type: ObjectType
    event_or_place_id: str
    snapshot_id: str
    request_id: str
    fetched_at: str
    available_at: str
    availability_basis: AvailabilityBasis
    availability_evidence: str | None


class OccurrenceRecord(TypedDict):
    schema_version: str
    event_id: str
    event_snapshot_id: str
    derivation_id: str
    occurrence_id: str
    date_index: int
    expansion_index: int
    place_id: str | None
    place_snapshot_id: str | None
    raw_date: dict[str, Any]
    start_at: str | None
    end_at: str | None
    start_date: str | None
    end_date: str | None
    time_precision: TimePrecision
    schedule_basis: ScheduleBasis
    available_at: str
    quality_flags: list[str]


class QuarantineRecord(TypedDict):
    request_id: str | None
    event_id: str | None
    place_id: str | None
    date_index: int | None
    reason: str
    detail: str
    raw_ref: str | None


class FileEntry(TypedDict):
    path: str
    sha256: str
    bytes: int
    rows: int | None


class RunCounts(TypedDict):
    events: int
    places: int
    occurrences: int
    observations: int
    quarantine: int
    requests: int


class RunScope(TypedDict):
    location: str
    movies_included: bool
    category_filter: str | None
    months: list[str]


class Manifest(TypedDict):
    """Written last, and never listing itself in `files`."""

    schema_version: str
    run_id: str
    owner: str
    created_at: str
    source: str
    api_version: str
    docs_url: str
    timezone: str
    targets: list[list[str]]
    windows: list[list[str]]
    collector_version: str
    normalizer_version: str
    python_version: str
    normalizer_config_sha256: str
    scope: RunScope
    availability_policy: str
    status: RunStatus
    files: list[FileEntry]
    counts: RunCounts
    unresolved_failed_request_ids: list[str]


def canonical_bytes(payload: Any) -> bytes:
    """The exact serialisation the contract hashes.

    Arrays keep their order and numbers keep their type: re-sorting or coercing
    would make two genuinely different source payloads collide, and the snapshot id
    is what tells one version of a card from another.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_id(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def derivation_id(
    *,
    event_snapshot_id: str,
    place_snapshot_id: str | None,
    normalizer_version: str,
    normalizer_config_sha256: str,
) -> str:
    """Identifies the inputs a session was derived from.

    The expansion windows and the place version are part of it: the same card
    expanded over different windows, or against a changed venue timetable, is a
    different derivation even though the card itself did not move. The windows
    enter through `normalizer_config_sha256`, which is the fingerprint of the
    whole normaliser configuration.
    """
    return hashlib.sha256(
        canonical_bytes(
            {
                "event_snapshot_id": event_snapshot_id,
                "place_snapshot_id": place_snapshot_id,
                "normalizer_version": normalizer_version,
                "normalizer_config_sha256": normalizer_config_sha256,
            }
        )
    ).hexdigest()


def occurrence_id(*, derivation: str, date_index: int, expansion_index: int) -> str:
    return hashlib.sha256(
        f"{derivation}:{date_index}:{expansion_index}".encode()
    ).hexdigest()

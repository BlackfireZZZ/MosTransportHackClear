"""Manifest and quality report for a delivery directory.

The quality report is the point of the pilot. It answers, with numerators and
denominators rather than adjectives, how many sessions carry coordinates, an exact
start, an exact end and provable availability at a 2025 origin. A pilot that shows
most cards lack exact times is a successful pilot; a report that hides it is not.
"""

import hashlib
import json
import platform
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tramflow_ml.ingestion.checkpoint import write_atomic

from .records import (
    API_VERSION,
    DOCS_URL,
    EVENTS_NAME,
    MANIFEST_NAME,
    OBSERVATIONS_NAME,
    OCCURRENCES_NAME,
    PLACES_NAME,
    QUALITY_REPORT_NAME,
    QUARANTINE_NAME,
    REQUESTS_NAME,
    SCHEMA_VERSION,
    SOURCE,
    FileEntry,
    Manifest,
    RunCounts,
    RunScope,
    RunStatus,
    canonical_bytes,
)

FLAG_OUTSIDE_WINDOW = "outside_collection_window"
AVAILABILITY_POLICY = "observed_or_proven_archived_snapshot"
TIMEZONE = "Europe/Moscow"
HASH_BLOCK_BYTES = 1 << 20

PILOT_ORIGINS: tuple[str, ...] = (
    "2025-05-01T00:00:00+03:00",
    "2025-07-01T00:00:00+03:00",
    "2025-09-01T00:00:00+03:00",
    "2025-11-01T00:00:00+03:00",
)

DELIVERY_FILES: tuple[str, ...] = (
    REQUESTS_NAME,
    EVENTS_NAME,
    PLACES_NAME,
    OCCURRENCES_NAME,
    OBSERVATIONS_NAME,
    QUARANTINE_NAME,
    QUALITY_REPORT_NAME,
)

JSONL_FILES: frozenset[str] = frozenset(
    (
        REQUESTS_NAME,
        EVENTS_NAME,
        PLACES_NAME,
        OCCURRENCES_NAME,
        OBSERVATIONS_NAME,
        QUARANTINE_NAME,
    )
)


class DeliveryError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Ratio:
    """A share is never reported without the two numbers it came from."""

    numerator: int
    denominator: int

    def as_dict(self) -> dict[str, Any]:
        share = self.numerator / self.denominator if self.denominator else None
        return {"numerator": self.numerator, "denominator": self.denominator, "share": share}


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    yield parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_BLOCK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def inventory(directory: Path, names: Iterable[str]) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for name in sorted(names):
        path = directory / name
        if not path.exists():
            continue
        entries.append(
            {
                "path": name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": _count_lines(path) if name in JSONL_FILES else None,
            }
        )
    return entries


def _best_place_versions(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The most informative version of each venue, chosen rather than whichever came last.

    A venue can appear under several snapshots and only some carry coordinates, so
    taking the last row read would make coverage depend on file order.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        place_id = str(row.get("place_id", ""))
        if not place_id:
            continue
        current = best.get(place_id)
        if current is None or _place_rank(row) > _place_rank(current):
            best[place_id] = row
    return best


def _place_rank(row: Mapping[str, Any]) -> tuple[int, int]:
    has_coordinates = 1 if row.get("lat") is not None and row.get("lon") is not None else 0
    return (has_coordinates, len(row.get("payload") or {}))


def _origin_availability(
    occurrences: list[dict[str, Any]], origins: Iterable[str]
) -> dict[str, Any]:
    """How many sessions a 2025 origin may legitimately see.

    A snapshot taken in 2026 fails every 2025 origin however old its
    `publication_date` claims to be, which is the whole reason this table exists.
    """
    out: dict[str, Any] = {}
    for origin in origins:
        allowed = sum(1 for row in occurrences if str(row.get("available_at", "")) <= origin)
        out[origin] = {
            "allowed_by_availability": allowed,
            "retrospective_only": len(occurrences) - allowed,
        }
    return out


def build_quality_report(
    directory: Path,
    *,
    scope: RunScope,
    status: RunStatus,
    normalize_report: Mapping[str, Any],
    fetch_summary: Mapping[str, Any],
    origins: Iterable[str] = PILOT_ORIGINS,
) -> dict[str, Any]:
    occurrences = list(_read_jsonl(directory / OCCURRENCES_NAME))
    events = list(_read_jsonl(directory / EVENTS_NAME))
    places = _best_place_versions(_read_jsonl(directory / PLACES_NAME))

    # Coverage is measured over sessions inside the delivery window only. A record
    # whose repeats all fall outside it is kept as one flagged row by design, and
    # counting those in the denominator would report the window's shape as a data
    # quality problem -- one card carrying a decade of history would sink the score.
    in_window = [
        row for row in occurrences if FLAG_OUTSIDE_WINDOW not in (row.get("quality_flags") or [])
    ]
    outside_window = len(occurrences) - len(in_window)
    total = len(in_window)
    precision = Counter(str(row.get("time_precision")) for row in in_window)
    with_place = [row for row in in_window if row.get("place_id")]
    with_coords = [
        row
        for row in with_place
        if (place := places.get(str(row["place_id"]))) is not None and place.get("lat") is not None
    ]

    # Bucketed by the resolved moment, not by the raw `start_date`: an unresolved
    # row keeps whatever date the card carried, so counting those put sessions in
    # 2014 in a report about two months of 2025.
    months = Counter(str(row["start_at"])[:7] for row in in_window if row.get("start_at"))
    categories: Counter[str] = Counter()
    for event in events:
        for category in event.get("categories") or []:
            categories[str(category)] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "scope": dict(scope),
        "status": status,
        "collection": {
            "first_fetched_at": normalize_report.get("first_fetched_at"),
            "last_fetched_at": normalize_report.get("last_fetched_at"),
            "run_ids": normalize_report.get("run_ids", []),
        },
        "counts": {
            "events": normalize_report.get("events", 0),
            "places": normalize_report.get("places", 0),
            "occurrences": len(occurrences),
            "occurrences_in_window": total,
            "observations": normalize_report.get("observations", 0),
            "quarantine": normalize_report.get("quarantined", 0),
            "requests": normalize_report.get("requests_total", 0),
        },
        "reconciliation": {
            "date_records": normalize_report.get("date_records", 0),
            "resolved": normalize_report.get("date_records_resolved", 0),
            "quarantined": normalize_report.get("date_records_quarantined", 0),
            "reconciled": normalize_report.get("reconciled", False),
        },
        "coverage": {
            "sessions_with_place": Ratio(len(with_place), total).as_dict(),
            "sessions_with_coordinates": Ratio(len(with_coords), total).as_dict(),
            "exact_start": Ratio(
                precision["exact_interval"] + precision["start_only"], total
            ).as_dict(),
            "exact_interval": Ratio(precision["exact_interval"], total).as_dict(),
            "date_only": Ratio(precision["date_only"], total).as_dict(),
            "unresolved": Ratio(precision["unresolved"], total).as_dict(),
        },
        "time_precision": dict(precision),
        "sessions_outside_window": outside_window,
        "schedule_basis": dict(normalize_report.get("schedule_basis", {})),
        "quality_flags": dict(normalize_report.get("quality_flags", {})),
        "quarantine_reasons": dict(normalize_report.get("quarantine_reasons", {})),
        "by_month": dict(sorted(months.items())),
        "by_category": dict(sorted(categories.items())),
        "availability_by_origin": _origin_availability(in_window, origins),
        "fetch": {
            "status": fetch_summary.get("status"),
            "unresolved_failed_request_ids": list(fetch_summary.get("unresolved_request_ids", [])),
            "failures": list(fetch_summary.get("failures", [])),
        },
        "limitations": [
            "The API catalogue is not the whole city: closed events, deleted cards and "
            "unregistered venues may be absent, and the share of real events covered is "
            "unknown without an external registry.",
            "Card presence is not proof the event took place; status starts as unknown.",
            "Place details describe the venue now, not necessarily in 2025.",
            "No archived snapshots exist, so availability_basis is observed_snapshot "
            "throughout and no session is provably available before its fetch.",
            "schedule_basis=place_schedule is unreachable in this pilot: expanding a venue "
            "timetable would require confirming its structure and its 2025 version.",
            "coordinate_precision is venue_area at best; the source supplies no evidence "
            "for entrance or building precision.",
            "Cinema sessions are out of scope and are not counted anywhere in this report.",
        ],
    }


def build_manifest(
    directory: Path,
    *,
    run_id: str,
    owner: str,
    created_at: str,
    scope: RunScope,
    status: RunStatus,
    targets: list[list[str]],
    windows: list[list[str]],
    collector_version: str,
    normalizer_version: str,
    normalizer_config_sha256: str,
    counts: RunCounts,
    unresolved_failed_request_ids: list[str],
) -> Manifest:
    if status == "complete" and unresolved_failed_request_ids:
        raise DeliveryError("a complete run cannot carry unresolved failed requests")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "owner": owner,
        "created_at": created_at,
        "source": SOURCE,
        "api_version": API_VERSION,
        "docs_url": DOCS_URL,
        "timezone": TIMEZONE,
        "targets": targets,
        "windows": windows,
        "collector_version": collector_version,
        "normalizer_version": normalizer_version,
        "python_version": platform.python_version(),
        "normalizer_config_sha256": normalizer_config_sha256,
        "scope": scope,
        "availability_policy": AVAILABILITY_POLICY,
        "status": status,
        "files": inventory(directory, DELIVERY_FILES),
        "counts": counts,
        "unresolved_failed_request_ids": unresolved_failed_request_ids,
    }


def write_json(path: Path, payload: Any) -> None:
    write_atomic(path, canonical_bytes(payload) + b"\n")


def publish(
    directory: Path,
    *,
    quality_report: Mapping[str, Any],
    manifest_kwargs: Mapping[str, Any],
) -> Manifest:
    """Quality report first, manifest last so it can hash the report it describes."""
    write_json(directory / QUALITY_REPORT_NAME, dict(quality_report))
    manifest = build_manifest(directory, **manifest_kwargs)
    write_json(directory / MANIFEST_NAME, manifest)
    return manifest


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")

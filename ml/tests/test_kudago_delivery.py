"""The quality report must not flatter the data, and the manifest must not lie."""

import json
from pathlib import Path
from typing import Any

import pytest

from tramflow_ml.external.kudago.delivery import (
    DeliveryError,
    _best_place_versions,
    build_manifest,
    build_quality_report,
    inventory,
    publish,
)
from tramflow_ml.external.kudago.records import (
    MANIFEST_NAME,
    QUALITY_REPORT_NAME,
    RunCounts,
    RunScope,
)

SCOPE: RunScope = {
    "location": "msk",
    "movies_included": False,
    "category_filter": None,
    "months": ["2025-01"],
}
COUNTS: RunCounts = {
    "events": 1,
    "places": 1,
    "occurrences": 2,
    "observations": 1,
    "quarantine": 0,
    "requests": 3,
}
REPORT: dict[str, Any] = {
    "events": 1,
    "places": 1,
    "observations": 1,
    "quarantined": 0,
    "requests_total": 3,
    "date_records": 2,
    "date_records_resolved": 2,
    "date_records_quarantined": 0,
    "reconciled": True,
    "quarantine_reasons": {},
    "quality_flags": {},
    "schedule_basis": {"explicit": 1},
    "first_fetched_at": "2026-09-27T00:00:00Z",
    "last_fetched_at": "2026-09-27T00:00:01Z",
    "run_ids": ["run-1"],
    "normalizer_version": "n.v1",
    "normalizer_config_sha256": "0" * 64,
}
FETCH: dict[str, Any] = {"status": "partial", "unresolved_request_ids": [], "failures": []}


def _write(directory: Path, name: str, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _occurrence(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": "1",
        "place_id": "10",
        "start_date": "2025-01-15",
        "start_at": "2025-01-15T19:00:00+03:00",
        "time_precision": "exact_interval",
        "available_at": "2026-09-27T00:00:00Z",
        "quality_flags": [],
    }
    row.update(overrides)
    return row


def test_out_of_window_sessions_do_not_dilute_coverage(tmp_path: Path) -> None:
    """One card carrying a decade of history must not read as a quality problem.

    A record whose repeats all fall outside the window is kept as a single flagged
    row by design. Counting those in the denominator reported 1% exact intervals
    for data that was in fact 71%.
    """
    _write(
        tmp_path,
        "occurrences.jsonl",
        [_occurrence()]
        + [
            _occurrence(
                start_date="2015-07-04",
                time_precision="unresolved",
                quality_flags=["outside_collection_window"],
            )
            for _ in range(99)
        ],
    )
    _write(tmp_path, "places.jsonl", [{"place_id": "10", "lat": 55.7, "lon": 37.6, "payload": {}}])
    _write(tmp_path, "events.jsonl", [])

    report = build_quality_report(
        tmp_path, scope=SCOPE, status="partial", normalize_report=REPORT, fetch_summary=FETCH
    )

    assert report["counts"]["occurrences"] == 100
    assert report["counts"]["occurrences_in_window"] == 1
    assert report["sessions_outside_window"] == 99
    assert report["coverage"]["exact_interval"]["share"] == 1.0
    assert report["coverage"]["unresolved"]["numerator"] == 0


def test_every_share_carries_its_two_numbers(tmp_path: Path) -> None:
    _write(tmp_path, "occurrences.jsonl", [_occurrence()])
    _write(tmp_path, "places.jsonl", [])
    _write(tmp_path, "events.jsonl", [])

    report = build_quality_report(
        tmp_path, scope=SCOPE, status="partial", normalize_report=REPORT, fetch_summary=FETCH
    )

    for name, ratio in report["coverage"].items():
        assert set(ratio) == {"numerator", "denominator", "share"}, name
        assert ratio["denominator"] == 1


def test_a_2026_snapshot_is_retrospective_only_for_a_2025_origin(tmp_path: Path) -> None:
    """The whole point of the availability table: no card fetched in 2026 may feed 2025."""
    _write(tmp_path, "occurrences.jsonl", [_occurrence(available_at="2026-09-27T00:00:00Z")])
    _write(tmp_path, "places.jsonl", [])
    _write(tmp_path, "events.jsonl", [])

    report = build_quality_report(
        tmp_path, scope=SCOPE, status="partial", normalize_report=REPORT, fetch_summary=FETCH
    )

    for origin, counts in report["availability_by_origin"].items():
        assert counts["allowed_by_availability"] == 0, origin
        assert counts["retrospective_only"] == 1


def test_the_venue_version_chosen_is_the_one_with_coordinates() -> None:
    reference = {"place_id": "10", "title": "", "lat": None, "lon": None, "payload": {"id": 10}}
    full = {
        "place_id": "10",
        "title": "Театр",
        "lat": 55.77,
        "lon": 37.61,
        "payload": dict.fromkeys(str(i) for i in range(24)),
    }

    assert _best_place_versions([full, reference])["10"]["lat"] == 55.77
    assert _best_place_versions([reference, full])["10"]["lat"] == 55.77


def test_a_complete_run_cannot_carry_unresolved_failures(tmp_path: Path) -> None:
    with pytest.raises(DeliveryError):
        build_manifest(
            tmp_path,
            run_id="run-1",
            owner="owner",
            created_at="2026-09-27T00:00:00Z",
            scope=SCOPE,
            status="complete",
            targets=[["2025-01-01T00:00:00+03:00", "2025-02-01T00:00:00+03:00"]],
            windows=[["2024-12-31T00:00:00+03:00", "2025-02-02T00:00:00+03:00"]],
            collector_version="0.1.0",
            normalizer_version="n.v1",
            normalizer_config_sha256="0" * 64,
            counts=COUNTS,
            unresolved_failed_request_ids=["req-9"],
        )


def test_the_manifest_hashes_the_report_and_never_itself(tmp_path: Path) -> None:
    _write(tmp_path, "occurrences.jsonl", [_occurrence()])
    _write(tmp_path, "places.jsonl", [])
    _write(tmp_path, "events.jsonl", [])
    quality = build_quality_report(
        tmp_path, scope=SCOPE, status="partial", normalize_report=REPORT, fetch_summary=FETCH
    )

    manifest = publish(
        tmp_path,
        quality_report=quality,
        manifest_kwargs={
            "run_id": "run-1",
            "owner": "owner",
            "created_at": "2026-09-27T00:00:00Z",
            "scope": SCOPE,
            "status": "partial",
            "targets": [["2025-01-01T00:00:00+03:00", "2025-02-01T00:00:00+03:00"]],
            "windows": [["2024-12-31T00:00:00+03:00", "2025-02-02T00:00:00+03:00"]],
            "collector_version": "0.1.0",
            "normalizer_version": "n.v1",
            "normalizer_config_sha256": "0" * 64,
            "counts": COUNTS,
            "unresolved_failed_request_ids": [],
        },
    )

    listed = {entry["path"] for entry in manifest["files"]}
    assert MANIFEST_NAME not in listed
    assert QUALITY_REPORT_NAME in listed
    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_inventory_counts_rows_only_for_jsonl(tmp_path: Path) -> None:
    _write(tmp_path, "occurrences.jsonl", [_occurrence(), _occurrence()])
    (tmp_path / QUALITY_REPORT_NAME).write_text("{}", encoding="utf-8")

    listed = inventory(tmp_path, ["occurrences.jsonl", QUALITY_REPORT_NAME])
    entries = {entry["path"]: entry for entry in listed}

    assert entries["occurrences.jsonl"]["rows"] == 2
    assert entries[QUALITY_REPORT_NAME]["rows"] is None


def test_by_month_counts_resolved_sessions_not_raw_card_dates(tmp_path: Path) -> None:
    """An unresolved row keeps its card's date, which can sit years outside the window."""
    _write(
        tmp_path,
        "occurrences.jsonl",
        [
            _occurrence(),
            _occurrence(
                occurrence_id="unresolved-1",
                start_at=None,
                start_date="2014-08-01",
                time_precision="unresolved",
            ),
        ],
    )
    _write(tmp_path, "places.jsonl", [])
    _write(tmp_path, "events.jsonl", [])

    report = build_quality_report(
        tmp_path, scope=SCOPE, status="partial", normalize_report=REPORT, fetch_summary=FETCH
    )

    assert report["by_month"] == {"2025-01": 1}

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.service_policy import load_service_policy


def notice(
    identity: str, routes: list[str], start: str, end: str | None, recurrence: str,
) -> dict[str, Any]:
    return {"id": identity, "routes": routes, "effective_start": start,
            "effective_end_exclusive": end, "recurrence": recurrence,
            "end_unknown": end is None and recurrence != "restoration",
            "url": "https://transport.mos.ru/mostrans/all_news/" + identity}


def registry(tmp_path: Path, notices: list[dict[str, Any]]) -> Path:
    snapshot = tmp_path / "notice.html"
    snapshot.write_text("fixture official source snapshot")
    values = [
        {**n, "source_path": str(snapshot), "source_sha256": digest(snapshot)} for n in notices
    ]
    path = tmp_path / "policy.json"
    write_json(path, {"schema_version": "boarding-service-notices-v1", "timezone": "Europe/Moscow",
                      "complete_historical_calendar": False, "notices": values})
    return path


def test_only_proven_matching_weekend_warning_is_restored_and_later_closure_wins(
    tmp_path: Path,
) -> None:
    policy = load_service_policy(registry(tmp_path, [
        notice("telegram20825", ["17"], "2025-04-19", None, "restoration"),
        notice("124110", ["17"], "2025-04-26", "2025-05-01", "weekends"),
        notice("7177", ["12"], "2025-06-21", None, "restoration"),
    ]))
    for day, expected in [("2025-04-12", False), ("2025-04-19", True),
                          ("2025-04-26", False), ("2025-04-28", True), ("2025-05-03", True)]:
        result = policy.evaluate("17", datetime.fromisoformat(day + "T12:00:00+03:00"),
                                 engine_applicable=False,
                                 warnings=("weekend_medvedkovo_closure_end_unknown",), soft=True)
        assert result.eligible is expected
    for day, expected in [("2025-06-14", False), ("2025-06-21", True)]:
        result = policy.evaluate("12", datetime.fromisoformat(day + "T12:00:00+03:00"),
                                 engine_applicable=False,
                                 warnings=("weekend_entuziastov_diversion_end_unknown",), soft=True)
        assert result.eligible is expected
    instant = datetime.fromisoformat("2025-06-22T12:00:00+03:00")
    assert not policy.evaluate("12", instant, engine_applicable=False,
                               warnings=("unknown_service_variant",), soft=True).eligible
    mixed = ("weekend_entuziastov_diversion_end_unknown",
             "incomplete_timetable_coverage_clock_evidence_disabled")
    assert policy.evaluate("12", instant, engine_applicable=False,
                           warnings=mixed, soft=True).eligible
    assert not policy.evaluate("12", instant, engine_applicable=False,
                               warnings=(*mixed, "unknown_service_variant"), soft=True).eligible
    assert not policy.evaluate("12", instant, engine_applicable=False,
                               warnings=("weekend_entuziastov_diversion_end_unknown",),
                               soft=False).eligible


@pytest.mark.parametrize("stamp,expected", [
    ("2025-07-18T00:00", True), ("2025-07-18T22:00", True),
    ("2025-07-18T23:00", False), ("2025-07-19T04:00", False),
    ("2025-07-19T05:00", True), ("2025-07-23T00:00", False),
    ("2025-07-23T23:00", True), ("2025-07-24T00:00", True),
])
def test_night_windows_anchor_previous_evening_and_restore_end_exclusively(
    tmp_path: Path, stamp: str, expected: bool,
) -> None:
    policy = load_service_policy(registry(tmp_path, [notice(
        "125448", ["11"], "2025-07-18T23:30:00+03:00", "2025-07-23T23:30:00+03:00",
        "night_after_23_30",
    )]))
    assert policy.evaluate("11", datetime.fromisoformat(stamp + "+03:00"),
                           engine_applicable=True, warnings=(), soft=True).eligible is expected


@pytest.mark.parametrize("stamp,expected", [
    ("2025-06-27T10:00", True), ("2025-06-27T11:00", False),
    ("2025-06-28T07:00", False), ("2025-06-28T08:00", True),
])
def test_continuous_partial_hour_and_approximate_end_are_conservative(
    tmp_path: Path, stamp: str, expected: bool,
) -> None:
    policy = load_service_policy(registry(tmp_path, [notice(
        "7223", ["11", "17", "25"], "2025-06-27T11:30:00+03:00",
        "2025-06-28T07:00:00+03:00", "continuous",
    )]))
    assert policy.evaluate("17", datetime.fromisoformat(stamp + "+03:00"),
                           engine_applicable=True, warnings=(), soft=True).eligible is expected


def test_modeled_short_route_exception_is_specific_to_seven_and_warning(tmp_path: Path) -> None:
    policy = load_service_policy(registry(tmp_path, [notice(
        "7319", ["7", "50"], "2025-07-10", "2025-08-11", "daily",
    )]))
    instant = datetime.fromisoformat("2025-07-29T12:00:00+03:00")
    warning = ("short_turn_geometry_from_ordered_base_and_official_endpoints",)
    assert policy.evaluate("7", instant, engine_applicable=True,
                           warnings=warning, soft=True).eligible
    assert not policy.evaluate("50", instant, engine_applicable=True,
                               warnings=warning, soft=True).eligible
    assert not policy.evaluate("7", instant, engine_applicable=True,
                               warnings=(), soft=True).eligible


def test_unknown_start_precaution_does_not_claim_confirmed_closure(tmp_path: Path) -> None:
    policy = load_service_policy(registry(tmp_path, [
        notice("126575", ["26"], "2025-10-01", None, "restoration"),
        notice("126404", ["11", "17", "25"], "2025-09-19", None, "restoration"),
    ]))
    for route, stamp, expected in [
        ("26", "2025-09-29T00:00", False), ("26", "2025-09-28T00:00", True),
        ("26", "2025-10-01T00:00", True), ("26", "2025-09-29T05:00", True),
        ("17", "2025-09-18T00:00", False), ("17", "2025-09-19T00:00", True),
    ]:
        result = policy.evaluate(route, datetime.fromisoformat(stamp + "+03:00"),
                                 engine_applicable=True, warnings=(), soft=True)
        assert result.eligible is expected
        if not expected:
            assert any(r.startswith("unknown_start_precaution:") for r in result.reasons)


def test_midnight_twenty_minutes_and_headway_unknown_end(tmp_path: Path) -> None:
    policy = load_service_policy(registry(tmp_path, [
        notice("126461", ["11"], "2025-09-24T00:20:00+03:00", None, "night_after_00_20"),
        notice("126476", ["28"], "2025-09-24", None, "daily"),
    ]))
    for route, stamp, expected in [
        ("11", "2025-09-23T00:00", True), ("11", "2025-09-24T00:00", False),
        ("11", "2025-09-24T05:00", True), ("28", "2025-10-31T14:00", False),
    ]:
        assert policy.evaluate(route, datetime.fromisoformat(stamp + "+03:00"),
                               engine_applicable=True, warnings=(), soft=True).eligible is expected


def test_missing_restoration_never_releases_engine_warning_and_tamper_fails(tmp_path: Path) -> None:
    path = registry(tmp_path, [notice("7319", ["7"], "2025-07-10", "2025-08-11", "daily")])
    policy = load_service_policy(path)
    assert not policy.evaluate("12", datetime.fromisoformat("2025-08-01T12:00:00+03:00"),
                               engine_applicable=False,
                               warnings=("weekend_entuziastov_diversion_end_unknown",),
                               soft=True).eligible
    (tmp_path / "notice.html").write_text("tampered")
    with pytest.raises(ValueError, match="snapshot missing or changed"):
        load_service_policy(path)


def test_unknown_recurrence_and_relative_snapshots_fail_closed(tmp_path: Path) -> None:
    path = registry(tmp_path, [notice("n", ["1"], "2025-01-01", None, "daily")])
    content = json.loads(path.read_text())
    content["notices"][0]["source_path"] = "notice.html"
    write_json(path, content)
    assert load_service_policy(path, source_root=tmp_path)
    content["notices"][0]["recurrence"] = "sometimes"
    write_json(path, content)
    with pytest.raises(ValueError, match="recurrence"):
        load_service_policy(path, source_root=tmp_path)

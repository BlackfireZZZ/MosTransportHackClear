import gzip
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.boarding import real
from tramflow_ml.boarding.composition import RouteEvidence
from tramflow_ml.boarding.real import RealConfig, _session, burst_groups, session_keys
from tramflow_ml.boarding.sequence import CycleTemplate


def evidence(lengths: tuple[float, ...] = (200, 200, 200, 200)) -> RouteEvidence:
    visits = tuple(
        {
            "stop_id": str(i),
            "name": f"Stop {i}",
            "lat": 55.7,
            "lon": 37.6 + i / 100,
            "pattern_id": "p",
            "direction": str(i // 2),
            "pattern_sequence": i,
            "boardable": True,
        }
        for i in range(len(lengths))
    )
    template = CycleTemplate(
        "1",
        tuple("p" for _ in lengths),
        tuple(v["stop_id"] for v in visits),
        tuple(v["direction"] for v in visits),
        lengths,
        tuple(False for _ in lengths),
        "synthetic:real-pipeline-test",
    )
    return RouteEvidence(
        template, visits, tuple(() for _ in lengths), tuple("absent" for _ in lengths), (), (), True
    )


def payments(seconds: list[int], day: str = "2025-01-15") -> Any:
    origin = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("Europe/Moscow"))
    return pd.DataFrame(
        [
            {
                "event_key": f"fixture:{i}",
                "event_at": (origin + timedelta(seconds=second)).strftime("%Y-%m-%d %H:%M:%S"),
                "second": origin.timestamp() + second,
                "route": "1",
                "device_key": "d1",
                "vehicle_key": "v1",
                "exit_key": "e1",
                "success": "1",
                "core": True,
            }
            for i, second in enumerate(seconds)
        ]
    )


def test_burst_gap_and_total_span_boundaries() -> None:
    times = np.array([0, 30, 60, 90, 91, 121, 152], dtype=float)
    assert burst_groups(times, 30, 90).tolist() == [0, 0, 0, 0, 1, 1, 2]
    assert burst_groups(np.array([0, 0, 0]), 30, 90).tolist() == [0, 0, 0]


def test_symmetric_route_never_gets_arbitrary_hard_stop_labels() -> None:
    frame = session_keys(payments([0, 60, 120, 180, 240]))
    rows, summary = _session(frame, evidence(), RealConfig(("2025-01-15",)))
    assert len(rows) == 5
    assert all(row["stop_id"] is None and not row["weak_label_eligible"] for row in rows)
    assert {row["assignment_status"] for row in rows} == {"ambiguous"}
    assert {row["reason"] for row in rows} == {"symmetry_unidentifiable"}
    assert all(row["conditional_posterior"] == pytest.approx(0.25) for row in rows)
    assert summary["source_success"] == 5
    for burst in summary["bursts"]:
        assert sum(a["weight"] for a in burst["alternatives"]) == pytest.approx(0.75)
        assert burst["unresolved_weight"] == pytest.approx(0.25)
        assert burst["calibrated"] is False


def test_soft_mass_uses_each_original_hour_when_one_burst_straddles_boundary() -> None:
    frame = session_keys(payments([3590, 3610, 3670, 3730, 3790]))
    rows, summary = _session(frame, evidence(), RealConfig(("2025-01-15",)))
    assert rows[0]["burst"] == rows[1]["burst"]
    assert rows[0]["event_at"] == "2025-01-15 00:59:50"
    assert rows[1]["event_at"] == "2025-01-15 01:00:10"
    soft = pd.DataFrame(summary["soft_rows"])
    assert soft.groupby("event_hour").expected_count.sum().to_dict() == pytest.approx(
        {"2025-01-15 00": 1, "2025-01-15 01": 4}
    )
    assert summary["bursts"][0]["core_events"] == 2
    assert soft.expected_count.sum() == pytest.approx(len(rows))


def test_same_vehicle_on_two_dates_has_distinct_reproducible_session_ids() -> None:
    summaries = []
    for day in ("2025-01-15", "2025-01-16", "2025-01-15"):
        _, summary = _session(session_keys(payments([0, 60], day)), evidence(), RealConfig((day,)))
        summaries.append(summary)
    assert summaries[0]["session_id"] != summaries[1]["session_id"]
    assert summaries[0]["session_id"] == summaries[2]["session_id"]


def test_reset_fragments_cannot_borrow_day_level_minimum_support() -> None:
    frame = session_keys(payments([0, 60, 6000, 6060]))
    config = RealConfig(("2025-01-15",), min_posterior=0.01, min_scenario_agreement=0.25)
    rows, summary = _session(frame, evidence((200, 500, 1000, 1500)), config)
    assert summary["gap_resets"] == 1
    assert summary["observed_payment_groups"] == 4
    assert {b["connected_segment_groups"] for b in summary["bursts"]} == {2}
    assert all(row["conditional_posterior"] > 0.5 for row in rows)
    assert {row["reason"] for row in rows} == {"sparse_or_scenario_sensitive"}
    assert not any(row["weak_label_eligible"] for row in rows)


def test_identity_device_vehicle_conflict_separates_groups_and_blocks_labels() -> None:
    frame = payments([0, 60, 120, 180])
    frame.loc[2:, "vehicle_key"] = "v2"
    grouped = session_keys(frame)
    assert grouped.identity_conflict.all()
    assert grouped.identity_kind.eq("device_only").all()
    assert grouped.group.nunique() == 2
    assert "group" not in frame.columns
    for _, part in grouped.groupby("group"):
        rows, _ = _session(part, evidence(), RealConfig(("2025-01-15",)))
        assert all(row["reason"] == "identity_conflict" for row in rows)
        assert not any(row["weak_label_eligible"] for row in rows)


def test_identity_vehicle_route_exit_conflict_and_missing_id_are_not_pooled() -> None:
    frame = payments([0, 60, 120, 180])
    frame.loc[1, "exit_key"] = "e2"
    frame.loc[2:, ["device_key", "vehicle_key"]] = ""
    grouped = session_keys(frame)
    assert grouped.identity_conflict.tolist() == [True, True, False, False]
    assert grouped.identity_kind.tolist() == [
        "device_only",
        "device_only",
        "identity_missing",
        "identity_missing",
    ]
    assert grouped.group.nunique() == 4


def test_compatible_devices_pool_by_vehicle_without_claiming_observed_identity() -> None:
    frame = payments([0, 60])
    frame.loc[1, "device_key"] = "d2"
    grouped = session_keys(frame)
    assert grouped.group.nunique() == 1
    assert not grouped.identity_conflict.any()
    assert grouped.identity_kind.eq("inferred_vehicle_pool").all()


def test_identity_conflict_blocks_otherwise_eligible_timing_candidate() -> None:
    frame = session_keys(payments([0, 60, 180, 400, 460, 580, 800]))
    config = RealConfig(("2025-01-15",), min_posterior=0.01, min_scenario_agreement=0.25)
    route = evidence((200, 500, 1000, 1500))
    baseline, _ = _session(frame, route, config)
    assert all(row["weak_label_eligible"] for row in baseline)
    frame["identity_conflict"] = True
    conflicted, _ = _session(frame, route, config)
    assert all(row["reason"] == "identity_conflict" for row in conflicted)
    assert not any(row["weak_label_eligible"] for row in conflicted)
    assert all(row["stop_id"] is not None for row in conflicted)


@pytest.mark.parametrize(
    "changes",
    [
        {"dates": ()},
        {"dates": ("2025-01-15", "2025-01-15")},
        {"dates": ("bad",)},
        {"routes": ()},
        {"routes": ("1", "1")},
        {"gap_seconds": 0},
        {"max_span_seconds": 5},
        {"gap_seconds": float("nan")},
        {"context_seconds": 1800},
        {"min_posterior": 0},
        {"min_scenario_agreement": 1.1},
        {"max_stationary_seconds": float("inf")},
    ],
)
def test_invalid_real_configuration_fails_before_source_read(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(RealConfig(("2025-01-15",)), **changes)


def source_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any) -> Path:
    def read_source(_path: Path, day: str, _routes: tuple[str, ...]) -> Any:
        assert day == "2025-01-15"
        return frame.copy()

    def compose(_patterns: Any, _schedules: Any, route: str, _day: Any) -> RouteEvidence:
        if route != "1":
            raise ValueError("fixture route has no ordered pattern")
        return evidence()

    def historical(_source: Path, _graph: Path, instant: datetime) -> list[Any]:
        assert instant.isoformat() == "2025-01-14T21:00:00+00:00"
        return []

    monkeypatch.setattr(real, "read_day", read_source)
    monkeypatch.setattr(real, "load_historical_catalog", historical)
    monkeypatch.setattr(real, "compose_route", compose)
    schedules = tmp_path / "schedules.json"
    schedules.write_text('{"sources": []}')
    return schedules


def test_reconstruct_day_conserves_original_hour_with_rejected_and_unsupported_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = payments([3590, 3610, 3670, 3730, 3790, 3800, 3900])
    frame.loc[5, "success"] = "0"
    frame.loc[6, "route"] = "99"
    frame.loc[6, "vehicle_key"] = "v99"
    frame.loc[6, "device_key"] = "d99"
    schedules = source_fixture(monkeypatch, tmp_path, frame)
    out = tmp_path / "result"
    report = real.reconstruct_day(
        tmp_path,
        tmp_path,
        tmp_path,
        schedules,
        "2025-01-15",
        RealConfig(("2025-01-15",), routes=("1", "99")),
        out,
    )
    assert report["source_success"] == 6
    assert report["source_rejected"] == 1
    assert report["ambiguous"] == 5 and report["unassigned"] == 1
    assert report["assigned_weak"] == 0
    assert report["soft_expected_count"] == pytest.approx(5)
    assert report["real_stop_accuracy"] == "unverified"
    events = pd.read_csv(out / "event_assignments.csv.gz", dtype=str, keep_default_na=False)
    assert len(events) == 6 and events.event_key.is_unique
    assert "fixture:5" not in set(events.event_key)
    unsupported = events.loc[events.route.eq("99")].iloc[0]
    assert unsupported.assignment_status == "unassigned" and unsupported.stop_id == ""
    soft = pd.read_csv(out / "stop_hour_soft_counts.csv.gz")
    assert soft.groupby("event_hour").expected_count.sum().to_dict() == pytest.approx(
        {"2025-01-15 00": 1, "2025-01-15 01": 4}
    )
    with gzip.open(out / "burst_candidates.jsonl.gz", "rt") as f:
        bursts = [json.loads(line) for line in f]
    assert sum(b["core_events"] for b in bursts) == 5
    for burst in bursts:
        assert (
            sum(a["weight"] for a in burst["alternatives"]) + burst["unresolved_weight"]
        ) == pytest.approx(1)
    assert pd.read_csv(out / "stop_hour_weak_counts.csv").empty


def test_reconstruct_day_all_unavailable_preserves_unassigned_mass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = payments([0, 60])
    frame["route"] = "99"
    schedules = source_fixture(monkeypatch, tmp_path, frame)
    report = real.reconstruct_day(
        tmp_path,
        tmp_path,
        tmp_path,
        schedules,
        "2025-01-15",
        RealConfig(("2025-01-15",), routes=("99",)),
        tmp_path / "out",
    )
    assert report["unassigned"] == report["source_success"] == 2
    assert report["soft_expected_count"] == 0


def test_reconstruct_day_rejects_duplicate_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = payments([0, 60])
    frame["event_key"] = "duplicate"
    schedules = source_fixture(monkeypatch, tmp_path, frame)
    with pytest.raises(ValueError, match="duplicate"):
        real.reconstruct_day(
            tmp_path,
            tmp_path,
            tmp_path,
            schedules,
            "2025-01-15",
            RealConfig(("2025-01-15",)),
            tmp_path / "out",
        )

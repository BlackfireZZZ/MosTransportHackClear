import json
from dataclasses import replace
from typing import Any

import pytest

from tramflow_ml.boarding.composition import RouteEvidence
from tramflow_ml.boarding.schedule_intervals import estimate_schedule_edges
from tramflow_ml.boarding.sequence import CycleTemplate


def evidence(a: tuple[int, ...], b: tuple[int, ...], length: float = 600) -> RouteEvidence:
    return RouteEvidence(
        CycleTemplate("7", ("p", "p"), ("a", "b"), ("0", "0"), (length, 0), (False, True), "test"),
        ({"boardable": True}, {"boardable": True}),
        (a, b),
        ("current_proxy_transfer", "current_proxy_transfer"),
        (),
        (),
        True,
    )


def irregular() -> tuple[int, ...]:
    return (3600, 4020, 4560, 4980, 5700, 6240, 7080, 7560, 8280, 8820, 9780, 10320)


def edge(value: RouteEvidence) -> dict[str, Any]:
    return estimate_schedule_edges(value)["edges"][0]  # type: ignore[no-any-return]


def test_unique_shift_is_accepted_without_claiming_trip_identity() -> None:
    a = irregular()
    result = estimate_schedule_edges(evidence(a, tuple(v + 120 for v in a)))
    first = result["edges"][0]
    assert first["accepted"] and first["seconds"] == 120
    assert first["coverage"] == 1
    assert first["uncertainty_seconds"] >= 30
    assert not first["historical_exact_day"]
    assert not result["trip_identity_verified"] and not result["historical_validity_verified"]
    assert result["accepted_edge_count"] == 1
    assert result["edges"][1]["reason"] == "terminal_connector"
    json.dumps(result, allow_nan=False)


def test_missing_departures_are_not_nth_paired() -> None:
    a = irregular()
    b = tuple(v + 120 for i, v in enumerate(a) if i not in (2, 8))
    result = edge(evidence(a, b))
    assert result["accepted"] and result["seconds"] == 120
    assert result["matched_departures"] == 10
    assert result["coverage"] == pytest.approx(10 / 12)


def test_regular_headway_alias_is_rejected_even_with_perfect_matching() -> None:
    a = tuple(3600 + 300 * i for i in range(50))
    result = edge(evidence(a, tuple(v + 120 for v in a)))
    assert not result["accepted"] and result["seconds"] is None
    assert result["reason"] == "periodic_or_competing_lag_alias"
    assert len(result["candidate_aliases"]) >= 2


def test_supported_longer_shift_wins_over_shorter_clock_peak() -> None:
    a = tuple(3600 + 1000 * i + 15 * i * i for i in range(12))
    b = tuple(sorted([v + 180 for v in a] + [v + 60 for v in a[:5]]))
    result = edge(evidence(a, b))
    assert result["accepted"] and result["seconds"] == 180
    assert any(p["seconds"] == 60 for p in result["candidate_aliases"])


def test_midnight_shift_preserves_one_to_one_matching() -> None:
    a = tuple(sorted((v + 81000) % 86400 for v in irregular()))
    b = tuple(sorted((v + 120) % 86400 for v in a))
    result = edge(evidence(a, b))
    assert result["accepted"] and result["seconds"] == 120
    assert result["matched_departures"] == len(a)
    assert result["temporal_half_coverage"] == [1, 1]


def test_unequal_service_half_coverage_is_rejected() -> None:
    a = irregular()
    b = tuple(v + 120 for v in a[:8])
    result = edge(evidence(a, b))
    assert not result["accepted"]
    assert result["reason"] == "inconsistent_temporal_half_coverage"


def test_drifting_halves_are_rejected() -> None:
    a = irregular()
    b = tuple(v + (100 if i < 6 else 140) for i, v in enumerate(a))
    result = edge(evidence(a, b))
    assert not result["accepted"]
    assert result["reason"] == "inconsistent_temporal_half_duration"


@pytest.mark.parametrize("invalid", [-1, 86400, float("nan"), float("inf"), True, "120"])
def test_invalid_departures_cannot_produce_estimates(invalid: Any) -> None:
    a = irregular()
    result = edge(evidence(a, (*a[:-1], invalid)))
    assert result["reason"] == "invalid_departure_clocks"
    assert result["seconds"] is None


def test_duplicate_clocks_cannot_fabricate_support() -> None:
    result = edge(evidence((3600,) * 20, (3720,) * 20))
    assert result["reason"] == "invalid_departure_clocks"


def test_sparse_and_absent_schedules_are_rejected() -> None:
    assert edge(evidence((3600,), (3720,)))["reason"] == "insufficient_departures"
    assert edge(evidence((), ()))["reason"] == "absent_schedule"


def test_nonboardable_and_mixed_source_boundaries_are_rejected() -> None:
    a = irregular()
    value = evidence(a, tuple(v + 120 for v in a))
    assert edge(replace(value, visits=({"boardable": True}, {"boardable": False})))["reason"] == (
        "nonboardable_boundary"
    )
    mixed = replace(value, schedule_kinds=("historical_exact_day", "current_proxy_transfer"))
    assert edge(mixed)["reason"] == "incompatible_schedule_kinds"


@pytest.mark.parametrize("length", [0, -10, float("nan"), float("inf")])
def test_invalid_distances_are_rejected(length: float) -> None:
    assert edge(evidence(irregular(), irregular(), length))["reason"] == "invalid_edge_length"


def test_impossible_speed_has_no_accepted_duration() -> None:
    a = tuple(3600 + 3600 * i for i in range(12))
    result = edge(evidence(a, tuple(v + 30 for v in a), 5000))
    assert not result["accepted"] and result["reason"] == "no_physically_feasible_lag"


def test_malformed_shape_fails_loudly() -> None:
    with pytest.raises(ValueError, match="equal nonzero"):
        estimate_schedule_edges(replace(evidence((), ()), departures=((),)))


def test_unknown_schedule_provenance_is_rejected() -> None:
    a = irregular()
    value = replace(evidence(a, a), schedule_kinds=("unknown", "unknown"))
    assert edge(value)["reason"] == "unsupported_schedule_kind"


def test_departure_budget_is_checked_before_pairing() -> None:
    a = tuple(range(3000))
    result = edge(evidence(a, a))
    assert result["reason"] == "invalid_departure_clocks"
    assert "budget" in result["error"]


def test_match_tolerance_cannot_bypass_physical_speed_bound() -> None:
    a = tuple(3600 + 3600 * i for i in range(12))
    result = edge(evidence(a, tuple(v + 30 for v in a), 900))
    assert not result["accepted"]

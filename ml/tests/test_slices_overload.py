"""Overload is a fraction of capacity, so it is absent unless both halves are real.

The repository has an open defect in the other direction: `ForecastPointModel.capacity`
defaults to `180.0` against a response schema demanding `gt=0`, so an unknown capacity
is served as a fabricated number. Nothing in this module has a default capacity.
"""

from datetime import datetime, timedelta

import pytest

from tramflow_ml.features import MOSCOW, EntityKey
from tramflow_ml.slices import (
    LOAD_TARGET,
    LOAD_UNIT,
    ScoredPoint,
    SliceError,
    SliceKey,
    build_report,
)

OVERALL = SliceKey("overall", "all")
START = datetime(2026, 3, 2, 10, tzinfo=MOSCOW)
SAMPLES = 12


def points(*, target, unit, capacity, actual=250.0, predicted=150.0):
    return [
        ScoredPoint(
            entity=EntityKey("route-A", "dir-N", "stop-1"),
            horizon="day",
            bucket_start=START + timedelta(hours=index),
            fold_id=f"f{index % 2}",
            target=target,
            unit=unit,
            event=None,
            actual=actual,
            predicted=predicted,
            baseline=None,
            interval=None,
            capacity=capacity,
        )
        for index in range(SAMPLES)
    ]


def test_counts_cannot_produce_an_overload_metric() -> None:
    metrics = build_report(
        points(target="synthetic_boardings", unit="event_count", capacity=None)
    ).by_key[OVERALL]

    assert metrics.overload is None
    assert metrics.to_dict()["overload"] is None
    assert LOAD_TARGET in metrics.overload_absent_reason
    assert "not an occupancy" in metrics.overload_absent_reason


def test_a_flow_in_passenger_units_is_still_not_an_occupancy() -> None:
    metrics = build_report(
        points(target="boarding_count", unit=LOAD_UNIT, capacity=200.0)
    ).by_key[OVERALL]

    assert metrics.overload is None
    assert "not an occupancy" in metrics.overload_absent_reason


def test_an_unknown_capacity_produces_absence_not_a_guess() -> None:
    metrics = build_report(
        points(target=LOAD_TARGET, unit=LOAD_UNIT, capacity=None)
    ).by_key[OVERALL]

    assert metrics.overload is None
    assert metrics.overload_absent_reason == (
        f"capacity is unknown for {SAMPLES} of {SAMPLES} points; overload is not "
        "computed against a guessed capacity"
    )


def test_one_missing_capacity_removes_the_metric_for_the_whole_slice() -> None:
    partial = points(target=LOAD_TARGET, unit=LOAD_UNIT, capacity=200.0)
    partial[0] = ScoredPoint(
        **{
            **{name: getattr(partial[0], name) for name in partial[0].__slots__},
            "capacity": None,
        }
    )

    metrics = build_report(partial).by_key[OVERALL]

    assert metrics.overload is None
    assert f"capacity is unknown for 1 of {SAMPLES} points" in metrics.overload_absent_reason


def test_a_known_capacity_on_a_load_target_produces_the_metric() -> None:
    metrics = build_report(
        points(target=LOAD_TARGET, unit=LOAD_UNIT, capacity=200.0)
    ).by_key[OVERALL]

    assert metrics.overload is not None
    assert metrics.overload.samples == SAMPLES
    assert metrics.overload.actual_overloaded == SAMPLES
    assert metrics.overload.predicted_overloaded == 0
    assert metrics.overload.actual_rate == 1.0
    assert metrics.overload.predicted_rate == 0.0
    assert metrics.overload_absent_reason == ""


def test_a_capacity_at_the_boundary_is_not_an_overload() -> None:
    metrics = build_report(
        points(target=LOAD_TARGET, unit=LOAD_UNIT, capacity=200.0, actual=200.0, predicted=201.0)
    ).by_key[OVERALL]

    assert metrics.overload is not None
    assert metrics.overload.actual_overloaded == 0
    assert metrics.overload.predicted_overloaded == SAMPLES


@pytest.mark.parametrize("capacity", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_capacity_is_refused_at_the_point(capacity: float) -> None:
    with pytest.raises(SliceError, match="capacity"):
        points(target=LOAD_TARGET, unit=LOAD_UNIT, capacity=capacity)

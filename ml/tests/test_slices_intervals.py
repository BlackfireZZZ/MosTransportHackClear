"""A giant interval has perfect coverage and no information; the score must say so."""

from datetime import datetime, timedelta

import pytest

from tramflow_ml.features import MOSCOW, EntityKey
from tramflow_ml.slices import (
    MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL,
    GateThresholds,
    IntervalBounds,
    ScoredPoint,
    SliceKey,
    build_report,
)

ENTITY = EntityKey("route-A", "dir-N", "stop-1")
OVERALL = SliceKey("overall", "all")
START = datetime(2026, 3, 2, 10, tzinfo=MOSCOW)
ACTUAL = 100.0
SAMPLES = 12


def points(predicted, bounds_of):
    return [
        ScoredPoint(
            entity=ENTITY,
            horizon="day",
            bucket_start=START + timedelta(hours=index),
            fold_id=f"f{index % 2}",
            target="synthetic_boardings",
            unit="event_count",
            event=None,
            actual=ACTUAL,
            predicted=predicted,
            baseline=None,
            interval=bounds_of(predicted),
            capacity=None,
        )
        for index in range(SAMPLES)
    ]


def tight(predicted):
    return IntervalBounds(predicted - 10.0, predicted + 10.0, 0.8, "hand")


def giant(predicted):
    return IntervalBounds(0.0, 10_000.0, 0.8, "hand")


def test_coverage_alone_cannot_tell_a_tight_interval_from_a_useless_one() -> None:
    tight_interval = build_report(points(ACTUAL, tight)).by_key[OVERALL].interval
    giant_interval = build_report(points(ACTUAL, giant)).by_key[OVERALL].interval

    assert tight_interval is not None and giant_interval is not None
    assert tight_interval.coverage == giant_interval.coverage == 1.0
    assert tight_interval.mean_width == 20.0
    assert giant_interval.mean_width == 10_000.0
    assert tight_interval.mean_score == 20.0
    assert giant_interval.mean_score == 10_000.0
    assert giant_interval.mean_score > tight_interval.mean_score


def test_a_giant_interval_fails_the_gate_a_tight_one_passes() -> None:
    tight_report = build_report(points(ACTUAL, tight))
    giant_report = build_report(points(ACTUAL, giant))

    assert tight_report.passed is True
    assert giant_report.passed is False
    assert {failure.metric for failure in giant_report.failures} == {"mean_interval_score"}
    (failure,) = [item for item in giant_report.failures if item.key == OVERALL]
    assert failure.unit == "event_count"
    assert failure.observed == 10_000.0
    assert failure.limit == ACTUAL * MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL
    assert failure.threshold_name == "MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL"
    assert failure.samples == SAMPLES
    assert failure.folds == 2


def test_relative_width_is_reported_beside_coverage() -> None:
    metrics = build_report(points(ACTUAL, giant)).by_key[OVERALL]

    assert metrics.relative_interval_width == 100.0
    assert metrics.mean_actual == ACTUAL


def test_an_interval_that_misses_too_often_fails_on_coverage() -> None:
    def narrow(predicted):
        return IntervalBounds(95.0, 99.0, 0.8, "hand")

    report = build_report(points(97.0, narrow))

    assert {failure.metric for failure in report.failures} == {"interval_coverage"}
    (failure,) = [item for item in report.failures if item.key == OVERALL]
    assert failure.direction == "at_least"
    assert failure.observed == 0.0
    assert failure.limit == pytest.approx(0.7)
    assert "falls below" in failure.message()


def test_mixed_levels_suppress_the_interval_with_a_reason() -> None:
    mixed = points(ACTUAL, tight)
    mixed[0] = ScoredPoint(
        **{
            **{name: getattr(mixed[0], name) for name in mixed[0].__slots__},
            "interval": IntervalBounds(90.0, 110.0, 0.5, "hand"),
        }
    )

    metrics = build_report(mixed).by_key[OVERALL]

    assert metrics.interval is None
    assert "different interval levels" in metrics.interval_absent_reason


def test_partial_intervals_suppress_the_interval_with_a_reason() -> None:
    partial = points(ACTUAL, tight)
    partial[0] = ScoredPoint(
        **{
            **{name: getattr(partial[0], name) for name in partial[0].__slots__},
            "interval": None,
        }
    )

    metrics = build_report(partial).by_key[OVERALL]

    assert metrics.interval is None
    assert f"only 11 of {SAMPLES} points carry an interval" in metrics.interval_absent_reason


def test_a_zero_width_interval_is_scored_by_its_misses() -> None:
    def spike(predicted):
        return IntervalBounds(predicted, predicted, 0.8, "hand")

    metrics = build_report(
        points(90.0, spike), thresholds=GateThresholds(max_wape=1.0)
    ).by_key[OVERALL]

    assert metrics.interval is not None
    assert metrics.interval.mean_width == 0.0
    assert metrics.interval.coverage == 0.0
    assert metrics.interval.mean_score == pytest.approx(100.0)


def test_mixed_methods_suppress_the_interval_with_a_reason() -> None:
    mixed = points(ACTUAL, tight)
    mixed[0] = ScoredPoint(
        **{
            **{name: getattr(mixed[0], name) for name in mixed[0].__slots__},
            "interval": IntervalBounds(90.0, 110.0, 0.8, "conformal"),
        }
    )

    metrics = build_report(mixed).by_key[OVERALL]

    assert metrics.interval is None
    assert metrics.interval_absent_reason == (
        "points declare different interval methods: ['conformal', 'hand']"
    )

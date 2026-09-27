"""Hand-computed fixture: every number below was computed on paper before the code ran.

Four scored points, one entity, ``day`` horizon (hourly buckets), level 0.8 so
``alpha = 0.2`` and the miss multiplier ``2/alpha`` is 10.

| # | fold | Moscow hour | actual | predicted | baseline | lower | upper |
|---|------|-------------|--------|-----------|----------|-------|-------|
| 1 | f1   | 08          |    100 |        90 |       60 |    80 |   110 |
| 2 | f1   | 18          |    200 |       240 |      300 |   210 |   260 |
| 3 | f2   | 08          |    150 |       140 |      100 |   120 |   170 |
| 4 | f2   | 12          |     50 |        45 |      100 |    40 |    60 |

overall   actual 100+200+150+50 = 500; |error| 10+40+10+5 = 65
          mae 65/4 = 16.25; wape 65/500 = 0.13
          baseline |error| 40+100+50+50 = 240; baseline wape 240/500 = 0.48
          covered 1,3,4 (point 2 has 200 < 210) so coverage 3/4 = 0.75
          widths 30+50+50+20 = 150, mean 37.5
          scores 30, 50+10*(210-200)=150, 50, 20 -> 250, mean 62.5
          mean actual 500/4 = 125; relative width 37.5/125 = 0.3
morning   points 1 and 3: actual 250, error 20, mae 10, wape 0.08,
          coverage 1.0, mean width 40, mean score 40
evening   point 2 alone: actual 200, error 40, mae 40, wape 0.2,
          coverage 0.0, mean width 50, mean score 150
offpeak   point 4 alone: actual 50, error 5, mae 5, wape 0.1,
          coverage 1.0, mean width 20, mean score 20
"""

from datetime import datetime

import pytest

from tramflow_ml.features import MOSCOW, EntityKey
from tramflow_ml.slices import (
    GateThresholds,
    IntervalBounds,
    OverloadQuality,
    ScoredPoint,
    SliceError,
    SliceKey,
    SliceMetrics,
    build_report,
    interval_score,
)

ENTITY = EntityKey("route-A", "dir-N", "stop-1")
LEVEL = 0.8
UNGATED = GateThresholds(min_samples=1, min_folds=1)
HAND = (
    ("f1", 8, 100.0, 90.0, 60.0, 80.0, 110.0),
    ("f1", 18, 200.0, 240.0, 300.0, 210.0, 260.0),
    ("f2", 8, 150.0, 140.0, 100.0, 120.0, 170.0),
    ("f2", 12, 50.0, 45.0, 100.0, 40.0, 60.0),
)


def hand_points() -> tuple[ScoredPoint, ...]:
    return tuple(
        ScoredPoint(
            entity=ENTITY,
            horizon="day",
            bucket_start=datetime(2026, 3, 2, hour, tzinfo=MOSCOW),
            fold_id=fold,
            target="synthetic_boardings",
            unit="event_count",
            event=None,
            actual=actual,
            predicted=predicted,
            baseline=baseline,
            interval=IntervalBounds(lower, upper, LEVEL, "hand"),
            capacity=None,
        )
        for fold, hour, actual, predicted, baseline, lower, upper in HAND
    )


@pytest.fixture
def report():
    return build_report(hand_points(), thresholds=UNGATED)


def test_overall_matches_the_hand_calculation(report) -> None:
    overall = report.by_key[SliceKey("overall", "all")]

    assert overall.unit == "event_count"
    assert overall.samples == 4
    assert overall.folds == 2
    assert overall.status == "evaluated"
    assert overall.totals.actual_total == 500
    assert overall.totals.error_total == 65
    assert overall.mae == 16.25
    assert overall.wape == 0.13
    assert overall.baseline_wape == 0.48
    assert overall.mean_actual == 125


def test_overall_interval_matches_the_hand_calculation(report) -> None:
    interval = report.by_key[SliceKey("overall", "all")].interval

    assert interval is not None
    assert interval.level == LEVEL
    assert interval.samples == 4
    assert interval.covered == 3
    assert interval.coverage == 0.75
    assert interval.mean_width == 37.5
    assert interval.mean_score == 62.5
    assert report.by_key[SliceKey("overall", "all")].relative_interval_width == 0.3


@pytest.mark.parametrize(
    ("daypart", "samples", "actual", "error", "mae", "wape", "coverage", "width", "score"),
    [
        ("morning_peak", 2, 250.0, 20.0, 10.0, 0.08, 1.0, 40.0, 40.0),
        ("evening_peak", 1, 200.0, 40.0, 40.0, 0.2, 0.0, 50.0, 150.0),
        ("offpeak", 1, 50.0, 5.0, 5.0, 0.1, 1.0, 20.0, 20.0),
    ],
)
def test_daypart_slices_match_the_hand_calculation(
    report, daypart, samples, actual, error, mae, wape, coverage, width, score
) -> None:
    metrics = report.by_key[SliceKey("daypart", daypart)]

    assert metrics.samples == samples
    assert metrics.totals.actual_total == actual
    assert metrics.totals.error_total == error
    assert metrics.mae == mae
    assert metrics.wape == pytest.approx(wape)
    assert metrics.interval is not None
    assert metrics.interval.coverage == coverage
    assert metrics.interval.mean_width == width
    assert metrics.interval.mean_score == score


@pytest.mark.parametrize(
    ("lower", "upper", "actual", "expected"),
    [
        (80.0, 110.0, 100.0, 30.0),
        (210.0, 260.0, 200.0, 150.0),
        (210.0, 260.0, 300.0, 450.0),
        (100.0, 100.0, 100.0, 0.0),
    ],
)
def test_interval_score_is_width_plus_the_miss_penalty(lower, upper, actual, expected) -> None:
    """``2/alpha`` is 10.000000000000002 in binary floats, so the miss term is approximate."""
    score = interval_score(IntervalBounds(lower, upper, LEVEL, "hand"), actual)

    assert score == pytest.approx(expected)


def test_every_axis_the_fixture_can_support_is_produced(report) -> None:
    assert {key.axis for key in report.by_key} == {
        "overall",
        "horizon",
        "fold",
        "route",
        "direction",
        "stop",
        "entity",
        "daypart",
        "route_daypart",
        "entity_daypart",
    }
    assert report.by_key[SliceKey("direction", "route-A|dir-N")].samples == 4
    assert report.by_key[SliceKey("entity", "route-A|dir-N|stop-1")].samples == 4
    assert report.by_key[SliceKey("route_daypart", "route-A|evening_peak")].samples == 1
    assert report.by_key[SliceKey("entity_daypart", "route-A|dir-N|stop-1|offpeak")].samples == 1
    assert report.by_key[SliceKey("fold", "f1")].samples == 2


def test_a_monthly_bucket_is_in_no_daypart_slice() -> None:
    points = [
        ScoredPoint(
            entity=ENTITY,
            horizon="year",
            bucket_start=datetime(2026, month, 1, tzinfo=MOSCOW),
            fold_id="f1",
            target="synthetic_boardings",
            unit="event_count",
            event=None,
            actual=100.0,
            predicted=100.0,
            baseline=None,
            interval=None,
            capacity=None,
        )
        for month in (1, 2)
    ]

    report = build_report(points, thresholds=UNGATED)

    assert not [key for key in report.by_key if key.axis in {"daypart", "route_daypart"}]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"samples": 0}, "at least one sample"),
        ({"folds": 0}, "at least one sample"),
        ({"folds": 9}, "more folds than it has samples"),
        ({"samples": 3}, "every sample of a slice is scored"),
    ],
)
def test_slice_metrics_refuse_a_support_they_cannot_carry(report, overrides, message) -> None:
    built = report.by_key[SliceKey("overall", "all")]
    fields = {name: getattr(built, name) for name in built.__slots__}

    with pytest.raises(SliceError, match=message):
        SliceMetrics(**{**fields, **overrides})


def test_overload_quality_refuses_a_count_outside_its_points() -> None:
    with pytest.raises(SliceError, match="must lie within the points it counts"):
        OverloadQuality(samples=2, actual_overloaded=3, predicted_overloaded=0)

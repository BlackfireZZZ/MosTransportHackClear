"""The three constructions an acceptance review used to pass models that should fail.

Each was reproduced before the fix and each is pinned here, so the headline promise --
that a bad slice cannot hide behind a good mean -- is checked against the ways it was
actually broken rather than against the ways it was designed.
"""

from datetime import datetime, timedelta

import pytest

from tramflow_ml.features import MOSCOW, EntityKey
from tramflow_ml.slices import (
    MIN_INTERVAL_LEVEL,
    GateThresholds,
    IntervalBounds,
    ScoredPoint,
    SliceError,
    SliceKey,
    build_report,
)

DAY = datetime(2026, 3, 2, tzinfo=MOSCOW)
OVERALL = SliceKey("overall", "all")
EVENING_HOURS = (17, 18, 19)
QUIET_HOURS = (9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23)


def point(*, stop="stop-1", day=0, hour=12, actual=100.0, predicted=100.0, **overrides):
    fields = {
        "entity": EntityKey("route-A", "dir-N", stop),
        "horizon": "day",
        "bucket_start": DAY + timedelta(days=day, hours=hour),
        "fold_id": f"f{day}",
        "target": "synthetic_boardings",
        "unit": "event_count",
        "event": None,
        "actual": actual,
        "predicted": predicted,
        "baseline": None,
        "interval": None,
        "capacity": None,
    }
    return ScoredPoint(**{**fields, **overrides})


def degenerate(level):
    """A point estimate wearing an interval's clothes: zero width, 30% error."""
    return [
        point(
            day=index % 2,
            hour=QUIET_HOURS[index // 2],
            actual=100.0,
            predicted=70.0,
            interval=IntervalBounds(70.0, 70.0, level, "degenerate"),
        )
        for index in range(12)
    ]


@pytest.mark.parametrize("level", [0.02, 0.5, 0.79])
def test_a_self_declared_low_level_no_longer_buys_a_pass(level: float) -> None:
    report = build_report(degenerate(level))

    assert report.passed is False
    (failure,) = [
        item
        for item in report.failures
        if item.key == OVERALL and item.metric == "interval_level"
    ]
    assert failure.observed == level
    assert failure.limit == MIN_INTERVAL_LEVEL
    assert failure.direction == "at_least"
    assert failure.threshold_name == "MIN_INTERVAL_LEVEL"


def test_the_vacuous_coverage_check_is_skipped_rather_than_computed_negative() -> None:
    report = build_report(degenerate(0.02))

    metrics = [item.metric for item in report.failures if item.key == OVERALL]
    assert "interval_coverage" not in metrics
    assert metrics == ["interval_level"]


def test_a_coverage_requirement_that_could_never_bite_is_refused() -> None:
    with pytest.raises(ValueError, match="max_coverage_shortfall must be below"):
        GateThresholds(min_interval_level=0.8, max_coverage_shortfall=0.8)


def collapsed_fold():
    """Five perfect origins and one where the model predicts zero against demand 100."""
    good = [point(day=day, hour=hour) for day in range(5) for hour in QUIET_HOURS]
    bad = [point(day=5, hour=hour, predicted=0.0) for hour in QUIET_HOURS]
    return good + bad


def test_a_model_that_collapses_on_one_origin_is_visible_and_fails() -> None:
    report = build_report(collapsed_fold())

    overall = report.by_key[OVERALL]
    assert overall.wape == pytest.approx(1 / 6)
    assert overall.status == "evaluated"

    collapsed = report.by_key[SliceKey("fold", "f5")]
    assert collapsed.samples == 12
    assert collapsed.folds == 1
    assert collapsed.wape == 1.0
    assert collapsed.status == "evaluated"
    assert report.passed is False
    assert SliceKey("fold", "f5") in {failure.key for failure in report.failures}


def test_a_healthy_fold_slice_is_judged_not_excused_for_being_one_origin() -> None:
    report = build_report(collapsed_fold())

    healthy = report.by_key[SliceKey("fold", "f0")]
    assert healthy.folds == 1
    assert healthy.status == "evaluated"
    assert SliceKey("fold", "f0") not in {failure.key for failure in report.failures}


def one_bad_cell():
    """Six stops on one route; stop-1 is catastrophic only at the evening peak."""
    points = []
    for day in range(4):
        for stop in range(1, 7):
            for hour in (*QUIET_HOURS, *EVENING_HOURS):
                broken = stop == 1 and hour in EVENING_HOURS
                points.append(
                    point(
                        stop=f"stop-{stop}",
                        day=day,
                        hour=hour,
                        predicted=0.0 if broken else 100.0,
                    )
                )
    return points


def test_one_stop_at_one_peak_is_cut_out_rather_than_diluted() -> None:
    report = build_report(one_bad_cell())

    assert report.by_key[OVERALL].samples == 360
    assert report.by_key[OVERALL].wape == pytest.approx(1 / 30)
    assert report.by_key[SliceKey("stop", "stop-1")].wape == pytest.approx(0.2)
    assert report.by_key[SliceKey("daypart", "evening_peak")].wape == pytest.approx(1 / 6)

    cell = report.by_key[SliceKey("entity_daypart", "route-A|dir-N|stop-1|evening_peak")]
    assert cell.samples == 12
    assert cell.folds == 4
    assert cell.wape == 1.0
    assert report.passed is False
    assert cell.key in {failure.key for failure in report.failures}


def test_the_diluting_axes_still_pass_so_the_cross_is_what_caught_it() -> None:
    report = build_report(one_bad_cell())

    failed = {failure.key for failure in report.failures}
    assert SliceKey("stop", "stop-1") not in failed
    assert SliceKey("daypart", "evening_peak") not in failed
    assert SliceKey("entity", "route-A|dir-N|stop-1") not in failed
    assert {key.axis for key in failed} == {"entity_daypart"}


def test_a_required_key_no_axis_emits_is_refused_at_the_boundary() -> None:
    with pytest.raises(SliceError, match="no axis emits \\['stop_event'\\]"):
        build_report(one_bad_cell(), required=(SliceKey("stop_event", "stop-1|fire"),))

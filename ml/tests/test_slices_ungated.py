"""What the gate did not judge, said out loud, and the two cases that forced it.

A floor keeps a slice out of the gate. It must not keep it out of the reader's view:
twenty rows placed on one origin is not sample scarcity, and a WAPE of 1.0 there is a
finding whatever the gate did with it.
"""

from datetime import datetime, timedelta

import pytest

from tramflow_ml.features import MOSCOW, EntityKey
from tramflow_ml.slices import (
    MIN_INTERVAL_SCORE_ALLOWANCE,
    IntervalBounds,
    ScoredPoint,
    SliceKey,
    build_report,
)

DAY = datetime(2026, 3, 2, tzinfo=MOSCOW)
OVERALL = SliceKey("overall", "all")
QUIET_HOURS = (0, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23)


def point(
    *, route="route-A", stop="stop-1", day=0, hour=12, actual=100.0, predicted=100.0, **overrides
):
    fields = {
        "entity": EntityKey(route, "dir-N", stop),
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


def healthy():
    return [point(day=day, hour=hour) for day in range(4) for hour in QUIET_HOURS]


ALL_QUIET_HOURS = (*QUIET_HOURS, 1, 2, 3, 4, 5, 6)


def busy():
    """Enough well-predicted rows on every origin to dilute a single-origin disaster."""
    return [
        point(stop=f"stop-{stop}", day=day, hour=hour)
        for day in range(4)
        for stop in range(1, 7)
        for hour in QUIET_HOURS
    ]


def test_a_disaster_parked_on_one_origin_is_named_even_though_it_is_ungated() -> None:
    """18 rows is not sample scarcity; placing them all on one origin dodges the gate.

    The `fold` axis catches the cruder version of this dodge, where the bad origin has
    nothing to hide behind. Here fold `f0` also carries 72 well-predicted rows, so its
    WAPE is 0.2 and it passes; every cut of the bad rows is below the fold floor. The
    ungated line is then the only thing that names them.
    """
    hidden = [
        point(route="route-B", day=0, hour=hour, predicted=0.0)
        for hour in ALL_QUIET_HOURS
    ]

    report = build_report(busy() + hidden)

    assert report.passed is True
    assert report.by_key[SliceKey("fold", "f0")].wape == pytest.approx(0.2)
    ungated_route = report.by_key[SliceKey("route", "route-B")]
    assert ungated_route.samples == 18
    assert ungated_route.samples > report.thresholds.min_samples
    assert ungated_route.folds == 1
    assert ungated_route.wape == 1.0
    assert ungated_route.status == "insufficient_history"
    assert report.ungated.worst_wape == 1.0
    assert report.ungated.worst_key == SliceKey("direction", "route-B|dir-N")
    assert "worst ungated WAPE 1 on direction=route-B|dir-N" in report.ungated.message()
    assert report.ungated.message() in report.verdict()


def test_the_crude_version_of_that_dodge_is_caught_by_the_fold_axis() -> None:
    """With nothing to dilute it, a bad origin is a gated `fold` slice and fails."""
    hidden = [
        point(route="route-B", day=9, hour=hour, predicted=0.0)
        for hour in ALL_QUIET_HOURS
    ]

    report = build_report(busy() + hidden)

    assert report.passed is False
    assert SliceKey("fold", "f9") in {failure.key for failure in report.failures}


def test_the_ungated_line_counts_each_floor_separately() -> None:
    tiny = [point(route="route-C", day=day, hour=10, predicted=50.0) for day in range(3)]
    dead = [
        point(route="route-D", day=day, hour=hour, actual=0.0, predicted=1.0)
        for day in range(4)
        for hour in QUIET_HOURS
    ]

    summary = build_report(healthy() + tiny + dead).ungated

    assert summary.below_sample_floor >= 1
    assert summary.without_signal >= 1
    assert summary.slices == (
        summary.below_sample_floor + summary.below_fold_floor + summary.without_signal
    )
    assert "below the sample floor" in summary.message()


def test_a_fully_judged_report_says_so() -> None:
    report = build_report(healthy())

    assert report.passed is True
    assert report.ungated.slices == 0
    assert report.ungated.message() == "every slice was judged"
    assert report.verdict().endswith("every slice was judged")


def test_the_ungated_line_survives_having_no_defined_wape() -> None:
    dead = [
        point(route="route-D", day=day, hour=hour, actual=0.0, predicted=1.0)
        for day in range(4)
        for hour in QUIET_HOURS
    ]

    summary = build_report(dead).ungated

    assert summary.worst_wape is None
    assert "no ungated slice has a defined WAPE" in summary.message()


def low_demand(upper):
    return [
        point(day=day, hour=hour, actual=2.0, predicted=2.0,
              interval=IntervalBounds(0.0, upper, 0.8, "count"))
        for day in range(2) for hour in QUIET_HOURS[:6]
    ]


def test_a_sane_interval_on_two_passenger_demand_is_not_punished() -> None:
    report = build_report(low_demand(7.0))

    metrics = report.by_key[OVERALL]
    assert metrics.mean_actual == 2.0
    assert metrics.interval is not None
    assert metrics.interval.mean_score == 7.0
    assert metrics.mean_actual * 2.0 < MIN_INTERVAL_SCORE_ALLOWANCE
    assert report.passed is True


def test_the_allowance_is_a_floor_not_a_licence() -> None:
    report = build_report(low_demand(40.0))

    (failure,) = [item for item in report.failures if item.key == OVERALL]
    assert failure.metric == "mean_interval_score"
    assert failure.observed == 40.0
    assert failure.limit == MIN_INTERVAL_SCORE_ALLOWANCE
    assert failure.threshold_name == "MIN_INTERVAL_SCORE_ALLOWANCE"


def test_high_demand_still_judges_against_the_ratio_not_the_allowance() -> None:
    wide = [
        point(day=day, hour=hour, actual=1000.0, predicted=1000.0,
              interval=IntervalBounds(0.0, 2500.0, 0.8, "wide"))
        for day in range(2) for hour in QUIET_HOURS[:6]
    ]

    (failure,) = [item for item in build_report(wide).failures if item.key == OVERALL]

    assert failure.threshold_name == "MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL"
    assert failure.limit == 2000.0
    assert failure.observed == 2500.0


@pytest.mark.parametrize(
    ("carried", "reason"),
    [
        (0, "no point in this slice carries a baseline"),
        (47, "only 47 of 48 points carry a baseline"),
    ],
)
def test_a_dropped_baseline_says_whether_it_was_the_run_or_one_row(carried, reason) -> None:
    slots = [(day, hour) for day in range(4) for hour in QUIET_HOURS]
    points = [
        point(day=day, hour=hour, baseline=90.0 if index < carried else None)
        for index, (day, hour) in enumerate(slots)
    ]

    metrics = build_report(points).by_key[OVERALL]

    assert metrics.baseline is None
    assert metrics.baseline_wape is None
    assert reason in metrics.baseline_absent_reason
    assert metrics.to_dict()["baseline_absent_reason"] == metrics.baseline_absent_reason


def test_a_complete_baseline_carries_no_absent_reason() -> None:
    points = [
        point(day=day, hour=hour, baseline=90.0)
        for day in range(4)
        for hour in QUIET_HOURS
    ]

    metrics = build_report(points).by_key[OVERALL]

    assert metrics.baseline is not None
    assert metrics.baseline_absent_reason == ""


def test_every_reported_metric_names_its_own_unit() -> None:
    """The slice's `unit` is the target's; a consumer applying it to `wape` is wrong."""
    dimensionless = {"wape", "baseline_wape", "relative_interval_width"}
    payload = build_report(
        [
            point(day=day, hour=hour, baseline=90.0,
                  interval=IntervalBounds(90.0, 110.0, 0.8, "hand"))
            for day in range(4)
            for hour in QUIET_HOURS
        ]
    ).to_dict()

    for item in payload["slices"]:
        units = item["units"]
        assert units.keys() >= dimensionless | {"mae", "actual_total", "samples", "folds"}
        assert {units[name] for name in dimensionless} == {"ratio"}
        assert units["mae"] == units["actual_total"] == item["unit"] == "event_count"
        assert units["samples"] == units["folds"] == "count"
        interval_units = item["interval"]["units"]
        assert interval_units["coverage"] == interval_units["level"] == "ratio"
        assert interval_units["mean_interval_score"] == "event_count"
        assert interval_units["mean_width"] == "event_count"
        assert "folds" not in item["interval"]


def test_overload_rates_are_ratios_and_its_counts_are_counts() -> None:
    loads = [
        point(day=day, hour=hour, actual=250.0, predicted=150.0,
              target="onboard_load", unit="passengers", capacity=200.0)
        for day in range(4)
        for hour in QUIET_HOURS
    ]

    overload = build_report(loads).by_key[OVERALL].to_dict()["overload"]

    assert overload["units"]["actual_overload_rate"] == "ratio"
    assert overload["units"]["predicted_overload_rate"] == "ratio"
    assert overload["units"]["actual_overloaded"] == "count"
    assert "folds" not in overload

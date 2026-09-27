"""The gate this task exists for: an excellent mean must not buy off a bad slice."""

import random
from datetime import datetime, timedelta

import pytest

from tramflow_ml.features import MOSCOW, EntityKey
from tramflow_ml.features.records import encode
from tramflow_ml.slices import (
    MAX_WAPE,
    GateThresholds,
    ScoredPoint,
    SliceError,
    SliceKey,
    build_report,
)

DAY = datetime(2026, 3, 2, tzinfo=MOSCOW)
OVERALL = SliceKey("overall", "all")
EVENING = SliceKey("daypart", "evening_peak")
ROUTE_EVENING = SliceKey("route_daypart", "route-A|evening_peak")
OFFPEAK_HOURS = (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16)
EVENING_HOURS = (17, 18, 19)


def point(*, route="route-A", day=0, hour=12, actual=100.0, predicted=100.0, **overrides):
    fields = {
        "entity": EntityKey(route, "dir-N", "stop-1"),
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


def evening_peak_disaster():
    """48 perfect off-peak points and 12 catastrophic evening-peak ones, four folds.

    overall  actual 60 * 100 = 6000; error 12 * 90 = 1080; wape 0.18 <= MAX_WAPE 0.40
    evening  actual 12 * 100 = 1200; error 1080; wape 0.90 > MAX_WAPE
    """
    quiet = [point(day=day, hour=hour) for day in range(4) for hour in OFFPEAK_HOURS[:12]]
    peak = [
        point(day=day, hour=hour, actual=100.0, predicted=10.0)
        for day in range(4)
        for hour in EVENING_HOURS
    ]
    return quiet + peak


def test_a_catastrophic_evening_peak_cannot_hide_in_a_passing_mean() -> None:
    report = build_report(evening_peak_disaster())

    overall = report.by_key[OVERALL]
    assert overall.samples == 60
    assert overall.folds == 4
    assert overall.wape == pytest.approx(0.18)
    assert overall.wape < MAX_WAPE
    assert report.passed is False
    assert {failure.key for failure in report.failures} >= {EVENING, ROUTE_EVENING}
    assert OVERALL not in {failure.key for failure in report.failures}


def test_the_failure_names_the_slice_its_support_and_the_threshold_it_breached() -> None:
    report = build_report(evening_peak_disaster())

    (failure,) = [item for item in report.failures if item.key == ROUTE_EVENING]
    assert failure.metric == "wape"
    assert failure.unit == "ratio"
    assert failure.observed == pytest.approx(0.9)
    assert failure.limit == MAX_WAPE
    assert failure.threshold_name == "MAX_WAPE"
    assert failure.samples == 12
    assert failure.folds == 4
    assert failure.message() == (
        "route_daypart=route-A|evening_peak: wape 0.9 ratio exceeds 0.4 "
        "(MAX_WAPE=0.4) on 12 samples across 4 folds"
    )
    assert "route_daypart=route-A|evening_peak" in report.verdict()


def test_the_same_slice_fails_whether_or_not_anyone_declared_it_required() -> None:
    undeclared = build_report(evening_peak_disaster())
    declared = build_report(evening_peak_disaster(), required=(ROUTE_EVENING,))

    assert undeclared.passed is False
    assert declared.passed is False
    assert {item.key for item in undeclared.failures} == {item.key for item in declared.failures}
    assert declared.unproven == ()


def test_the_quiet_hours_alone_would_have_passed() -> None:
    quiet = [point(day=day, hour=hour) for day in range(4) for hour in OFFPEAK_HOURS[:12]]

    report = build_report(quiet)

    assert report.passed is True
    assert report.by_key[OVERALL].wape == 0.0
    assert report.verdict().startswith("PASS:")


def test_a_required_slice_with_no_points_is_unproven_not_a_silent_pass() -> None:
    missing = SliceKey("route", "route-B")
    quiet = [point(day=day, hour=hour) for day in range(4) for hour in OFFPEAK_HOURS[:12]]

    report = build_report(quiet, required=(missing,))

    assert report.passed is False
    assert report.failures == ()
    (unproven,) = report.unproven
    assert unproven.key == missing
    assert unproven.samples == 0
    assert unproven.status == "missing"
    assert unproven.message() == (
        "route=route-B: required slice is missing: no scored point falls in this slice"
    )


def test_a_small_slice_is_reported_in_full_marked_and_not_gated() -> None:
    tiny = [point(day=day, hour=12, actual=100.0, predicted=10.0) for day in range(3)]

    report = build_report(tiny)

    overall = report.by_key[OVERALL]
    assert overall.samples == 3
    assert overall.folds == 3
    assert overall.wape == pytest.approx(0.9)
    assert overall.status == "insufficient_samples"
    assert "MIN_SAMPLES_FOR_GATE=12" in overall.status_reason
    assert report.failures == ()
    assert report.passed is False
    assert report.unproven[0].status == "insufficient_samples"


def test_a_single_fold_is_insufficient_history_not_a_verdict() -> None:
    one_origin = [point(day=0, hour=hour) for hour in OFFPEAK_HOURS]

    overall = build_report(one_origin).by_key[OVERALL]

    assert overall.samples == 14
    assert overall.folds == 1
    assert overall.status == "insufficient_history"
    assert "MIN_FOLDS_FOR_GATE=2" in overall.status_reason


def zero_demand_report(**kwargs):
    quiet = [point(day=day, hour=hour) for day in range(4) for hour in OFFPEAK_HOURS[:12]]
    dead = [
        point(route="route-Z", day=day, hour=hour, actual=0.0, predicted=5.0)
        for day in range(4)
        for hour in OFFPEAK_HOURS[:3]
    ]
    return build_report(quiet + dead, **kwargs)


def test_a_zero_demand_slice_reports_null_wape_and_insufficient_signal() -> None:
    metrics = zero_demand_report().by_key[SliceKey("route", "route-Z")]

    assert metrics.samples == 12
    assert metrics.totals.actual_total == 0.0
    assert metrics.wape is None
    assert metrics.mae == 5.0
    assert metrics.status == "insufficient_signal"
    assert "WAPE is undefined" in metrics.status_reason


def test_zero_demand_raises_the_pooled_ratio_it_cannot_flatter_it() -> None:
    report = zero_demand_report()

    overall = report.by_key[OVERALL]
    assert overall.totals.actual_total == 4800.0
    assert overall.totals.error_total == 60.0
    assert overall.wape == pytest.approx(0.0125)
    assert overall.wape > report.by_key[SliceKey("route", "route-A")].wape


def test_a_required_zero_demand_slice_is_unproven() -> None:
    report = zero_demand_report(required=(SliceKey("route", "route-Z"),))

    assert report.passed is False
    (unproven,) = report.unproven
    assert unproven.status == "insufficient_signal"
    assert unproven.samples == 12


def test_a_report_covers_one_quantity() -> None:
    mixed = [point(day=0, hour=10), point(day=1, hour=10, unit="passengers")]

    with pytest.raises(SliceError, match="a report covers one quantity"):
        build_report(mixed)


def test_an_empty_report_is_refused() -> None:
    with pytest.raises(SliceError, match="at least one scored point"):
        build_report([])


def test_the_report_prints_the_thresholds_it_used_and_that_they_are_uncertified() -> None:
    report = build_report(evening_peak_disaster(), thresholds=GateThresholds(max_wape=0.95))

    assert report.passed is True
    assert report.to_dict()["thresholds"] == {
        "certified": False,
        "MIN_SAMPLES_FOR_GATE": 12,
        "MIN_FOLDS_FOR_GATE": 2,
        "MAX_WAPE": 0.95,
        "MAX_WAPE_RATIO_TO_BASELINE": 1.0,
        "MAX_COVERAGE_SHORTFALL": 0.10,
        "MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL": 2.0,
        "MIN_INTERVAL_LEVEL": 0.8,
        "MIN_INTERVAL_SCORE_ALLOWANCE": 8.0,
    }


def test_the_report_does_not_depend_on_the_order_it_was_given() -> None:
    points = evening_peak_disaster()
    shuffled = list(points)
    random.Random(7).shuffle(shuffled)

    first = build_report(points)
    second = build_report(shuffled)

    assert first.digest == second.digest
    assert first.to_dict() == second.to_dict()


def test_a_slice_that_loses_to_its_own_baseline_fails() -> None:
    losing = [
        point(day=day, hour=hour, actual=100.0, predicted=80.0, baseline=95.0)
        for day in range(4)
        for hour in OFFPEAK_HOURS[:3]
    ]

    report = build_report(losing)

    (failure,) = [item for item in report.failures if item.key == OVERALL]
    assert failure.metric == "wape_vs_baseline"
    assert failure.observed == pytest.approx(0.2)
    assert failure.limit == pytest.approx(0.05)
    assert failure.threshold_name == "MAX_WAPE_RATIO_TO_BASELINE"


def test_a_baseline_missing_on_one_point_removes_the_comparison_not_the_slice() -> None:
    losing = [
        point(day=day, hour=hour, actual=100.0, predicted=80.0, baseline=95.0)
        for day in range(4)
        for hour in OFFPEAK_HOURS[:3]
    ]
    losing[0] = point(day=0, hour=OFFPEAK_HOURS[0], actual=100.0, predicted=80.0)

    metrics = build_report(losing).by_key[OVERALL]

    assert metrics.baseline_wape is None
    assert metrics.wape == pytest.approx(0.2)


def test_the_report_serializes_to_identical_bytes_across_runs() -> None:
    first = encode(build_report(evening_peak_disaster()).to_dict())
    second = encode(build_report(evening_peak_disaster()).to_dict())

    assert first == second
    assert b"2026-" not in first
    assert b"/home" not in first


def test_every_slice_carries_its_unit_samples_and_folds() -> None:
    payload = build_report(evening_peak_disaster()).to_dict()

    assert payload["slices"]
    for item in payload["slices"]:
        assert item["unit"] == "event_count"
        assert item["samples"] >= 1
        assert item["folds"] >= 1
        assert item["status"]


def test_a_bucket_scored_twice_in_one_fold_is_refused() -> None:
    twice = [point(day=0, hour=10), point(day=0, hour=10, predicted=50.0)]

    with pytest.raises(SliceError, match="a bucket is scored once per fold"):
        build_report(twice)

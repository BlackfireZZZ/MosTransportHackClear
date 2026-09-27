"""What the slice report adds to the golden gate, and where the two must agree.

``evaluation.py`` is deliberately untouched: it backs ``make ml-eval`` and its output is
byte-identical to what it was before this module existed. These tests pin the WAPE the
two compute against each other on the real golden dataset, and demonstrate the gap that
justified building the second reporter -- the golden gate reports interval coverage with
no width, so it cannot tell a tight interval from a useless one.
"""

from datetime import datetime
from pathlib import Path

import pytest

from tramflow_ml.evaluation import ForecastCase, evaluate, load_cases
from tramflow_ml.features import MOSCOW
from tramflow_ml.slices import IntervalBounds, ScoredPoint, SliceKey, build_report

GOLDEN = Path("ml/evals/golden_cases.json")
OVERALL = SliceKey("overall", "all")
LEVEL = 0.8


def monthly_points(cases, *, bounds_of=None):
    return [
        ScoredPoint(
            entity=None,
            horizon="month",
            bucket_start=datetime(2026, index + 1, 1, tzinfo=MOSCOW),
            fold_id=case.id,
            target="synthetic_boardings",
            unit="event_count",
            event=case.scenario,
            actual=actual,
            predicted=predicted,
            baseline=baseline,
            interval=None if bounds_of is None else bounds_of(predicted),
            capacity=None,
        )
        for case in cases
        for index, (actual, predicted, baseline) in enumerate(
            zip(case.actual, case.prediction, case.baseline_prediction, strict=True)
        )
    ]


def test_slice_wape_agrees_with_the_golden_evaluation_gate() -> None:
    cases = load_cases(GOLDEN)
    monthly = [case for case in cases if case.horizon == "month"]

    summary = evaluate(cases)
    report = build_report(monthly_points(monthly))

    overall = report.by_key[OVERALL]
    assert overall.samples == summary.by_horizon["month"].observations
    assert overall.wape == pytest.approx(summary.by_horizon["month"].wape)
    assert overall.mae == pytest.approx(summary.by_horizon["month"].mae)
    assert overall.baseline_wape == pytest.approx(summary.by_horizon["month"].baseline_wape)


def test_the_scenario_of_a_golden_case_becomes_an_event_slice() -> None:
    monthly = [case for case in load_cases(GOLDEN) if case.horizon == "month"]

    report = build_report(monthly_points(monthly))

    assert report.by_key[SliceKey("event", "route_interval_change")].samples == 4
    assert report.by_key[SliceKey("event", "edge_closure")].samples == 4
    assert not [key for key in report.by_key if key.axis in {"route", "daypart"}]


def giant_cases():
    """Two cases per horizon, eight points each, so a month slice has 16 samples."""
    return [
        ForecastCase(
            id=f"{horizon}-{fold}",
            horizon=horizon,
            scenario="useless_interval",
            actual=[100.0] * 8,
            prediction=[100.0] * 8,
            baseline_prediction=[150.0] * 8,
            lower_bound=[0.0] * 8,
            upper_bound=[10_000.0] * 8,
        )
        for horizon in ("day", "month", "year")
        for fold in ("a", "b")
    ]


def test_the_golden_gate_passes_an_interval_the_slice_report_rejects() -> None:
    cases = giant_cases()

    summary = evaluate(cases)
    monthly = [case for case in cases if case.horizon == "month"]
    report = build_report(
        monthly_points(monthly, bounds_of=lambda _: IntervalBounds(0.0, 10_000.0, LEVEL, "flat"))
    )

    assert summary.passed is True
    assert summary.overall.interval_coverage == 1.0
    assert "mean_width" not in summary.overall.model_dump()

    overall = report.by_key[OVERALL]
    assert overall.interval is not None
    assert overall.interval.coverage == 1.0
    assert overall.interval.mean_width == 10_000.0
    assert overall.interval.mean_score == 10_000.0
    assert report.passed is False
    assert {failure.metric for failure in report.failures} == {"mean_interval_score"}
    assert {failure.key.axis for failure in report.failures} == {"overall", "horizon", "event"}


def test_zero_demand_is_where_the_two_reporters_deliberately_differ() -> None:
    cases = [
        case.model_copy(update={"actual": [0.0] * 8, "lower_bound": [0.0] * 8})
        for case in giant_cases()
    ]

    with pytest.raises(ValueError, match="zero passenger demand; WAPE is undefined"):
        evaluate(cases)

    monthly = [case for case in cases if case.horizon == "month"]
    overall = build_report(monthly_points(monthly)).by_key[OVERALL]
    assert overall.wape is None
    assert overall.status == "insufficient_signal"

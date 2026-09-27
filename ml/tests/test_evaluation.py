from pathlib import Path

import pytest
from pydantic import ValidationError

from tramflow_ml.evaluation import ForecastCase, evaluate, load_cases


def test_golden_evaluation_beats_baseline_on_every_horizon() -> None:
    cases = load_cases(Path("ml/evals/golden_cases.json"))

    summary = evaluate(cases)

    assert summary.passed is True
    assert set(summary.by_horizon) == {"day", "month", "year"}
    assert summary.overall.interval_coverage >= 0.8


def test_case_rejects_misaligned_series() -> None:
    with pytest.raises(ValidationError, match="same non-zero length"):
        ForecastCase(
            id="broken",
            horizon="day",
            scenario="broken",
            actual=[1, 2],
            prediction=[1],
            baseline_prediction=[1],
            lower_bound=[0],
            upper_bound=[2],
        )


@pytest.fixture
def cases() -> list[ForecastCase]:
    return [
        ForecastCase(
            id=horizon,
            horizon=horizon,
            scenario="boundary",
            actual=[10],
            prediction=[10],
            baseline_prediction=[0],
            lower_bound=[10],
            upper_bound=[10],
        )
        for horizon in ("day", "month", "year")
    ]


@pytest.mark.parametrize(
    "field",
    [
        "actual",
        "prediction",
        "baseline_prediction",
        "lower_bound",
        "upper_bound",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_series_rejected(cases: list[ForecastCase], field: str, value: float) -> None:
    data = cases[0].model_dump()
    data[field] = [value]
    with pytest.raises(ValidationError, match="finite"):
        ForecastCase.model_validate(data)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_threshold_rejected(cases: list[ForecastCase], threshold: float) -> None:
    with pytest.raises(ValueError, match="minimum_interval_coverage"):
        evaluate(cases, minimum_interval_coverage=threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_threshold_endpoints_accepted(cases: list[ForecastCase], threshold: float) -> None:
    assert evaluate(cases, minimum_interval_coverage=threshold).passed


def test_coverage_failure_cannot_hide_behind_overall(cases: list[ForecastCase]) -> None:
    cases[0] = ForecastCase(
        id="day",
        horizon="day",
        scenario="many covered points",
        actual=[10] * 8,
        prediction=[10] * 8,
        baseline_prediction=[0] * 8,
        lower_bound=[10] * 8,
        upper_bound=[10] * 8,
    )
    cases[2].actual = [11]
    summary = evaluate(cases)
    assert summary.overall.interval_coverage == 0.9
    assert summary.by_horizon["year"].interval_coverage == 0
    assert not summary.passed


def test_duplicate_ids_rejected_across_horizons(cases: list[ForecastCase]) -> None:
    cases[2].id = cases[0].id
    with pytest.raises(ValueError, match="duplicate evaluation case ID"):
        evaluate(cases)


@pytest.mark.parametrize("horizon", ["day", "month", "year"])
def test_missing_horizon_rejected(cases: list[ForecastCase], horizon: str) -> None:
    with pytest.raises(ValueError, match=f"missing required horizons: {horizon}"):
        evaluate([case for case in cases if case.horizon != horizon])


def test_unsupported_horizon_rejected(cases: list[ForecastCase]) -> None:
    cases[0].horizon = "week"
    with pytest.raises(ValueError, match="unsupported horizons: week"):
        evaluate(cases)


def test_empty_evaluation_rejected() -> None:
    with pytest.raises(ValueError, match="at least one evaluation case"):
        evaluate([])


@pytest.mark.parametrize("all_horizons", [False, True])
def test_zero_demand_is_explicit(cases: list[ForecastCase], all_horizons: bool) -> None:
    for case in cases if all_horizons else cases[2:]:
        case.actual = [0]
    with pytest.raises(
        ValueError,
        match=rf"{'day' if all_horizons else 'year'}.*zero passenger demand.*WAPE is undefined",
    ):
        evaluate(cases)


def test_zero_observations_remain_in_metrics(cases: list[ForecastCase]) -> None:
    cases[0] = ForecastCase(
        id="day",
        horizon="day",
        scenario="zero and positive demand",
        actual=[0, 10],
        prediction=[2, 10],
        baseline_prediction=[0, 0],
        lower_bound=[1, 10],
        upper_bound=[3, 10],
    )
    summary = evaluate(cases)
    metrics = summary.by_horizon["day"]
    assert metrics.observations == 2
    assert metrics.mae == 1
    assert metrics.wape == 0.2
    assert metrics.baseline_wape == 1
    assert metrics.interval_coverage == 0.5
    assert not summary.passed


def test_nonfinite_aggregate_rejected(cases: list[ForecastCase]) -> None:
    for case in cases:
        case.actual = [1e308]
    with pytest.raises(ValueError, match="finite"):
        evaluate(cases)


def test_golden_metrics_unchanged() -> None:
    summary = evaluate(load_cases(Path("ml/evals/golden_cases.json")))
    assert summary.overall.observations == 20
    assert summary.overall.mae == 56.55
    assert summary.overall.wape == pytest.approx(1131 / 51380)
    assert summary.overall.baseline_wape == pytest.approx(6277 / 51380)
    assert all(item.interval_coverage == 1 for item in summary.by_horizon.values())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_mutated_case_revalidated(cases: list[ForecastCase], value: float) -> None:
    cases[0].upper_bound[0] = value
    with pytest.raises(ValueError, match="finite"):
        evaluate(cases)


def test_wape_overflow_rejected(cases: list[ForecastCase]) -> None:
    cases[0].actual = [1e-308]
    cases[0].prediction = [1e308]
    cases[0].upper_bound = [1e308]
    with pytest.raises(ValueError, match="day: evaluation metrics must be finite"):
        evaluate(cases)


def test_horizon_baseline_regression_fails_gate(cases: list[ForecastCase]) -> None:
    cases[2].baseline_prediction = [10]
    cases[2].prediction = [11]
    cases[2].upper_bound = [11]
    summary = evaluate(cases)
    assert summary.overall.wape < summary.overall.baseline_wape
    assert summary.overall.interval_coverage == 1
    assert not summary.passed


def test_zero_demand_case_allowed_in_positive_horizon(cases: list[ForecastCase]) -> None:
    cases.append(
        ForecastCase(
            id="day-zero",
            horizon="day",
            scenario="zero demand",
            actual=[0],
            prediction=[0],
            baseline_prediction=[0],
            lower_bound=[0],
            upper_bound=[0],
        )
    )
    summary = evaluate(cases)
    assert summary.passed
    assert summary.by_horizon["day"].observations == 2
    assert summary.by_horizon["day"].wape == 0

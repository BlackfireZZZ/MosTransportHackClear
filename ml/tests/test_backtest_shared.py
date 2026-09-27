from datetime import date, datetime, timedelta

import pytest

from tramflow_ml.backtest import (
    BacktestConfig,
    BacktestData,
    BacktestError,
    FoldRules,
    combine,
    fold_metrics,
    run_backtest,
)
from tramflow_ml.evaluation import ForecastCase, evaluate
from tramflow_ml.features import MONTH_DAY, MOSCOW, CoverageCalendar, EntityKey, Observation

TARGET = "synthetic_boardings"
UNIT = "event_count"
PUBLICATION_LAG = timedelta(minutes=1)
ENTITIES = (
    EntityKey("route-1", "dir-A", "stop-1"),
    EntityKey("route-1", "dir-B", "stop-2"),
)


class Recorder:
    """A model that answers a constant and remembers exactly what it was shown."""

    def __init__(self, name, answer):
        self.name = name
        self.version = "0.1.0"
        self.answer = answer
        self.fold_ids = []
        self.row_keys = []
        self.fold_identities = []
        self.data_identities = []

    def predict(self, fold, data):
        self.fold_ids.append(fold.fold_id)
        self.row_keys.append(
            tuple(
                (row["route_id"], row["direction_id"], row["stop_id"], row["bucket_start"])
                for row in data.test_features
            )
        )
        self.fold_identities.append(id(fold))
        self.data_identities.append(id(data))
        return [self.answer] * data.rows


def observation(entity, moment):
    return Observation(
        entity=entity,
        event_at=moment,
        available_at=moment + PUBLICATION_LAG,
        target=TARGET,
        unit=UNIT,
    )


def dataset():
    start = datetime(2023, 1, 1, 9, tzinfo=MOSCOW)
    observations = tuple(
        observation(entity, start + timedelta(days=offset))
        for offset in range(0, 1090, 3)
        for entity in ENTITIES
    )
    coverage = CoverageCalendar.from_range(date(2023, 1, 1), date(2026, 1, 1))
    return BacktestData.of(ENTITIES, observations, coverage)


def experiment():
    return BacktestConfig(rules=(FoldRules(policy=MONTH_DAY),), target=TARGET, unit=UNIT)


@pytest.fixture
def compared():
    candidate, baseline = Recorder("candidate", 1.0), Recorder("baseline", 0.0)
    outcome = run_backtest(experiment(), dataset(), [candidate, baseline])
    return outcome, candidate, baseline


def test_both_models_see_the_same_fold_ids_in_the_same_order(compared):
    outcome, candidate, baseline = compared

    assert candidate.fold_ids == baseline.fold_ids
    assert tuple(candidate.fold_ids) == outcome.horizons["month"].fold_ids
    assert candidate.fold_ids


def test_both_models_see_the_same_test_rows(compared):
    _, candidate, baseline = compared

    assert candidate.row_keys == baseline.row_keys
    assert all(len(set(keys)) == len(keys) for keys in candidate.row_keys)


def test_both_models_are_handed_the_very_same_fold_and_data_objects(compared):
    _, candidate, baseline = compared

    assert candidate.fold_identities == baseline.fold_identities
    assert candidate.data_identities == baseline.data_identities


def test_the_fold_set_in_the_outcome_is_the_one_the_models_were_scored_on(compared):
    outcome, candidate, _ = compared

    assert [fold.fold_id for fold in outcome.fold_set.for_horizon("month")] == candidate.fold_ids


def test_models_that_answer_differently_get_different_metrics_on_identical_folds(compared):
    outcome, _, _ = compared
    horizon = outcome.horizons["month"]

    assert horizon.totals["candidate"].error_total != horizon.totals["baseline"].error_total
    assert horizon.totals["candidate"].actual_total == horizon.totals["baseline"].actual_total
    assert horizon.totals["candidate"].scored == horizon.totals["baseline"].scored


def test_each_fold_result_scores_every_test_row(compared):
    outcome, _, _ = compared

    for name in ("candidate", "baseline"):
        assert outcome.horizons["month"].results[name]
        for result in outcome.horizons["month"].results[name]:
            assert result.rows == result.metrics.scored
            assert result.predictions_digest


def test_the_same_model_on_the_same_folds_produces_the_same_prediction_digest():
    digests = [
        [
            item.predictions_digest
            for item in run_backtest(
                experiment(), dataset(), [Recorder("candidate", 1.0)]
            ).horizons["month"].results["candidate"]
        ]
        for _ in range(2)
    ]

    assert digests[0] == digests[1]
    assert digests[0]


class Short:
    name = "short"
    version = "0.1.0"

    def predict(self, fold, data):
        return [0.0] * (data.rows - 1)


class Negative:
    name = "negative"
    version = "0.1.0"

    def predict(self, fold, data):
        return [-1.0] * data.rows


class NotANumber:
    name = "nan"
    version = "0.1.0"

    def predict(self, fold, data):
        return [float("nan")] * data.rows


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (Short(), "predictions"),
        (Negative(), "finite and non-negative"),
        (NotANumber(), "finite and non-negative"),
    ],
)
def test_a_prediction_must_be_what_a_forecast_point_accepts(model, message):
    with pytest.raises(BacktestError, match=message):
        run_backtest(experiment(), dataset(), [model])


def test_fold_wape_agrees_with_the_golden_evaluation_gate():
    actual = [90.0, 140.0, 210.0, 170.0]
    prediction = [94.0, 145.0, 202.0, 176.0]
    cases = [
        ForecastCase(
            id=f"{horizon}-case",
            horizon=horizon,
            scenario="shared",
            actual=actual,
            prediction=prediction,
            baseline_prediction=prediction,
            lower_bound=[value - 20 for value in prediction],
            upper_bound=[value + 20 for value in prediction],
        )
        for horizon in ("day", "month", "year")
    ]

    summary = evaluate(cases)

    ours = fold_metrics(actual, prediction)
    assert ours.wape == pytest.approx(summary.by_horizon["day"].wape)
    assert ours.mae == pytest.approx(summary.by_horizon["day"].mae)


class Vandal:
    """A model that tries to rewrite the input the next model is about to read."""

    name = "vandal"
    version = "0.1.0"

    def __init__(self):
        self.refusals = 0

    def predict(self, fold, data):
        row = data.test_features[0]
        try:
            row["route_id"] = "HACKED"
        except TypeError:
            self.refusals += 1
        try:
            row["features"]["lag_1d"] = 999999.0
        except TypeError:
            self.refusals += 1
        return [0.0] * data.rows


def test_a_model_cannot_rewrite_the_input_the_next_model_reads():
    vandal, victim = Vandal(), Recorder("victim", 0.0)

    run_backtest(experiment(), dataset(), [vandal, victim])

    assert vandal.refusals == 2 * len(victim.row_keys)
    assert all("HACKED" not in keys[0] for keys in victim.row_keys)


def test_the_rows_handed_to_a_model_reject_assignment():
    recorded = []

    class Peek:
        name = "peek"
        version = "0.1.0"

        def predict(self, fold, data):
            recorded.append(data.test_features[0])
            return [0.0] * data.rows

    run_backtest(experiment(), dataset(), [Peek()])

    row = recorded[0]
    with pytest.raises(TypeError):
        row["route_id"] = "HACKED"
    with pytest.raises(TypeError):
        row["features"]["lag_1d"] = 1.0


def test_wape_is_undefined_where_the_gate_refuses_to_score_at_all():
    """The two definitions agree on the ratio and diverge only on zero demand."""
    empty = fold_metrics([0.0, 0.0], [1.0, 2.0])

    assert empty.wape is None
    with pytest.raises(ValueError, match="zero passenger demand"):
        evaluate(
            [
                ForecastCase(
                    id=f"{horizon}-zero",
                    horizon=horizon,
                    scenario="zero",
                    actual=[0.0, 0.0],
                    prediction=[1.0, 2.0],
                    baseline_prediction=[1.0, 2.0],
                    lower_bound=[0.0, 0.0],
                    upper_bound=[4.0, 4.0],
                )
                for horizon in ("day", "month", "year")
            ]
        )


def test_pooling_folds_equals_scoring_their_concatenation():
    first = fold_metrics([90.0, 140.0], [94.0, 145.0])
    second = fold_metrics([210.0, 170.0], [202.0, 176.0])

    pooled = combine([first, second])

    whole = fold_metrics([90.0, 140.0, 210.0, 170.0], [94.0, 145.0, 202.0, 176.0])
    assert pooled.scored == whole.scored
    assert pooled.actual_total == whole.actual_total
    assert pooled.error_total == whole.error_total
    assert pooled.wape == whole.wape

from datetime import date, datetime, timedelta

import pytest

from tramflow_ml.backtest import (
    BacktestConfig,
    BacktestData,
    BacktestError,
    FoldMetrics,
    FoldRules,
    HorizonOutcome,
    run_backtest,
)
from tramflow_ml.features import (
    MONTH_DAY,
    MOSCOW,
    YEAR_MONTH,
    CoverageCalendar,
    EntityKey,
    Observation,
)

TARGET = "synthetic_boardings"
UNIT = "event_count"
PUBLICATION_LAG = timedelta(minutes=1)
ENTITY = EntityKey("route-1", "dir-A", "stop-1")


class Zero:
    name = "zero"
    version = "0.1.0"

    def predict(self, fold, data):
        return [0.0] * data.rows


def observation(moment):
    return Observation(
        entity=ENTITY,
        event_at=moment,
        available_at=moment + PUBLICATION_LAG,
        target=TARGET,
        unit=UNIT,
    )


def events(start, days, every=3):
    moment = datetime(start.year, start.month, start.day, 9, tzinfo=MOSCOW)
    return tuple(observation(moment + timedelta(days=offset)) for offset in range(0, days, every))


def experiment(policy, **kwargs):
    return BacktestConfig(rules=(FoldRules(policy=policy, **kwargs),), target=TARGET, unit=UNIT)


def dataset(start, end, observations):
    return BacktestData.of((ENTITY,), observations, CoverageCalendar.from_range(start, end))


def two_year_dataset():
    return dataset(date(2024, 1, 1), date(2026, 1, 1), events(date(2024, 1, 1), 700))


def three_year_dataset():
    return dataset(date(2023, 1, 1), date(2026, 1, 1), events(date(2023, 1, 1), 1090))


def test_two_years_of_history_cannot_supply_three_year_folds():
    outcome = run_backtest(experiment(YEAR_MONTH), two_year_dataset(), [Zero()])

    horizon = outcome.horizons["year"]
    assert horizon.status == "insufficient_history"
    assert horizon.passed is False
    assert outcome.passed is False


def test_the_refusal_names_the_horizon_the_requirement_and_what_exists():
    outcome = run_backtest(experiment(YEAR_MONTH), two_year_dataset(), [Zero()])

    reason = outcome.horizons["year"].reason
    assert reason.startswith("year:")
    assert "0 eligible origins" in reason
    assert "3 required" in reason
    assert "insufficient_train_history=2" in reason
    assert "no fold was scored" in reason
    assert outcome.reasons == (reason,)


def test_an_insufficient_run_scores_nothing_and_reports_no_totals():
    outcome = run_backtest(experiment(YEAR_MONTH), two_year_dataset(), [Zero()])

    horizon = outcome.horizons["year"]
    assert horizon.results["zero"] == ()
    assert horizon.totals["zero"].scored == 0
    assert horizon.totals["zero"].wape is None


def test_eligible_but_too_few_origins_still_refuses_and_lists_them():
    data = dataset(date(2018, 1, 1), date(2026, 1, 1), events(date(2018, 1, 1), 2900))

    outcome = run_backtest(experiment(YEAR_MONTH, minimum_origins=9), data, [Zero()])

    horizon = outcome.horizons["year"]
    assert horizon.status == "insufficient_history"
    assert horizon.eligible_origins == 4
    assert len(horizon.fold_ids) == 4
    assert "4 eligible origins, 9 required" in horizon.reason


def test_enough_origins_with_demand_is_the_one_passing_state():
    outcome = run_backtest(experiment(MONTH_DAY), three_year_dataset(), [Zero()])

    horizon = outcome.horizons["month"]
    assert horizon.status == "evaluated"
    assert horizon.reason == ""
    assert horizon.eligible_origins >= horizon.required_origins
    assert outcome.passed is True


def test_folds_without_any_demand_are_not_a_pass_either():
    data = dataset(date(2023, 1, 1), date(2026, 1, 1), ())

    outcome = run_backtest(experiment(MONTH_DAY), data, [Zero()])

    horizon = outcome.horizons["month"]
    assert horizon.status == "insufficient_signal"
    assert horizon.passed is False
    assert horizon.origins_with_demand == 0
    assert "WAPE is undefined" in horizon.reason


def test_one_failing_horizon_fails_the_whole_run():
    config = BacktestConfig(
        rules=(FoldRules(policy=MONTH_DAY), FoldRules(policy=YEAR_MONTH)),
        target=TARGET,
        unit=UNIT,
    )

    outcome = run_backtest(config, three_year_dataset(), [Zero()])

    assert outcome.horizons["month"].passed is True
    assert outcome.horizons["year"].passed is False
    assert outcome.passed is False


def test_a_passing_outcome_with_too_few_folds_cannot_be_constructed():
    with pytest.raises(BacktestError, match="cannot pass a requirement"):
        HorizonOutcome(
            horizon="year",
            status="evaluated",
            required_origins=3,
            required_origins_with_demand=3,
            eligible_origins=0,
            origins_with_demand=0,
            reason="",
            fold_ids=(),
            results={},
            totals={},
        )


def test_a_passing_outcome_with_too_few_folds_carrying_demand_cannot_be_constructed():
    with pytest.raises(BacktestError, match="carry demand"):
        HorizonOutcome(
            horizon="year",
            status="evaluated",
            required_origins=3,
            required_origins_with_demand=3,
            eligible_origins=3,
            origins_with_demand=1,
            reason="",
            fold_ids=("a", "b", "c"),
            results={},
            totals={},
        )


def test_a_non_passing_outcome_must_state_a_reason():
    with pytest.raises(BacktestError, match="must state why"):
        HorizonOutcome(
            horizon="year",
            status="insufficient_history",
            required_origins=3,
            required_origins_with_demand=3,
            eligible_origins=0,
            origins_with_demand=0,
            reason="",
            fold_ids=(),
            results={},
            totals={},
        )


def test_a_passing_outcome_carries_no_refusal_reason():
    with pytest.raises(BacktestError, match="carries no refusal reason"):
        HorizonOutcome(
            horizon="year",
            status="evaluated",
            required_origins=1,
            required_origins_with_demand=1,
            eligible_origins=1,
            origins_with_demand=1,
            reason="looked fine to me",
            fold_ids=("a",),
            results={},
            totals={},
        )


def test_a_requirement_of_zero_origins_is_refused_at_the_configuration_boundary():
    with pytest.raises(BacktestError, match="may evaluate"):
        FoldRules(policy=YEAR_MONTH, minimum_origins=0)


def test_a_backtest_needs_a_model():
    with pytest.raises(BacktestError, match="at least one model"):
        run_backtest(experiment(MONTH_DAY), three_year_dataset(), [])


def test_two_models_may_not_share_a_name():
    with pytest.raises(BacktestError, match="names must be unique"):
        run_backtest(experiment(MONTH_DAY), three_year_dataset(), [Zero(), Zero()])


def test_metrics_report_an_undefined_ratio_rather_than_zero():
    empty = FoldMetrics(4, 0.0, 0.0)

    assert empty.wape is None
    assert empty.mae == 0.0


def test_the_demand_gate_is_configured_apart_from_the_origin_gate():
    config = experiment(MONTH_DAY, minimum_origins_with_demand=99)

    outcome = run_backtest(config, three_year_dataset(), [Zero()])

    horizon = outcome.horizons["month"]
    assert horizon.status == "insufficient_signal"
    assert horizon.eligible_origins >= horizon.required_origins
    assert "99 required" in horizon.reason


def test_raising_the_origin_gate_alone_does_not_tighten_the_demand_gate():
    config = experiment(MONTH_DAY, minimum_origins=11, minimum_origins_with_demand=1)

    outcome = run_backtest(config, three_year_dataset(), [Zero()])

    assert outcome.horizons["month"].status == "evaluated"


def test_an_explicit_demand_requirement_reaches_the_configuration_hash():
    plain = experiment(MONTH_DAY)
    loosened = experiment(MONTH_DAY, minimum_origins_with_demand=1)

    assert plain.config_hash != loosened.config_hash


def test_an_integer_and_a_float_ratio_are_one_configuration():
    assert (
        experiment(MONTH_DAY, minimum_label_unit_ratio=1).config_hash
        == experiment(MONTH_DAY, minimum_label_unit_ratio=1.0).config_hash
    )


def test_a_passing_outcome_with_empty_results_cannot_be_constructed():
    with pytest.raises(BacktestError, match="must carry model results"):
        HorizonOutcome(
            horizon="year",
            status="evaluated",
            required_origins=1,
            required_origins_with_demand=1,
            eligible_origins=1,
            origins_with_demand=1,
            reason="",
            fold_ids=("a",),
            results={},
            totals={},
        )


def test_a_passing_outcome_that_scored_fewer_folds_than_it_had_cannot_be_constructed():
    with pytest.raises(BacktestError, match="scored 0 folds"):
        HorizonOutcome(
            horizon="year",
            status="evaluated",
            required_origins=1,
            required_origins_with_demand=1,
            eligible_origins=1,
            origins_with_demand=1,
            reason="",
            fold_ids=("a",),
            results={"zero": ()},
            totals={"zero": FoldMetrics(0, 0.0, 0.0)},
        )

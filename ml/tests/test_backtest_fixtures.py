"""The real path: synthetic fixture -> ingestion -> features -> folds."""

import json
from collections import Counter
from datetime import date

import pytest

from tramflow_ml.backtest import (
    BacktestConfig,
    BacktestData,
    FoldRules,
    build_fold_set,
    run_backtest,
)
from tramflow_ml.features import (
    DAY_HOUR,
    MONTH_DAY,
    YEAR_MONTH,
    CoverageCalendar,
    load_entity_keys,
    load_observations,
)
from tramflow_ml.ingestion import IngestionConfig, ingest
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset

TARGET = "synthetic_boardings"
UNIT = "event_count"
POLICIES = (DAY_HOUR, MONTH_DAY, YEAR_MONTH)


class Zero:
    name = "zero"
    version = "0.1.0"

    def predict(self, fold, data):
        return [0.0] * data.rows


def fixture_runs(root, config, runs=1):
    source = root / "in"
    generate_dataset(config, source)
    outputs = [root / f"out-{index}" for index in range(runs)]
    for output in outputs:
        ingest(IngestionConfig(input=source, output=output))
    report = json.loads((source / "generation.json").read_bytes())
    gaps = [date.fromisoformat(day) for day in report["gap_dates"]]
    return {
        "coverage": CoverageCalendar.from_range(config.start, config.end, gaps=gaps),
        "report": report,
        "entities": load_entity_keys(source / "entities.json"),
        "runs": [load_observations(output / "validations.jsonl") for output in outputs],
    }


def experiment(*policies, **kwargs):
    return BacktestConfig(
        rules=tuple(FoldRules(policy=policy, **kwargs) for policy in policies),
        target=TARGET,
        unit=UNIT,
    )


def reasons(fold_set, horizon):
    return Counter(item.reason for item in fold_set.refusals_for(horizon))


def data_of(fixture, index=0):
    return BacktestData.of(fixture["entities"], fixture["runs"][index], fixture["coverage"])


@pytest.fixture(scope="module")
def gapped(tmp_path_factory):
    return fixture_runs(tmp_path_factory.mktemp("gapped"), SyntheticConfig(), runs=2)


@pytest.fixture(scope="module")
def gapless(tmp_path_factory):
    return fixture_runs(
        tmp_path_factory.mktemp("gapless"), SyntheticConfig(gap_every_days=0, events=2000)
    )


def test_the_default_fixture_states_its_gaps(gapped):
    assert len(gapped["report"]["gap_dates"]) == 56
    assert gapped["report"]["counts"]["unique_validations"] == 64
    assert len(gapped["entities"]) == 12
    assert len(gapped["runs"][0]) == 64


def test_gap_days_cost_day_folds_one_for_one(gapped):
    fold_set = build_fold_set(experiment(DAY_HOUR), gapped["coverage"])

    assert len(fold_set.for_horizon("day")) == 667
    assert reasons(fold_set, "day") == Counter(
        {"incomplete_labels": 56, "insufficient_train_history": 8}
    )


def test_a_monthly_gap_leaves_no_month_fold_at_all(gapped):
    fold_set = build_fold_set(experiment(MONTH_DAY), gapped["coverage"])

    assert fold_set.for_horizon("month") == ()
    assert reasons(fold_set, "month") == Counter({"incomplete_labels": 24})


def test_a_diluted_year_is_refused_and_says_by_how_much(gapped):
    fold_set = build_fold_set(experiment(YEAR_MONTH), gapped["coverage"])

    assert fold_set.for_horizon("year") == ()
    assert reasons(fold_set, "year") == Counter({"diluted_labels": 2})
    assert "338 of 366" in fold_set.refusals_for("year")[0].detail


def test_the_gapped_fixture_cannot_pass_a_year_backtest(gapped):
    outcome = run_backtest(experiment(YEAR_MONTH), data_of(gapped), [Zero()])

    horizon = outcome.horizons["year"]
    assert horizon.status == "insufficient_history"
    assert outcome.passed is False
    assert "diluted_labels=2" in horizon.reason


def test_a_gapless_two_year_fixture_yields_month_folds_but_no_year_fold(gapless):
    fold_set = build_fold_set(experiment(*POLICIES), gapless["coverage"])

    assert len(fold_set.for_horizon("day")) == 723
    assert len(fold_set.for_horizon("month")) == 11
    assert fold_set.for_horizon("year") == ()
    assert reasons(fold_set, "year") == Counter({"insufficient_train_history": 2})


def test_the_gapless_month_backtest_scores_every_eligible_fold(gapless):
    outcome = run_backtest(experiment(MONTH_DAY), data_of(gapless), [Zero()])

    horizon = outcome.horizons["month"]
    assert horizon.status == "evaluated"
    assert horizon.eligible_origins == 11
    assert horizon.origins_with_demand == 11
    assert horizon.totals["zero"].actual_total == sum(
        item.metrics.actual_total for item in horizon.results["zero"]
    )


def test_a_zero_forecast_carries_the_whole_demand_as_its_error(gapless):
    outcome = run_backtest(experiment(MONTH_DAY), data_of(gapless), [Zero()])

    totals = outcome.horizons["month"].totals["zero"]
    assert totals.error_total == totals.actual_total
    assert totals.wape == 1.0


def test_every_eligible_fold_lies_inside_the_stated_coverage(gapless):
    fold_set = build_fold_set(experiment(*POLICIES), gapless["coverage"])

    for policy in POLICIES:
        assert fold_set.for_horizon(policy.horizon) or policy is YEAR_MONTH
        for fold in fold_set.for_horizon(policy.horizon):
            assert fold.test.end <= fold_set.span.end
            assert fold.train.start >= fold_set.span.start
            assert fold.label_covered_units == fold.label_total_units


def test_two_independent_ingestion_runs_produce_the_same_manifest(gapped):
    first, second = (
        run_backtest(experiment(MONTH_DAY), data_of(gapped, index), [Zero()]).manifest
        for index in (0, 1)
    )

    assert first.data_hash == second.data_hash
    assert first.fold_hash == second.fold_hash
    assert first.manifest_hash == second.manifest_hash


def test_repeating_one_run_reproduces_every_hash(gapless):
    first, second = (
        run_backtest(experiment(MONTH_DAY), data_of(gapless), [Zero()]).manifest
        for _ in range(2)
    )

    assert first.manifest_hash == second.manifest_hash
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


class One:
    name = "constant-one"
    version = "0.1.0"

    def predict(self, fold, data):
        return [1.0] * data.rows


PUBLISHED_MODELS = (Zero, One)
PUBLISHED_HASHES = {
    "config_hash": "da8ffcae72422a69b31add316766c715fe5e1929ed88df40ec2f40fa9073ee53",
    "data_hash": "bc572e95e28d52341e3283f00330ba38f198b27c9170d0ff9a6b834ea00d1704",
    "fold_hash": "755ca80bd8af1e72b570fae7abee5b2c40994619a011a919c05b96868317ec39",
    "manifest_hash": "21a3b973a5ee94eeea49c0a3eaca2dd73884e8f349d96d79b0e18292655797f9",
}


def test_the_published_manifest_hashes_reproduce(gapless):
    """Pins the row published in ml/README.md and the ExecPlan.

    `manifest_hash` covers the model names and versions, so the model set is part of the
    evidence and is named here: zero@0.1.0 then constant-one@0.1.0, in that order. Two
    runs agreeing with each other proves determinism but not that a reader can reproduce
    a published number; only a pinned value does that.
    """
    models = [factory() for factory in PUBLISHED_MODELS]

    manifest = run_backtest(experiment(MONTH_DAY), data_of(gapless), models).manifest

    assert [model.to_dict() for model in manifest.models] == [
        {"name": "zero", "version": "0.1.0"},
        {"name": "constant-one", "version": "0.1.0"},
    ]
    assert {name: getattr(manifest, name) for name in PUBLISHED_HASHES} == PUBLISHED_HASHES


def test_the_model_order_is_part_of_the_published_hash(gapless):
    swapped = [One(), Zero()]

    manifest = run_backtest(experiment(MONTH_DAY), data_of(gapless), swapped).manifest

    assert manifest.fold_hash == PUBLISHED_HASHES["fold_hash"]
    assert manifest.manifest_hash != PUBLISHED_HASHES["manifest_hash"]

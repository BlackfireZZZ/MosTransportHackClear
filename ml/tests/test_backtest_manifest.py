import json
import re
from datetime import date, datetime, timedelta

import pytest

from tramflow_ml.backtest import (
    BacktestConfig,
    BacktestData,
    BacktestError,
    FoldRules,
    ModelVersion,
    build_fold_set,
    run_backtest,
)
from tramflow_ml.features import MONTH_DAY, MOSCOW, CoverageCalendar, EntityKey, Observation

TARGET = "synthetic_boardings"
UNIT = "event_count"
PUBLICATION_LAG = timedelta(minutes=1)
ENTITY = EntityKey("route-1", "dir-A", "stop-1")
CLOCK_WORDS = re.compile(r"generated_at|created_at|run_at|timestamp|now", re.IGNORECASE)


class Constant:
    def __init__(self, name="candidate", answer=1.0):
        self.name = name
        self.version = "0.1.0"
        self.answer = answer

    def predict(self, fold, data):
        return [self.answer] * data.rows


def observation(moment):
    return Observation(
        entity=ENTITY,
        event_at=moment,
        available_at=moment + PUBLICATION_LAG,
        target=TARGET,
        unit=UNIT,
    )


def observations(days=1090, every=3):
    start = datetime(2023, 1, 1, 9, tzinfo=MOSCOW)
    return tuple(observation(start + timedelta(days=offset)) for offset in range(0, days, every))


def coverage(end=date(2026, 1, 1)):
    return CoverageCalendar.from_range(date(2023, 1, 1), end)


def dataset(events=None, end=date(2026, 1, 1)):
    return BacktestData.of((ENTITY,), observations() if events is None else events, coverage(end))


def experiment(**kwargs):
    return BacktestConfig(rules=(FoldRules(policy=MONTH_DAY, **kwargs),), target=TARGET, unit=UNIT)


def manifest_of(config=None, data=None, models=None):
    return run_backtest(config or experiment(), data or dataset(), models or [Constant()]).manifest


def test_two_runs_of_the_same_experiment_produce_the_same_manifest():
    first, second = manifest_of(), manifest_of()

    assert first.manifest_hash == second.manifest_hash
    assert first.config_hash == second.config_hash
    assert first.data_hash == second.data_hash
    assert first.fold_hash == second.fold_hash
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_a_manifest_names_nothing_that_changes_between_runs():
    payload = json.dumps(manifest_of().to_dict(), sort_keys=True)

    assert not CLOCK_WORDS.search(payload)
    assert "/home/" not in payload
    assert "\\" not in payload


def test_changing_the_embargo_changes_the_configuration_and_the_folds_but_not_the_data():
    plain, embargoed = manifest_of(), manifest_of(config=experiment(embargo_buckets=3))

    assert embargoed.config_hash != plain.config_hash
    assert embargoed.fold_hash != plain.fold_hash
    assert embargoed.manifest_hash != plain.manifest_hash
    assert embargoed.data_hash == plain.data_hash


def test_changing_the_minimum_origin_count_changes_the_configuration_hash():
    plain, stricter = manifest_of(), manifest_of(config=experiment(minimum_origins=5))

    assert stricter.config_hash != plain.config_hash
    assert stricter.data_hash == plain.data_hash


def test_a_resolved_default_reaches_the_configuration_hash():
    implicit = experiment()
    explicit = experiment(minimum_train_buckets=364)
    different = experiment(minimum_train_buckets=400)

    assert implicit.config_hash == explicit.config_hash
    assert implicit.config_hash != different.config_hash


def test_changing_one_event_changes_the_data_hash_alone():
    events = list(observations())
    events[0] = observation(events[0].event_at + timedelta(hours=1))

    moved = manifest_of(data=dataset(tuple(events)))
    plain = manifest_of()

    assert moved.data_hash != plain.data_hash
    assert moved.config_hash == plain.config_hash
    assert moved.manifest_hash != plain.manifest_hash


def test_reordering_the_events_changes_no_hash():
    shuffled = manifest_of(data=dataset(tuple(reversed(observations()))))
    plain = manifest_of()

    assert shuffled.data_hash == plain.data_hash
    assert shuffled.manifest_hash == plain.manifest_hash


def test_changing_the_coverage_statement_changes_the_data_and_the_folds():
    shorter = manifest_of(data=dataset(observations(days=1000), end=date(2025, 11, 1)))
    plain = manifest_of()

    assert shorter.data_hash != plain.data_hash
    assert shorter.fold_hash != plain.fold_hash


def test_a_different_answer_changes_the_manifest_but_not_the_folds():
    other = manifest_of(models=[Constant(answer=5.0)])
    plain = manifest_of()

    assert other.fold_hash == plain.fold_hash
    assert other.config_hash == plain.config_hash
    assert other.manifest_hash != plain.manifest_hash


def test_a_different_model_version_changes_the_manifest():
    renamed = Constant()
    renamed.version = "0.2.0"

    assert manifest_of(models=[renamed]).manifest_hash != manifest_of().manifest_hash


def test_the_manifest_hash_covers_every_other_hash():
    manifest = manifest_of()
    body = manifest.body()

    assert body["config_hash"] == manifest.config_hash
    assert body["data_hash"] == manifest.data_hash
    assert body["fold_hash"] == manifest.fold_hash
    assert "manifest_hash" not in body
    assert manifest.to_dict()["manifest_hash"] == manifest.manifest_hash


def test_the_fold_hash_is_independent_of_the_events():
    config = experiment()

    assert (
        build_fold_set(config, coverage()).fold_hash
        == run_backtest(config, dataset(), [Constant()]).manifest.fold_hash
    )


def test_an_unnamed_model_is_refused():
    with pytest.raises(BacktestError, match="non-empty name and version"):
        ModelVersion("", "0.1.0")


def test_two_configurations_built_separately_hash_alike():
    assert experiment().config_hash == experiment().config_hash

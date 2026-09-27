from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from tramflow_ml.backtest import (
    BacktestConfig,
    BacktestData,
    FoldRules,
    build_fold_set,
    fold_data_for,
    fold_id,
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
COVERAGE_START = date(2023, 1, 1)
COVERAGE_END = date(2026, 1, 1)
ORIGIN = datetime(2025, 6, 1, tzinfo=MOSCOW)
HORIZON_END = datetime(2025, 7, 1, tzinfo=MOSCOW)
LAST_HISTORY_EVENT = datetime(2025, 5, 31, 9, tzinfo=MOSCOW)
PUBLICATION_LAG = timedelta(minutes=1)
ENTITIES = (
    EntityKey("route-1", "dir-A", "stop-1"),
    EntityKey("route-1", "dir-B", "stop-2"),
)


def observation(entity, moment, lag=PUBLICATION_LAG):
    return Observation(
        entity=entity, event_at=moment, available_at=moment + lag, target=TARGET, unit=UNIT
    )


def base_observations():
    start = datetime(2023, 1, 1, 9, tzinfo=MOSCOW)
    events = [
        observation(entity, start + timedelta(days=offset))
        for offset in range(0, 880, 3)
        for entity in ENTITIES
    ]
    events.append(observation(ENTITIES[0], LAST_HISTORY_EVENT))
    return tuple(events)


def coverage():
    return CoverageCalendar.from_range(COVERAGE_START, COVERAGE_END)


def rules(**kwargs):
    return FoldRules(policy=MONTH_DAY, **kwargs)


def config(**kwargs):
    return BacktestConfig(rules=(rules(**kwargs),), target=TARGET, unit=UNIT)


def data_for(observations):
    return BacktestData.of(ENTITIES, observations, coverage())


def the_fold(experiment):
    wanted = fold_id("month", ORIGIN)
    fold_set = build_fold_set(experiment, coverage())
    return next(fold for fold in fold_set.for_horizon("month") if fold.fold_id == wanted)


def view_for(observations, **kwargs):
    experiment = config(**kwargs)
    return fold_data_for(experiment, data_for(observations), rules(**kwargs), the_fold(experiment))


class Zero:
    name = "zero"
    version = "0.1.0"

    def predict(self, fold, data):
        return [0.0] * data.rows


def label_total(observations):
    outcome = run_backtest(config(minimum_origins=1), data_for(observations), [Zero()])
    scored = {item.fold_id: item for item in outcome.horizons["month"].results["zero"]}
    return scored[fold_id("month", ORIGIN)].metrics.actual_total


def after_horizon():
    moment = HORIZON_END + timedelta(hours=9)
    return tuple(
        observation(entity, moment + timedelta(days=offset))
        for offset in range(40)
        for entity in ENTITIES
    )


def inside_horizon():
    moment = ORIGIN + timedelta(hours=9)
    return tuple(
        observation(entity, moment + timedelta(days=offset))
        for offset in range(20)
        for entity in ENTITIES
    )


@pytest.fixture
def baseline():
    return view_for(base_observations())


def test_appending_events_after_the_horizon_leaves_the_fold_byte_identical(baseline):
    perturbed = view_for(base_observations() + after_horizon())

    assert perturbed.feature_digest == baseline.feature_digest
    assert perturbed.test_features == baseline.test_features
    assert perturbed.train_observations == baseline.train_observations
    assert perturbed.validation_observations == baseline.validation_observations


def test_appending_events_inside_the_horizon_moves_labels_but_not_features(baseline):
    events = base_observations() + inside_horizon()

    perturbed = view_for(events)

    assert perturbed.feature_digest == baseline.feature_digest
    assert perturbed.test_features == baseline.test_features
    assert perturbed.train_observations == baseline.train_observations
    assert label_total(base_observations()) < label_total(events)


def test_the_fold_set_is_reproducible_and_never_reads_an_event():
    experiment = config()

    first = build_fold_set(experiment, coverage())
    second = build_fold_set(experiment, coverage())

    assert first.fold_hash == second.fold_hash


def test_moving_an_availability_instant_across_the_cutoff_does_change_the_features(baseline):
    events = list(base_observations())
    late = events[-1]
    assert late.event_at == LAST_HISTORY_EVENT
    events[-1] = replace(late, available_at=datetime(2025, 6, 2, tzinfo=MOSCOW))

    perturbed = view_for(tuple(events))

    assert perturbed.feature_digest != baseline.feature_digest
    assert late in baseline.validation_observations
    assert late not in perturbed.validation_observations


def test_reversing_the_observation_order_changes_nothing(baseline):
    reversed_events = tuple(reversed(base_observations()))

    perturbed = view_for(reversed_events)

    assert perturbed.feature_digest == baseline.feature_digest
    assert perturbed.train_observations == baseline.train_observations
    assert data_for(reversed_events).data_hash == data_for(base_observations()).data_hash


def test_a_model_is_never_handed_the_label_it_is_asked_to_predict(baseline):
    keys = {key for row in baseline.test_features for key in row}

    assert keys == {
        "route_id",
        "direction_id",
        "stop_id",
        "bucket_start",
        "bucket_end",
        "cutoff",
        "features",
    }


def test_the_fold_cutoff_and_the_feature_layer_cutoff_are_one_instant(baseline):
    stamped = {row["cutoff"] for row in baseline.test_features}

    assert stamped == {baseline.fold.cutoff.isoformat()}
    assert baseline.fold.cutoff == ORIGIN


def test_every_train_observation_is_visible_at_the_cutoff(baseline):
    assert baseline.train_observations
    for item in baseline.train_observations:
        assert item.visible_at(baseline.fold.cutoff)
        assert baseline.fold.train.contains(item.event_at)


def test_no_train_or_validation_observation_reaches_into_the_test_window(baseline):
    both = baseline.train_observations + baseline.validation_observations

    assert both
    assert not any(baseline.fold.test.contains(item.event_at) for item in both)


def test_an_embargo_removes_information_the_model_would_otherwise_have(baseline):
    embargoed = view_for(base_observations(), embargo_buckets=7)

    assert embargoed.fold.cutoff == ORIGIN - timedelta(days=7)
    assert embargoed.feature_digest != baseline.feature_digest
    assert len(embargoed.train_observations) < len(baseline.train_observations)


def test_a_model_is_given_no_coverage_calendar(baseline):
    """A CoverageView carries its whole calendar, which answers about the future."""
    assert not hasattr(baseline, "coverage")
    assert not hasattr(baseline, "calendar")


def test_the_fold_a_model_sees_carries_no_label_unit_counts(baseline):
    assert not hasattr(baseline.fold, "label_covered_units")
    assert not hasattr(baseline.fold, "label_total_units")
    assert baseline.fold.cutoff == ORIGIN
    assert baseline.fold.train_covered_buckets == baseline.fold.train_buckets


YEAR_START = date(2016, 1, 1)
YEAR_END = date(2026, 1, 1)
YEAR_ORIGIN = datetime(2025, 1, 1, tzinfo=MOSCOW)
DILUTED_RATIO = 0.5


def year_coverage(*gaps):
    return CoverageCalendar.from_range(YEAR_START, YEAR_END, gaps=gaps)


def year_view(coverage):
    rules = FoldRules(policy=YEAR_MONTH, minimum_label_unit_ratio=DILUTED_RATIO)
    experiment = BacktestConfig(rules=(rules,), target=TARGET, unit=UNIT)
    fold_set = build_fold_set(experiment, coverage)
    fold = next(
        item for item in fold_set.for_horizon("year") if item.origin == YEAR_ORIGIN
    )
    data = BacktestData.of(ENTITIES, (), coverage)
    return fold_data_for(experiment, data, rules, fold)


def test_a_coverage_hole_after_the_cutoff_cannot_move_a_year_folds_features():
    """`*_units` columns exist only at monthly granularity, so only a year fold shows this."""
    hole = tuple(date(2025, 7, 2) + timedelta(days=offset) for offset in range(20))

    plain = year_view(year_coverage())
    holed = year_view(year_coverage(*hole))

    assert min(hole) > YEAR_ORIGIN.date()
    assert any(name.endswith("_units") for name in plain.feature_names)
    assert holed.feature_digest == plain.feature_digest
    assert holed.test_features == plain.test_features


def test_a_coverage_hole_before_the_cutoff_does_move_a_year_folds_features():
    hole = tuple(date(2024, 7, 2) + timedelta(days=offset) for offset in range(25))

    plain = year_view(year_coverage())
    holed = year_view(year_coverage(*hole))

    assert holed.feature_digest != plain.feature_digest

import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tramflow_ml.backtest import (
    BacktestConfig,
    BacktestError,
    Fold,
    FoldRules,
    Window,
    bucket_span,
    build_fold_set,
    candidate_origins,
    data_span,
    fold_id,
    horizon_end,
    period_ceiling,
    period_start_of,
    period_step,
)
from tramflow_ml.features import (
    DAY_HOUR,
    MONTH_DAY,
    MOSCOW,
    YEAR_MONTH,
    CoverageCalendar,
    horizon_buckets,
)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.calendar_v1 import forecast_buckets  # noqa: E402, I001

TARGET = "synthetic_boardings"
UNIT = "event_count"
HOURS_PER_DAY = 24


def moscow(year, month, day, hour=0):
    return datetime(year, month, day, hour, tzinfo=MOSCOW)


def span_coverage(start, end, gaps=()):
    return CoverageCalendar.from_range(start, end, gaps=gaps)


def config_for(*rules):
    return BacktestConfig(rules=tuple(rules), target=TARGET, unit=UNIT)


def eligible_folds(policy, start, end, **rule_kwargs):
    rules = FoldRules(policy=policy, **rule_kwargs)
    fold_set = build_fold_set(config_for(rules), span_coverage(start, end))
    return fold_set.for_horizon(policy.horizon)


@pytest.mark.parametrize(
    ("origin", "horizon"),
    [
        (moscow(2024, 2, 1), "month"),
        (moscow(2025, 1, 1), "month"),
        (moscow(2025, 4, 1), "month"),
        (moscow(2024, 1, 1), "year"),
        (moscow(2025, 1, 1), "year"),
        (moscow(2025, 6, 10), "day"),
        (moscow(2011, 3, 27), "day"),
        (moscow(2014, 10, 26), "day"),
    ],
)
def test_horizon_end_matches_the_calendar_contract(origin, horizon):
    end = horizon_end(origin, horizon)

    assert end == horizon_buckets(origin, horizon)[-1].end
    assert end == forecast_buckets(origin, horizon)[-1][1]


@pytest.mark.parametrize("month", [1, 2, 4, 12])
def test_a_month_horizon_is_calendar_days_not_thirty(month):
    origin = moscow(2024, month, 1)

    end = horizon_end(origin, "month")

    days = (end.astimezone(UTC) - origin.astimezone(UTC)).days
    assert (end == origin + timedelta(days=30)) == (days == 30)


def test_a_year_horizon_is_twelve_calendar_months_not_365_days():
    origin = moscow(2024, 1, 1)

    end = horizon_end(origin, "year")

    assert end == moscow(2025, 1, 1)
    assert (end.astimezone(UTC) - origin.astimezone(UTC)).days == 366


def test_a_day_horizon_on_a_dst_day_is_twenty_four_elapsed_hours():
    origin = moscow(2011, 3, 27)

    end = horizon_end(origin, "day")

    assert end.astimezone(UTC) - origin.astimezone(UTC) == timedelta(hours=HOURS_PER_DAY)
    assert end != moscow(2011, 3, 28)
    assert end == moscow(2011, 3, 28, 1)


@pytest.mark.parametrize(
    ("instant", "horizon"),
    [(moscow(2025, 6, 10), "month"), (moscow(2025, 6, 1), "year")],
)
def test_period_step_refuses_an_unaligned_instant(instant, horizon):
    with pytest.raises(BacktestError, match="not the start"):
        period_step(instant, horizon, 1)


def test_period_ceiling_returns_an_aligned_instant_unchanged():
    assert period_ceiling(moscow(2025, 1, 1), "year") == moscow(2025, 1, 1)
    assert period_ceiling(moscow(2025, 6, 10), "year") == moscow(2026, 1, 1)
    assert period_ceiling(moscow(2025, 6, 10), "month") == moscow(2025, 7, 1)


def test_a_day_period_is_aligned_to_the_hour_because_the_contract_allows_any_hour():
    assert period_start_of(moscow(2025, 6, 10, 7), "day") == moscow(2025, 6, 10, 7)


@pytest.mark.parametrize("policy", [DAY_HOUR, MONTH_DAY, YEAR_MONTH])
def test_candidate_origins_are_period_aligned_and_strictly_increasing(policy):
    span = data_span(span_coverage(date(2024, 1, 1), date(2026, 1, 1)))

    origins = candidate_origins(FoldRules(policy=policy), span)

    assert origins[0] == moscow(2024, 1, 1)
    assert all(
        later.astimezone(UTC) > earlier.astimezone(UTC)
        for earlier, later in zip(origins, origins[1:], strict=False)
    )
    assert all(origin.astimezone(UTC) < span.end.astimezone(UTC) for origin in origins)


def test_candidate_origins_cover_a_year_horizon_once_per_year():
    span = data_span(span_coverage(date(2024, 1, 1), date(2026, 1, 1)))

    origins = candidate_origins(FoldRules(policy=YEAR_MONTH), span)

    assert origins == (moscow(2024, 1, 1), moscow(2025, 1, 1))


def test_origin_stride_skips_whole_periods():
    span = data_span(span_coverage(date(2024, 1, 1), date(2024, 7, 1)))

    origins = candidate_origins(FoldRules(policy=MONTH_DAY, origin_stride_periods=2), span)

    assert origins == (moscow(2024, 1, 1), moscow(2024, 3, 1), moscow(2024, 5, 1))


def test_successive_test_windows_are_contiguous_and_never_overlap():
    folds = eligible_folds(MONTH_DAY, date(2024, 1, 1), date(2026, 1, 1))

    assert len(folds) > 1
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test.end == later.test.start
        assert not earlier.test.contains(later.test.start)


def test_day_origins_stay_contiguous_across_a_dst_transition():
    folds = eligible_folds(DAY_HOUR, date(2011, 3, 20), date(2011, 4, 5), minimum_train_buckets=0)

    assert folds
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test.end == later.origin
    drifted = [fold for fold in folds if fold.origin.hour != 0]
    assert drifted, "after a spring-forward a 24-hour horizon no longer begins at midnight"


def test_every_fold_orders_train_validation_cutoff_and_origin():
    folds = eligible_folds(MONTH_DAY, date(2024, 1, 1), date(2026, 1, 1))

    assert folds
    for fold in folds:
        assert fold.train.end.astimezone(UTC) <= fold.validation.start.astimezone(UTC)
        assert fold.validation.start.astimezone(UTC) < fold.validation.end.astimezone(UTC)
        assert fold.validation.end.astimezone(UTC) <= fold.cutoff.astimezone(UTC)
        assert fold.cutoff.astimezone(UTC) <= fold.origin.astimezone(UTC)
        assert fold.test.start == fold.origin


def test_without_an_embargo_the_cutoff_is_the_origin_and_validation_abuts_it():
    fold = eligible_folds(MONTH_DAY, date(2024, 1, 1), date(2026, 1, 1))[0]

    assert fold.cutoff == fold.origin
    assert fold.validation.end == fold.origin
    assert fold.train.end == fold.validation.start


def test_the_embargo_moves_the_cutoff_and_the_train_boundary():
    folds = {
        item.fold_id: item
        for item in eligible_folds(
            MONTH_DAY, date(2024, 1, 1), date(2026, 1, 1), embargo_buckets=5
        )
    }

    fold = folds[fold_id("month", moscow(2025, 6, 1))]

    assert fold.cutoff == moscow(2025, 5, 27)
    assert fold.validation.end == moscow(2025, 5, 1)
    assert fold.validation.start == moscow(2025, 4, 1)
    assert fold.train.end == moscow(2025, 3, 27)


def test_validation_is_a_whole_period_even_when_the_embargo_is_not():
    fold = next(
        item
        for item in eligible_folds(
            MONTH_DAY, date(2024, 1, 1), date(2026, 1, 1), embargo_buckets=5
        )
        if item.origin == moscow(2025, 6, 1)
    )

    assert horizon_end(fold.validation.start, "month") == fold.validation.end


def test_more_validation_periods_take_more_history():
    one, two = (
        next(
            item
            for item in eligible_folds(
                MONTH_DAY, date(2023, 1, 1), date(2026, 1, 1), validation_periods=periods
            )
            if item.origin == moscow(2025, 6, 1)
        )
        for periods in (1, 2)
    )

    assert one.validation.start == moscow(2025, 5, 1)
    assert two.validation.start == moscow(2025, 4, 1)
    assert two.train_buckets < one.train_buckets


def test_fold_ids_are_keyed_on_the_origin():
    folds = eligible_folds(YEAR_MONTH, date(2018, 1, 1), date(2026, 1, 1))

    assert folds
    assert [item.fold_id for item in folds] == [
        f"year@{item.origin.isoformat()}" for item in folds
    ]
    assert len({item.fold_id for item in folds}) == len(folds)


def test_fold_indexes_number_the_eligible_origins_in_order():
    folds = eligible_folds(MONTH_DAY, date(2024, 1, 1), date(2026, 1, 1))

    assert [item.index for item in folds] == list(range(len(folds)))


@pytest.mark.parametrize(
    ("start", "end", "granularity", "expected"),
    [
        (moscow(2011, 3, 27), moscow(2011, 3, 28), "hourly", 23),
        (moscow(2014, 10, 26), moscow(2014, 10, 27), "hourly", 25),
        (moscow(2024, 2, 1), moscow(2024, 3, 1), "daily", 29),
        (moscow(2024, 1, 1), moscow(2025, 1, 1), "monthly", 12),
    ],
)
def test_bucket_span_counts_whole_buckets(start, end, granularity, expected):
    assert bucket_span(start, end, granularity) == expected


def test_bucket_span_refuses_a_partial_hour():
    with pytest.raises(BacktestError, match="whole number of hours"):
        bucket_span(moscow(2025, 6, 10), moscow(2025, 6, 10) + timedelta(minutes=30), "hourly")


def test_a_fold_whose_validation_overlaps_its_test_cannot_be_constructed():
    origin = moscow(2025, 6, 1)

    with pytest.raises(BacktestError, match="must not overlap"):
        Fold(
            fold_id="month@overlap",
            index=0,
            horizon="month",
            policy_name=MONTH_DAY.name,
            origin=origin,
            cutoff=origin,
            train=Window(moscow(2024, 1, 1), moscow(2025, 5, 1)),
            validation=Window(moscow(2025, 5, 1), moscow(2025, 6, 2)),
            test=Window(origin, moscow(2025, 7, 1)),
            train_buckets=486,
            train_covered_buckets=486,
            test_buckets=30,
            label_covered_units=30,
            label_total_units=30,
        )


def test_a_fold_whose_test_does_not_start_at_the_origin_cannot_be_constructed():
    origin = moscow(2025, 6, 1)

    with pytest.raises(BacktestError, match="must start at the origin"):
        Fold(
            fold_id="month@misplaced",
            index=0,
            horizon="month",
            policy_name=MONTH_DAY.name,
            origin=origin,
            cutoff=origin,
            train=Window(moscow(2024, 1, 1), moscow(2025, 5, 1)),
            validation=Window(moscow(2025, 5, 1), moscow(2025, 6, 1)),
            test=Window(moscow(2025, 7, 1), moscow(2025, 8, 1)),
            train_buckets=486,
            train_covered_buckets=486,
            test_buckets=31,
            label_covered_units=31,
            label_total_units=31,
        )


def test_each_horizon_may_appear_only_once_in_one_configuration():
    with pytest.raises(BacktestError, match="at most once"):
        config_for(FoldRules(policy=MONTH_DAY), FoldRules(policy=MONTH_DAY))


def test_a_configuration_refuses_a_passenger_valued_target():
    with pytest.raises(ValueError, match="observations carry no magnitude"):
        BacktestConfig(
            rules=(FoldRules(policy=MONTH_DAY),), target="onboard_load", unit="passengers"
        )

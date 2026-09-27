import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from tramflow_ml.backtest import (
    BacktestConfig,
    FoldRules,
    build_fold_set,
    candidate_origins,
    data_span,
    horizon_end,
)
from tramflow_ml.features import MONTH_DAY, MOSCOW, YEAR_MONTH, CoverageCalendar, horizon_buckets

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.calendar_v1 import forecast_buckets  # noqa: E402, I001

TARGET = "synthetic_boardings"
UNIT = "event_count"


def moscow(year, month, day):
    return datetime(year, month, day, tzinfo=MOSCOW)


def graded(policy, start, end, gaps=(), **rule_kwargs):
    rules = FoldRules(policy=policy, **rule_kwargs)
    config = BacktestConfig(rules=(rules,), target=TARGET, unit=UNIT)
    coverage = CoverageCalendar.from_range(start, end, gaps=gaps)
    return rules, coverage, build_fold_set(config, coverage)


def refusal_at(fold_set, horizon, origin):
    return next(item for item in fold_set.refusals_for(horizon) if item.origin == origin)


def test_a_year_horizon_needs_a_complete_future_year():
    _, _, fold_set = graded(YEAR_MONTH, date(2018, 1, 1), date(2025, 12, 31))

    refusal = refusal_at(fold_set, "year", moscow(2025, 1, 1))

    assert refusal.reason == "horizon_beyond_data"
    assert "2026-01-01" in refusal.detail
    assert moscow(2025, 1, 1) not in [fold.origin for fold in fold_set.for_horizon("year")]


def test_a_year_horizon_that_ends_exactly_at_the_data_edge_is_a_fold():
    _, _, fold_set = graded(YEAR_MONTH, date(2018, 1, 1), date(2026, 1, 1))

    origins = [fold.origin for fold in fold_set.for_horizon("year")]

    assert moscow(2025, 1, 1) in origins
    assert origins == [moscow(year, 1, 1) for year in (2022, 2023, 2024, 2025)]


def test_a_month_horizon_needs_a_complete_month():
    _, _, fold_set = graded(MONTH_DAY, date(2023, 1, 1), date(2025, 6, 30))

    refusal = refusal_at(fold_set, "month", moscow(2025, 6, 1))

    assert refusal.reason == "horizon_beyond_data"
    assert refusal.horizon == "month"


def test_no_eligible_fold_is_ever_a_truncated_horizon():
    _, _, fold_set = graded(MONTH_DAY, date(2023, 1, 1), date(2025, 6, 30))

    assert fold_set.for_horizon("month")
    for fold in fold_set.for_horizon("month"):
        assert fold.test.end == horizon_end(fold.origin, "month")
        assert fold.test.end == forecast_buckets(fold.origin, "month")[-1][1]
        assert fold.test_buckets == len(horizon_buckets(fold.origin, "month"))
        assert fold.test.end <= fold_set.span.end


def test_a_gap_day_inside_a_month_horizon_makes_it_not_a_fold():
    _, _, fold_set = graded(
        MONTH_DAY, date(2023, 1, 1), date(2026, 1, 1), gaps=(date(2025, 6, 14),)
    )

    refusal = refusal_at(fold_set, "month", moscow(2025, 6, 1))

    assert refusal.reason == "incomplete_labels"
    assert refusal.detail == "1 of 30 horizon buckets have no label"


def test_a_gap_day_outside_the_horizon_leaves_the_neighbouring_fold_alone():
    _, _, fold_set = graded(
        MONTH_DAY, date(2023, 1, 1), date(2026, 1, 1), gaps=(date(2025, 6, 14),)
    )

    origins = [fold.origin for fold in fold_set.for_horizon("month")]

    assert moscow(2025, 7, 1) in origins
    assert moscow(2025, 6, 1) not in origins


def test_a_monthly_label_bucket_survives_a_gap_but_is_reported_diluted():
    gaps = tuple(date(2025, 6, day) for day in range(2, 30))

    _, _, fold_set = graded(YEAR_MONTH, date(2018, 1, 1), date(2026, 1, 1), gaps=gaps)

    refusal = refusal_at(fold_set, "year", moscow(2025, 1, 1))
    assert refusal.reason == "diluted_labels"
    assert "337 of 365" in refusal.detail


def test_a_lowered_dilution_threshold_accepts_the_same_year():
    gaps = tuple(date(2025, 6, day) for day in range(2, 30))

    _, _, fold_set = graded(
        YEAR_MONTH,
        date(2018, 1, 1),
        date(2026, 1, 1),
        gaps=gaps,
        minimum_label_unit_ratio=0.9,
    )

    fold = next(item for item in fold_set.for_horizon("year") if item.origin == moscow(2025, 1, 1))
    assert fold.label_covered_units == 337
    assert fold.label_total_units == 365
    assert fold.label_unit_ratio < 1.0


def test_an_origin_with_no_history_behind_it_is_not_a_fold():
    _, _, fold_set = graded(YEAR_MONTH, date(2018, 1, 1), date(2026, 1, 1))

    refusal = refusal_at(fold_set, "year", moscow(2020, 1, 1))

    assert refusal.reason == "insufficient_train_history"
    assert "36 required" in refusal.detail


def test_the_earliest_origin_is_refused_because_validation_would_precede_the_data():
    _, _, fold_set = graded(YEAR_MONTH, date(2018, 1, 1), date(2026, 1, 1))

    refusal = refusal_at(fold_set, "year", moscow(2018, 1, 1))

    assert refusal.reason == "insufficient_train_history"
    assert "before the data" in refusal.detail


@pytest.mark.parametrize("policy", [MONTH_DAY, YEAR_MONTH])
def test_every_candidate_origin_is_recorded_exactly_once(policy):
    rules, coverage, fold_set = graded(
        policy, date(2023, 1, 1), date(2026, 1, 1), gaps=(date(2025, 6, 14),)
    )

    candidates = candidate_origins(rules, data_span(coverage))

    graded_origins = [fold.origin for fold in fold_set.for_horizon(policy.horizon)] + [
        item.origin for item in fold_set.refusals_for(policy.horizon)
    ]
    assert sorted(graded_origins) == sorted(candidates)
    assert len(graded_origins) == len(candidates)


def test_a_refusal_names_its_horizon_and_origin():
    _, _, fold_set = graded(MONTH_DAY, date(2023, 1, 1), date(2025, 6, 30))

    payload = refusal_at(fold_set, "month", moscow(2025, 6, 1)).to_dict()

    assert payload["horizon"] == "month"
    assert payload["origin"] == moscow(2025, 6, 1).isoformat()
    assert payload["reason"] == "horizon_beyond_data"
    assert payload["detail"]


def test_label_units_count_the_horizon_civil_dates():
    _, _, fold_set = graded(YEAR_MONTH, date(2018, 1, 1), date(2026, 1, 1))

    fold = next(item for item in fold_set.for_horizon("year") if item.origin == moscow(2024, 1, 1))

    assert fold.label_total_units == 366
    assert fold.label_covered_units == 366
    assert fold.label_unit_ratio == 1.0


def sparse_coverage():
    """One ancient date, then a recent block: long on the calendar, short on data."""
    recent = [date(2025, 4, 30) + timedelta(days=offset) for offset in range(246)]
    return CoverageCalendar.from_dates([date(2023, 1, 1), *recent])


def sparse_fold_set(policy, **rule_kwargs):
    rules = FoldRules(policy=policy, **rule_kwargs)
    config = BacktestConfig(rules=(rules,), target=TARGET, unit=UNIT)
    return build_fold_set(config, sparse_coverage())


def test_train_history_is_counted_in_covered_buckets_not_calendar_distance():
    fold_set = sparse_fold_set(MONTH_DAY)

    refusal = refusal_at(fold_set, "month", moscow(2025, 5, 1))

    assert refusal.reason == "insufficient_train_history"
    assert refusal.detail == "1 of 821 train buckets are covered, 364 required"


def test_a_long_but_hollow_history_yields_no_month_fold_at_all():
    fold_set = sparse_fold_set(MONTH_DAY)

    assert fold_set.for_horizon("month") == ()
    assert {item.reason for item in fold_set.refusals_for("month")} == {
        "incomplete_labels",
        "insufficient_train_history",
    }


def test_the_covered_count_is_recorded_on_every_fold():
    _, _, fold_set = graded(MONTH_DAY, date(2023, 1, 1), date(2026, 1, 1))

    assert fold_set.for_horizon("month")
    for fold in fold_set.for_horizon("month"):
        assert fold.train_covered_buckets == fold.train_buckets
        assert fold.train_covered_buckets >= 364


def test_holes_reduce_the_covered_count_without_shortening_the_window():
    gaps = tuple(date(2024, 3, 1) + timedelta(days=offset) for offset in range(20))

    _, _, fold_set = graded(MONTH_DAY, date(2023, 1, 1), date(2026, 1, 1), gaps=gaps)

    fold = next(
        item for item in fold_set.for_horizon("month") if item.origin == moscow(2025, 8, 1)
    )
    assert fold.train_covered_buckets == fold.train_buckets - 20

from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from tramflow_ml.features import (
    DAY_HOUR,
    MONTH_DAY,
    MOSCOW,
    POLICIES,
    YEAR_MONTH,
    CoverageCalendar,
    EntityKey,
    FeatureRequest,
    Observation,
    anchor_start,
    build_features,
    carries_unit_ratio,
    feature_names,
)

TARGET = "synthetic_boardings"
UNIT = "event_count"
ENTITY = EntityKey("route-1", "dir-0", "stop-1")
OTHER = EntityKey("route-1", "dir-1", "stop-1")
CUTOFF_LEAD_HOURS = 6
MONTHS_PER_YEAR = 12
SEASONAL_PERIODS = 3


def moscow(text):
    return datetime.fromisoformat(text).replace(tzinfo=MOSCOW)


def observation(text, entity=ENTITY):
    event_at = moscow(text)
    return Observation(entity, event_at, event_at + timedelta(minutes=1), TARGET, UNIT)


def repeated(text, times, entity=ENTITY):
    return tuple(observation(text, entity) for _ in range(times))


def build(policy, origin, observations, coverage):
    return build_features(
        FeatureRequest(
            policy=policy,
            origin=moscow(origin),
            entities=(ENTITY, OTHER),
            observations=tuple(observations),
            coverage=coverage,
            target=TARGET,
            unit=UNIT,
        )
    )


def row_at(built, start, entity=ENTITY):
    return next(
        row for row in built.rows if row.entity == entity and row.bucket.start == moscow(start)
    )


def window_feature(built, name, entity=ENTITY):
    """Rolling and seasonal columns are constant per entity, so any row of it will do."""
    return next(row.features[name] for row in built.rows if row.entity == entity)


@pytest.fixture
def two_years():
    return CoverageCalendar.from_range(date(2024, 1, 1), date(2026, 1, 1))


def test_the_three_named_cutoff_policies_exist_with_matching_granularity():
    assert sorted(POLICIES) == ["day/hour", "month/day", "year/month"]
    assert (DAY_HOUR.granularity, MONTH_DAY.granularity, YEAR_MONTH.granularity) == (
        "hourly",
        "daily",
        "monthly",
    )


def test_a_lag_reaching_into_the_horizon_is_missing():
    coverage = CoverageCalendar.from_range(date(2024, 5, 1), date(2024, 5, 10))

    built = build(DAY_HOUR, "2024-05-08T00:00", repeated("2024-05-07T23:30", 4), coverage)

    assert row_at(built, "2024-05-08T00:00").features["lag_1h"] == 4.0
    assert row_at(built, "2024-05-08T01:00").features["lag_1h"] is None
    assert row_at(built, "2024-05-08T05:00").features["lag_2h"] is None
    assert row_at(built, "2024-05-08T05:00").features["lag_24h"] == 0.0


def test_daily_lags_land_on_the_leap_day(two_years):
    built = build(MONTH_DAY, "2024-03-01T00:00", repeated("2024-02-29T10:00", 5), two_years)

    assert row_at(built, "2024-03-01T00:00").features["lag_1d"] == 5.0
    assert row_at(built, "2024-03-02T00:00").features["lag_1d"] is None


def test_monthly_lags_land_on_the_same_calendar_month_a_year_earlier(two_years):
    events = (*repeated("2024-02-14T10:00", 3), *repeated("2024-03-14T10:00", 7))

    built = build(YEAR_MONTH, "2025-01-01T00:00", events, two_years)

    assert row_at(built, "2025-02-01T00:00").features["lag_12m"] == 3.0
    assert row_at(built, "2025-03-01T00:00").features["lag_12m"] == 7.0
    assert row_at(built, "2025-01-01T00:00").features["lag_1m"] == 0.0
    assert row_at(built, "2025-02-01T00:00").features["lag_1m"] is None


def test_seasonal_periods_beyond_the_history_stay_missing(two_years):
    built = build(YEAR_MONTH, "2025-01-01T00:00", repeated("2024-02-14T10:00", 3), two_years)

    features = row_at(built, "2025-02-01T00:00").features

    assert features["seasonal_3x12m_mean"] == 3.0
    assert features["seasonal_3x12m_coverage"] == pytest.approx(1 / SEASONAL_PERIODS)


def test_rolling_windows_are_identical_for_every_bucket_of_one_horizon(two_years):
    built = build(DAY_HOUR, "2024-05-08T00:00", repeated("2024-05-07T08:00", 2), two_years)

    sums = {row.features["roll_24h_sum"] for row in built.rows if row.entity == ENTITY}
    means = {row.features["roll_168h_mean"] for row in built.rows if row.entity == ENTITY}

    assert sums == {2.0}
    assert len(means) == 1


def test_the_rolling_anchor_is_the_last_fully_elapsed_bucket():
    cutoff = moscow("2024-05-08T00:00")

    assert anchor_start(DAY_HOUR, cutoff) == moscow("2024-05-07T23:00")
    assert anchor_start(MONTH_DAY, cutoff) == moscow("2024-05-07T00:00")
    assert anchor_start(YEAR_MONTH, cutoff) == moscow("2024-04-01T00:00")
    assert anchor_start(DAY_HOUR, moscow("2024-05-08T13:45")) == moscow("2024-05-08T12:00")


def test_a_cutoff_lead_moves_the_whole_cutoff_earlier(two_years):
    late = replace(DAY_HOUR, name="day/hour late", cutoff_lead_buckets=CUTOFF_LEAD_HOURS)
    events = repeated("2024-05-07T23:30", 4)

    prompt = build(DAY_HOUR, "2024-05-08T00:00", events, two_years)
    delayed = build(late, "2024-05-08T00:00", events, two_years)

    assert prompt.header["cutoff"] == "2024-05-08T00:00:00+03:00"
    assert delayed.header["cutoff"] == "2024-05-07T18:00:00+03:00"
    assert row_at(prompt, "2024-05-08T00:00").features["lag_1h"] == 4.0
    assert row_at(delayed, "2024-05-08T00:00").features["lag_1h"] is None


def test_feature_names_match_the_header_and_every_row(two_years):
    built = build(MONTH_DAY, "2024-03-01T00:00", (), two_years)

    names = feature_names(MONTH_DAY)

    assert built.feature_names == names
    assert all(tuple(row.features) == names for row in built.rows)
    assert "lag_364d" in names
    assert "roll_28d_coverage" in names


def test_entities_and_buckets_are_ordered_and_complete(two_years):
    built = build(YEAR_MONTH, "2025-01-01T00:00", (), two_years)

    assert built.header["entities"] == 2
    assert built.header["buckets"] == MONTHS_PER_YEAR
    assert len(built.rows) == 2 * MONTHS_PER_YEAR
    keys = [(row.entity, row.bucket.start) for row in built.rows]
    assert keys == sorted(keys)


def test_a_diluted_month_is_distinguishable_from_a_complete_one():
    """The finding: 1 of 29 covered days read identically to a complete February."""
    complete = CoverageCalendar.from_range(date(2024, 1, 1), date(2026, 1, 1))
    diluted = CoverageCalendar.from_dates(
        {date(2024, 2, 14)}
        | {day for day in complete.dates if not (day.year == 2024 and day.month == 2)}
    )
    events = repeated("2024-02-14T10:00", 1)

    full = build(YEAR_MONTH, "2025-01-01T00:00", events, complete)
    thin = build(YEAR_MONTH, "2025-01-01T00:00", events, diluted)

    assert row_at(full, "2025-02-01T00:00").features["lag_12m"] == 1.0
    assert row_at(thin, "2025-02-01T00:00").features["lag_12m"] == 1.0
    assert row_at(full, "2025-02-01T00:00").features["lag_12m_units"] == 1.0
    assert row_at(thin, "2025-02-01T00:00").features["lag_12m_units"] == pytest.approx(1 / 29)
    assert full.feature_digest != thin.feature_digest


def test_a_partly_covered_month_lowers_the_window_and_season_unit_ratios():
    complete = CoverageCalendar.from_range(date(2024, 1, 1), date(2026, 1, 1))
    def in_late_may(day):
        return day.year == 2024 and day.month == 5 and day.day > 16

    missing_half_of_may = CoverageCalendar.from_dates(
        {day for day in complete.dates if not in_late_may(day)}
    )

    full = build(YEAR_MONTH, "2025-01-01T00:00", (), complete)
    thin = build(YEAR_MONTH, "2025-01-01T00:00", (), missing_half_of_may)

    assert row_at(full, "2025-05-01T00:00").features["lag_12m_units"] == 1.0
    assert row_at(thin, "2025-05-01T00:00").features["lag_12m_units"] == pytest.approx(16 / 31)
    assert window_feature(full, "roll_12m_coverage") == window_feature(thin, "roll_12m_coverage")
    assert window_feature(thin, "roll_12m_units") < window_feature(full, "roll_12m_units")


def test_the_unit_ratio_is_carried_only_where_a_bucket_spans_several_dates():
    assert carries_unit_ratio(YEAR_MONTH)
    assert not carries_unit_ratio(DAY_HOUR)
    assert not carries_unit_ratio(MONTH_DAY)
    assert "lag_12m_units" in feature_names(YEAR_MONTH)
    assert "roll_12m_units" in feature_names(YEAR_MONTH)
    assert "seasonal_3x12m_units" in feature_names(YEAR_MONTH)
    assert not [name for name in feature_names(DAY_HOUR) if name.endswith("_units")]
    assert not [name for name in feature_names(MONTH_DAY) if name.endswith("_units")]

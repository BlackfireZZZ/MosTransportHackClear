import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tramflow_ml.features import (
    MOSCOW,
    Bucket,
    FeatureError,
    bucket_dates,
    bucket_of,
    bucket_start_of,
    calendar_features,
    horizon_buckets,
    service_date,
    step,
)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.calendar_v1 import forecast_buckets  # noqa: E402, I001
from contracts.calendar_v1 import in_window as contract_in_window  # noqa: E402, I001
from contracts.calendar_v1 import service_date as contract_service_date  # noqa: E402, I001

DAYS_IN_2024 = 366
DAYS_IN_2023 = 365
HOURS_PER_DAY = 24
HOURS_IN_2024 = DAYS_IN_2024 * HOURS_PER_DAY
MONTHS_PER_YEAR = 12


def moscow(text):
    return datetime.fromisoformat(text).replace(tzinfo=MOSCOW)


def pairs(buckets):
    return [(bucket.start, bucket.end) for bucket in buckets]


@pytest.mark.parametrize(
    "origin",
    ["2024-03-10T00:00", "2024-03-10T13:00", "2024-12-31T23:00", "2011-03-27T00:00"],
)
def test_day_buckets_match_the_calendar_contract(origin):
    start = moscow(origin)

    assert pairs(horizon_buckets(start, "day")) == list(forecast_buckets(start, "day"))


@pytest.mark.parametrize(
    ("origin", "days"),
    [
        ("2024-02-01T00:00", 29),
        ("2023-02-01T00:00", 28),
        ("2024-01-01T00:00", 31),
        ("2024-04-01T00:00", 30),
        ("2011-03-01T00:00", 31),
    ],
)
def test_month_buckets_are_calendar_days_not_thirty(origin, days):
    start = moscow(origin)

    buckets = horizon_buckets(start, "month")

    assert len(buckets) == days
    assert pairs(buckets) == list(forecast_buckets(start, "month"))
    assert (buckets[-1].end == start + timedelta(days=30)) == (days == 30)


@pytest.mark.parametrize(("origin", "span"), [("2024-01-01T00:00", 366), ("2023-01-01T00:00", 365)])
def test_year_buckets_are_twelve_calendar_months(origin, span):
    start = moscow(origin)

    buckets = horizon_buckets(start, "year")

    assert len(buckets) == MONTHS_PER_YEAR
    assert pairs(buckets) == list(forecast_buckets(start, "year"))
    assert sum(len(bucket_dates(bucket)) for bucket in buckets) == span
    assert buckets[-1].end == start.replace(year=start.year + 1)
    assert buckets[-1].end != start + timedelta(days=360)


def test_year_of_monthly_buckets_covers_every_hour_of_a_leap_year():
    start = moscow("2024-01-01T00:00")

    buckets = horizon_buckets(start, "year")

    assert sum(bucket.elapsed_hours for bucket in buckets) == HOURS_IN_2024
    assert DAYS_IN_2024 != DAYS_IN_2023


def test_monthly_step_is_a_calendar_month_in_both_directions():
    january = moscow("2024-01-01T00:00")

    assert step(january, "monthly", 1) == moscow("2024-02-01T00:00")
    assert step(january, "monthly", 1) != january + timedelta(days=30)
    assert step(january, "monthly", -1) == moscow("2023-12-01T00:00")
    assert step(january, "monthly", MONTHS_PER_YEAR) == moscow("2025-01-01T00:00")
    assert step(moscow("2024-03-01T00:00"), "monthly", -1) == moscow("2024-02-01T00:00")


def test_daily_step_crosses_month_and_leap_boundaries():
    assert step(moscow("2024-12-31T00:00"), "daily", 1) == moscow("2025-01-01T00:00")
    assert step(moscow("2024-02-28T00:00"), "daily", 1) == moscow("2024-02-29T00:00")
    assert step(moscow("2023-02-28T00:00"), "daily", 1) == moscow("2023-03-01T00:00")
    assert step(moscow("2024-01-01T00:00"), "daily", -1) == moscow("2023-12-31T00:00")


def test_moscow_dst_days_are_not_twenty_four_hours():
    """Moscow lost an hour on 2011-03-27 and gained one on 2014-10-26."""
    short_day = bucket_of(moscow("2011-03-27T12:00"), "daily")
    long_day = bucket_of(moscow("2014-10-26T12:00"), "daily")

    assert short_day.elapsed_hours == 23
    assert long_day.elapsed_hours == 25
    assert calendar_features(short_day, "daily")["bucket_hours"] == 23
    assert calendar_features(long_day, "daily")["bucket_hours"] == 25
    assert bucket_of(moscow("2024-03-31T12:00"), "daily").elapsed_hours == HOURS_PER_DAY


def test_hour_steps_stay_one_elapsed_hour_across_a_dst_transition():
    origin = moscow("2011-03-27T00:00")

    buckets = horizon_buckets(origin, "day")

    assert all(bucket.elapsed_hours == 1 for bucket in buckets)
    elapsed = timedelta(hours=HOURS_PER_DAY)
    assert buckets[-1].end.astimezone(UTC) == origin.astimezone(UTC) + elapsed
    assert buckets[-1].end.isoformat() == "2011-03-28T01:00:00+04:00"
    assert buckets[-1].end != origin + elapsed


def test_hourly_bucket_truncates_and_is_half_open():
    instant = moscow("2024-05-06T08:41:37")

    bucket = bucket_of(instant, "hourly")

    assert bucket.start == moscow("2024-05-06T08:00")
    assert bucket.end == moscow("2024-05-06T09:00")
    assert contract_in_window(instant, bucket.start, bucket.end)
    assert contract_in_window(bucket.start, bucket.start, bucket.end)
    assert not contract_in_window(bucket.end, bucket.start, bucket.end)


def test_bucket_start_agrees_for_equivalent_instants_in_other_zones():
    utc_instant = datetime(2024, 5, 6, 5, 41, tzinfo=UTC)
    berlin_instant = utc_instant.astimezone(ZoneInfo("Europe/Berlin"))

    assert bucket_start_of(utc_instant, "hourly") == bucket_start_of(berlin_instant, "hourly")
    assert bucket_start_of(utc_instant, "daily") == moscow("2024-05-06T00:00")
    assert bucket_start_of(utc_instant, "monthly") == moscow("2024-05-01T00:00")


def test_service_date_matches_the_calendar_contract():
    late_utc = datetime(2024, 5, 6, 22, 30, tzinfo=UTC)

    assert service_date(late_utc) == contract_service_date(late_utc) == date(2024, 5, 7)
    assert service_date(moscow("2024-05-06T00:00")) == date(2024, 5, 6)


def test_monthly_bucket_dates_list_every_civil_day():
    february = bucket_of(moscow("2024-02-14T09:00"), "monthly")

    days = bucket_dates(february)

    assert len(days) == 29
    assert days[0] == date(2024, 2, 1)
    assert days[-1] == date(2024, 2, 29)


@pytest.mark.parametrize(
    ("origin", "horizon"),
    [
        ("2024-03-10T13:30", "day"),
        ("2024-03-10T00:00", "month"),
        ("2024-03-01T05:00", "month"),
        ("2024-03-01T00:00", "year"),
    ],
)
def test_misaligned_origins_are_refused_like_the_contract(origin, horizon):
    start = moscow(origin)

    with pytest.raises(FeatureError):
        horizon_buckets(start, horizon)
    with pytest.raises(ValueError):
        forecast_buckets(start, horizon)


def test_a_date_without_a_moscow_midnight_is_refused_not_guessed():
    """Any date whose Moscow midnight names no single instant is refused, 1981-04-01 here."""
    with pytest.raises(FeatureError, match="1981-04-01"):
        bucket_of(moscow("1981-04-01T12:00"), "daily")
    with pytest.raises(FeatureError, match="1981-04-01"):
        bucket_of(moscow("1981-04-01T12:00"), "monthly")
    with pytest.raises(FeatureError, match="1981-04-01"):
        step(moscow("1981-03-31T00:00"), "daily", 1)

    assert bucket_of(moscow("1981-04-01T12:00"), "hourly").elapsed_hours == 1
    assert bucket_of(moscow("1981-04-02T12:00"), "daily").elapsed_hours == HOURS_PER_DAY


def test_naive_and_empty_intervals_are_refused():
    with pytest.raises(FeatureError):
        bucket_of(datetime(2024, 5, 6, 8, 0), "hourly")
    with pytest.raises(FeatureError):
        Bucket(moscow("2024-05-06T08:00"), moscow("2024-05-06T08:00"))


def test_a_datetime_already_tagged_moscow_is_still_normalized():
    """``astimezone`` short-circuits on its own zone, so a bare tag can carry a phantom time."""
    tagged = datetime(1981, 4, 1, tzinfo=MOSCOW)
    same_instant = tagged.astimezone(UTC)

    assert tagged.isoformat() == "1981-04-01T00:00:00+03:00"
    assert bucket_start_of(tagged, "hourly") == bucket_start_of(same_instant, "hourly")
    assert bucket_start_of(tagged, "hourly").isoformat() == "1981-04-01T01:00:00+04:00"


@pytest.mark.parametrize("horizon", ["month", "year"])
def test_both_spellings_of_one_instant_are_refused_alike(horizon):
    tagged = datetime(1981, 4, 1, tzinfo=MOSCOW)
    same_instant = tagged.astimezone(UTC)

    with pytest.raises(FeatureError):
        horizon_buckets(tagged, horizon)
    with pytest.raises(FeatureError):
        horizon_buckets(same_instant, horizon)
    with pytest.raises(ValueError):
        forecast_buckets(tagged, horizon)


def test_a_sub_hour_historical_offset_keeps_its_minutes_in_an_hourly_bucket():
    """Moscow ran at +04:31:19 in 1918; hour steps are taken in UTC, not assumed whole."""
    start = bucket_start_of(moscow("1918-06-01T12:00"), "hourly")

    assert start.utcoffset() == timedelta(hours=4, minutes=31, seconds=19)
    assert (start.minute, start.second) == (31, 19)
    assert bucket_of(moscow("1918-06-01T12:00"), "hourly").elapsed_hours == 1

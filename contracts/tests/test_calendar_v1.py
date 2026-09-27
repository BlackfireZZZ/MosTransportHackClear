from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from contracts.calendar_v1 import forecast_buckets, in_window, service_date

MOSCOW = ZoneInfo("Europe/Moscow")


def test_day_starts_at_any_local_hour_and_crosses_year_boundary() -> None:
    origin = datetime(2025, 12, 31, 18, tzinfo=MOSCOW)
    buckets = forecast_buckets(origin, "day")
    assert len(buckets) == 24
    assert buckets[0] == (origin, datetime(2025, 12, 31, 19, tzinfo=MOSCOW))
    assert buckets[-1][1] == datetime(2026, 1, 1, 18, tzinfo=MOSCOW)
    assert all(end - start == timedelta(hours=1) for start, end in buckets)
    assert all(left[1] == right[0] for left, right in zip(buckets[:-1], buckets[1:], strict=True))


@pytest.mark.parametrize(
    ("year", "month", "days", "end_year", "end_month"),
    [
        (2024, 2, 29, 2024, 3),
        (2025, 2, 28, 2025, 3),
        (2026, 4, 30, 2026, 5),
        (2026, 12, 31, 2027, 1),
    ],
)
def test_month_contains_actual_calendar_days(
    year: int,
    month: int,
    days: int,
    end_year: int,
    end_month: int,
) -> None:
    buckets = forecast_buckets(datetime(year, month, 1, tzinfo=MOSCOW), "month")
    assert len(buckets) == days
    assert [start.day for start, _ in buckets] == list(range(1, days + 1))
    assert all(start.hour == end.hour == 0 for start, end in buckets)
    assert buckets[-1][1] == datetime(end_year, end_month, 1, tzinfo=MOSCOW)
    assert all(left[1] == right[0] for left, right in zip(buckets[:-1], buckets[1:], strict=True))


@pytest.mark.parametrize(("year", "february_days"), [(2024, 29), (2025, 28)])
def test_year_contains_twelve_calendar_months(year: int, february_days: int) -> None:
    buckets = forecast_buckets(datetime(year, 1, 1, tzinfo=MOSCOW), "year")
    assert len(buckets) == 12
    assert [start.month for start, _ in buckets] == list(range(1, 13))
    assert buckets[1][1] - buckets[1][0] == timedelta(days=february_days)
    assert buckets[-1][1] == datetime(year + 1, 1, 1, tzinfo=MOSCOW)
    assert all(left[1] == right[0] for left, right in zip(buckets[:-1], buckets[1:], strict=True))


@pytest.mark.parametrize("horizon", ["day", "month", "year"])
def test_equivalent_utc_origin_preserves_buckets(horizon: str) -> None:
    local = datetime(2026, 1, 1, tzinfo=MOSCOW)
    assert forecast_buckets(local, horizon) == forecast_buckets(
        datetime(2025, 12, 31, 21, tzinfo=UTC),
        horizon,
    )


@pytest.mark.parametrize(
    ("origin", "horizon"),
    [
        (datetime(2026, 1, 1), "day"),
        (datetime(2026, 1, 1, tzinfo=MOSCOW), "week"),
        (datetime(2026, 1, 1, 1, 1, tzinfo=MOSCOW), "day"),
        (datetime(2026, 1, 1, second=1, tzinfo=MOSCOW), "day"),
        (datetime(2026, 1, 1, microsecond=1, tzinfo=MOSCOW), "day"),
        (datetime(2026, 1, 2, tzinfo=MOSCOW), "month"),
        (datetime(2026, 1, 1, 1, tzinfo=MOSCOW), "month"),
        (datetime(2026, 2, 1, tzinfo=MOSCOW), "year"),
        (datetime(2026, 1, 1, 1, tzinfo=MOSCOW), "year"),
    ],
)
def test_invalid_origins_and_horizons_fail(origin: datetime, horizon: str) -> None:
    with pytest.raises(ValueError):
        forecast_buckets(origin, horizon)


@pytest.mark.parametrize(
    ("instant", "expected"),
    [
        (datetime(2025, 12, 31, 20, 59, 59, tzinfo=UTC), date(2025, 12, 31)),
        (datetime(2025, 12, 31, 21, tzinfo=UTC), date(2026, 1, 1)),
    ],
)
def test_service_date_rolls_at_moscow_midnight(instant: datetime, expected: date) -> None:
    assert service_date(instant) == expected


def test_service_date_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        service_date(datetime(2026, 1, 1))


def test_window_is_half_open_and_compares_instants() -> None:
    start = datetime(2026, 1, 1, tzinfo=MOSCOW)
    end = datetime(2026, 1, 2, tzinfo=MOSCOW)
    assert in_window(datetime(2025, 12, 31, 21, tzinfo=UTC), start, end)
    assert in_window(end - timedelta(microseconds=1), start, end)
    assert not in_window(datetime(2026, 1, 1, 21, tzinfo=UTC), start, end)
    assert not in_window(start - timedelta(microseconds=1), start, end)


@pytest.mark.parametrize("invalid_argument", ["instant", "start", "end", "empty", "reversed"])
def test_window_rejects_naive_or_unordered_bounds(invalid_argument: str) -> None:
    start = datetime(2026, 1, 1, tzinfo=MOSCOW)
    end = datetime(2026, 1, 2, tzinfo=MOSCOW)
    instant = start
    if invalid_argument == "instant":
        instant = instant.replace(tzinfo=None)
    elif invalid_argument == "start":
        start = start.replace(tzinfo=None)
    elif invalid_argument == "end":
        end = end.replace(tzinfo=None)
    elif invalid_argument == "empty":
        end = start.astimezone(UTC)
    else:
        start, end = end, start
    with pytest.raises(ValueError):
        in_window(instant, start, end)


def test_day_uses_elapsed_hours_across_historical_moscow_offset_change() -> None:
    buckets = forecast_buckets(datetime(2014, 10, 26, tzinfo=MOSCOW), "day")
    assert len(buckets) == 24
    assert buckets[-1][1].astimezone(UTC) - buckets[0][0].astimezone(UTC) == timedelta(
        hours=24,
    )
    assert all(
        end.astimezone(UTC) - start.astimezone(UTC) == timedelta(hours=1) for start, end in buckets
    )


@pytest.mark.parametrize("horizon", ["day", "month", "year"])
def test_unrepresentable_horizon_fails_explicitly(horizon: str) -> None:
    origin = (
        datetime(9999, 1, 1, tzinfo=MOSCOW)
        if horizon == "year"
        else datetime(
            9999,
            12,
            31 if horizon == "day" else 1,
            tzinfo=MOSCOW,
        )
    )
    with pytest.raises(ValueError, match="supported datetime range"):
        forecast_buckets(origin, horizon)


@pytest.mark.parametrize("raw", ["0001-01-01T00:00:00+03:00", "9999-12-31T23:00:00-03:00"])
def test_extreme_offsets_fail_as_validation_errors(raw: str) -> None:
    value = datetime.fromisoformat(raw)
    with pytest.raises(ValueError, match="range"):
        forecast_buckets(value, "day")
    with pytest.raises(ValueError, match="range"):
        service_date(value)

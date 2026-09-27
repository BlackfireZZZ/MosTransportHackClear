"""Provisional calendar.v1 policy; organizer service-day semantics remain unconfirmed."""

from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

MOSCOW = ZoneInfo("Europe/Moscow")


def _utc(instant: datetime) -> datetime:
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("calendar.v1 requires timezone-aware timestamps")
    try:
        return instant.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("instant exceeds the supported datetime range") from error


def forecast_buckets(origin: datetime, horizon: str) -> tuple[tuple[datetime, datetime], ...]:
    """Return contiguous [start, end) buckets in Europe/Moscow for calendar.v1.

    Day accepts any exact Moscow hour and covers 24 elapsed hours. Month requires
    the first day at midnight and covers its calendar days. Year requires January
    1 at midnight and covers twelve calendar months. Equivalent aware instants
    share a result; naive, misaligned or unsupported input raises ValueError.
    """
    try:
        local = _utc(origin).astimezone(MOSCOW)
    except OverflowError as error:
        raise ValueError("origin exceeds the supported calendar range") from error
    if horizon not in {"day", "month", "year"}:
        raise ValueError("unsupported calendar.v1 horizon")
    if local.minute or local.second or local.microsecond:
        raise ValueError("calendar.v1 origin must align to an exact Moscow hour")
    if horizon in {"month", "year"} and (local.day != 1 or local.hour != 0):
        raise ValueError("calendar.v1 month/year origin must be the first day at midnight")
    if horizon == "year" and local.month != 1:
        raise ValueError("calendar.v1 year origin must be January 1")
    try:
        if horizon == "day":
            utc_origin = _utc(local)
            boundaries = tuple(
                (utc_origin + timedelta(hours=hour)).astimezone(MOSCOW) for hour in range(25)
            )
        elif horizon == "month":
            days = monthrange(local.year, local.month)[1]
            boundaries = tuple(local + timedelta(days=day) for day in range(days + 1))
        else:
            boundaries = tuple(
                datetime(local.year, month, 1, tzinfo=MOSCOW) for month in range(1, 13)
            ) + (datetime(local.year + 1, 1, 1, tzinfo=MOSCOW),)
    except (OverflowError, ValueError) as error:
        raise ValueError("calendar.v1 horizon exceeds the supported datetime range") from error
    return tuple(zip(boundaries[:-1], boundaries[1:], strict=True))


def service_date(event_at: datetime) -> date:
    """Return the provisional Moscow civil date, rolling over at local midnight."""
    try:
        return _utc(event_at).astimezone(MOSCOW).date()
    except OverflowError as error:
        raise ValueError("event exceeds the supported calendar range") from error


def in_window(instant: datetime, start: datetime, end: datetime) -> bool:
    """Test [start, end) by instant; reject naive timestamps and start >= end."""
    utc_instant, utc_start, utc_end = (_utc(value) for value in (instant, start, end))
    if utc_start >= utc_end:
        raise ValueError("calendar.v1 window must have start before end")
    return utc_start <= utc_instant < utc_end

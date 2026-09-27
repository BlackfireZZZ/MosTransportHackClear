"""Whole-horizon period arithmetic, delegated to the feature layer's calendar steps.

A *period* is one whole horizon: 24 elapsed hours, one calendar month, twelve calendar
months. It is expressed as a whole number of buckets in a granularity the feature layer
already steps correctly, so nothing here approximates a month as 30 days or a year as
365, and a month step is never a day count.

The period granularity is not the bucket granularity: a month horizon is forecast in
daily buckets but advances by whole calendar months.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType

from tramflow_ml.backtest.records import BacktestError
from tramflow_ml.features import Granularity, Horizon, bucket_start_of, step
from tramflow_ml.features.periods import moscow, validate_origin

HOURS_PER_DAY = 24
MONTHS_PER_YEAR = 12
SECONDS_PER_HOUR = 3600

PERIOD_STEP: Mapping[Horizon, tuple[Granularity, int]] = MappingProxyType(
    {
        "day": ("hourly", HOURS_PER_DAY),
        "month": ("monthly", 1),
        "year": ("monthly", MONTHS_PER_YEAR),
    }
)
PERIOD_ALIGNMENT: Mapping[Horizon, Granularity] = MappingProxyType(
    {"day": "hourly", "month": "monthly", "year": "monthly"}
)


def _period_step(horizon: Horizon) -> tuple[Granularity, int]:
    try:
        return PERIOD_STEP[horizon]
    except KeyError as error:
        raise BacktestError(f"unsupported horizon {horizon!r}") from error


def period_start_of(instant: datetime, horizon: Horizon) -> datetime:
    """The coarsest boundary at or before ``instant`` that may begin a horizon period.

    For ``month`` and ``year`` that is the calendar month or January, because
    ``calendar.v1`` accepts no other origin. For ``day`` it is the exact hour: a day
    horizon may legally start at any exact Moscow hour and covers 24 *elapsed* hours, so
    successive day origins are 24 hours apart and, across a DST transition, stop landing
    on midnight. That is the faithful reading of the contract, and it is what keeps
    successive test windows exactly contiguous instead of overlapping by an hour.
    """
    try:
        alignment = PERIOD_ALIGNMENT[horizon]
    except KeyError as error:
        raise BacktestError(f"unsupported horizon {horizon!r}") from error
    start = bucket_start_of(instant, alignment)
    return step(start, "monthly", 1 - start.month) if horizon == "year" else start


def period_step(instant: datetime, horizon: Horizon, periods: int) -> datetime:
    """Move a period-aligned instant by whole horizon periods.

    A monthly step truncates its input to the month start, so an unaligned instant would
    be silently moved to a different period. Alignment is therefore a precondition here
    rather than something a caller is trusted to remember.
    """
    granularity, size = _period_step(horizon)
    if period_start_of(instant, horizon) != moscow(instant, "period instant"):
        raise BacktestError(f"{instant.isoformat()} is not the start of a {horizon} period")
    return step(instant, granularity, size * periods)


def period_ceiling(instant: datetime, horizon: Horizon) -> datetime:
    """The earliest period boundary at or after ``instant``."""
    start = period_start_of(instant, horizon)
    return start if start == moscow(instant, "period instant") else period_step(start, horizon, 1)


def horizon_end(origin: datetime, horizon: Horizon) -> datetime:
    """End of the horizon a forecast made at ``origin`` covers, exclusive."""
    return period_step(validate_origin(origin, horizon), horizon, 1)


def bucket_ceiling(instant: datetime, granularity: Granularity) -> datetime:
    """The earliest bucket start at or after ``instant``; a partial bucket is excluded."""
    start = bucket_start_of(instant, granularity)
    return start if start == moscow(instant, "bucket instant") else step(start, granularity, 1)


def bucket_span(start: datetime, end: datetime, granularity: Granularity) -> int:
    """Whole buckets in ``[start, end)``; both ends must be bucket starts."""
    if granularity == "hourly":
        return _whole_hours(start, end)
    if granularity == "daily":
        return (moscow(end, "span end").date() - moscow(start, "span start").date()).days
    if granularity == "monthly":
        first, last = moscow(start, "span start"), moscow(end, "span end")
        return (last.year - first.year) * MONTHS_PER_YEAR + (last.month - first.month)
    raise BacktestError(f"unsupported granularity {granularity!r}")


def _whole_hours(start: datetime, end: datetime) -> int:
    elapsed = moscow(end, "span end").astimezone(UTC) - moscow(start, "span start").astimezone(UTC)
    hours, remainder = divmod(elapsed.total_seconds(), SECONDS_PER_HOUR)
    if remainder:
        raise BacktestError("an hourly span must be a whole number of hours")
    return int(hours)

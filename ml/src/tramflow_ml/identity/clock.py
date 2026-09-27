"""Event-time alignment: source clock in, Europe/Moscow out, day shifts recorded."""

from datetime import UTC, datetime, timedelta

from tramflow_ml.identity.config import MOSCOW, SourceClock
from tramflow_ml.identity.types import (
    AlignedTime,
    IdentityError,
    LocalTimeReason,
    UnresolvedLocalTime,
)


class LocalTimeError(IdentityError):
    """A naive wall time falls in a DST gap or overlap of the source zone."""

    def __init__(self, reason: LocalTimeReason) -> None:
        super().__init__(reason)
        self.reason: LocalTimeReason = reason


def localize(clock: SourceClock, value: datetime) -> datetime:
    """Aware values keep their offset; a naive value must name exactly one instant."""
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value
    zone = clock.zone
    local = value.replace(tzinfo=zone, fold=0)
    round_trip = local.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
    if round_trip != value.replace(fold=0):
        raise LocalTimeError("nonexistent_local_time")
    if local.utcoffset() != value.replace(tzinfo=zone, fold=1).utcoffset():
        raise LocalTimeError("ambiguous_local_time")
    return local


def adjust(clock: SourceClock, value: datetime) -> datetime:
    """Apply the offset in UTC so a source DST gap cannot yield a phantom wall time."""
    try:
        shifted = localize(clock, value).astimezone(UTC) + timedelta(seconds=clock.offset_seconds)
        return shifted.astimezone(MOSCOW)
    except OverflowError as error:
        raise IdentityError("timestamp outside the supported datetime range") from error


def align_time(
    clock: SourceClock, event_at: datetime, available_at: datetime | None
) -> AlignedTime | UnresolvedLocalTime:
    try:
        aligned_event_at = adjust(clock, event_at)
    except LocalTimeError as error:
        return UnresolvedLocalTime(event_at, available_at, "event_at", error.reason)
    try:
        aligned_available_at = _availability(clock, aligned_event_at, available_at)
    except LocalTimeError as error:
        return UnresolvedLocalTime(event_at, available_at, "available_at", error.reason)
    source_event_at = localize(clock, event_at)
    return AlignedTime(
        source_event_at=source_event_at,
        event_at=aligned_event_at,
        source_available_at=None if available_at is None else localize(clock, available_at),
        available_at=aligned_available_at,
        offset_seconds=clock.offset_seconds,
        service_day_shifted=source_event_at.astimezone(MOSCOW).date() != aligned_event_at.date(),
    )


def _availability(
    clock: SourceClock, aligned_event_at: datetime, available_at: datetime | None
) -> datetime | None:
    if available_at is not None:
        return adjust(clock, available_at)
    if clock.availability_lag_seconds is None:
        return None
    lag = timedelta(seconds=clock.availability_lag_seconds)
    try:
        return (aligned_event_at.astimezone(UTC) + lag).astimezone(MOSCOW)
    except OverflowError as error:
        raise IdentityError("availability outside the supported datetime range") from error

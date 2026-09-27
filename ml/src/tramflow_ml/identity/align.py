"""Batch entry point: one aligned record per source row, source rows left untouched."""

from collections.abc import Iterable
from datetime import datetime

from tramflow_ml.identity.catalog import CanonicalCatalog
from tramflow_ml.identity.clock import LocalTimeError, adjust, align_time
from tramflow_ml.identity.config import AlignmentConfig, SourceClock
from tramflow_ml.identity.crosswalk import Crosswalk
from tramflow_ml.identity.matching import match_event
from tramflow_ml.identity.types import (
    AlignedEvent,
    AlignedTime,
    IdentityError,
    LocalTimeReason,
    MatchResult,
    SourceEvent,
    Unmatched,
    UnresolvedLocalTime,
)


class AlignmentError(IdentityError):
    """Catalog, crosswalk and clock configuration do not describe the same stream."""


def align_event(
    catalog: CanonicalCatalog,
    crosswalk: Crosswalk,
    config: AlignmentConfig,
    event: SourceEvent,
) -> AlignedEvent:
    clock = config.clocks.get(event.source_id)
    if clock is None:
        raise AlignmentError(f"no clock configured for source {event.source_id!r}")
    if crosswalk.entity_version != catalog.entity_version:
        raise AlignmentError("crosswalk and catalog entity_version differ")
    time = align_time(clock, event.event_at, event.available_at)
    return AlignedEvent(
        event_id=event.event_id,
        source_id=event.source_id,
        entity_version=catalog.entity_version,
        crosswalk_version=crosswalk.crosswalk_version,
        time=time,
        match=_match(catalog, crosswalk, config, clock, event, time),
    )


def align_events(
    catalog: CanonicalCatalog,
    crosswalk: Crosswalk,
    config: AlignmentConfig,
    events: Iterable[SourceEvent],
) -> tuple[AlignedEvent, ...]:
    return tuple(align_event(catalog, crosswalk, config, event) for event in events)


def _match(
    catalog: CanonicalCatalog,
    crosswalk: Crosswalk,
    config: AlignmentConfig,
    clock: SourceClock,
    event: SourceEvent,
    time: AlignedTime | UnresolvedLocalTime,
) -> MatchResult:
    if isinstance(time, UnresolvedLocalTime):
        return Unmatched(time.reason)
    if time.available_at is None:
        return Unmatched("availability_missing")
    if time.available_at < time.event_at:
        return Unmatched("availability_precedes_event")
    fix_at = _fix_at(clock, event.fix_at)
    return match_event(catalog, crosswalk, config, event, time.event_at, fix_at)


def _fix_at(clock: SourceClock, fix_at: datetime | None) -> datetime | LocalTimeReason | None:
    """An unresolvable fix only matters when GPS is the join key; matching decides that."""
    if fix_at is None:
        return None
    try:
        return adjust(clock, fix_at)
    except LocalTimeError as error:
        return error.reason

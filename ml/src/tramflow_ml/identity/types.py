"""Immutable identity records; every join outcome is explicit, never a guess."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MatchKind = Literal["exact_id", "name_route_direction", "geo_nearest"]
LocalTimeReason = Literal["nonexistent_local_time", "ambiguous_local_time"]
UnmatchedReason = Literal[
    "route_missing",
    "unknown_route",
    "direction_missing",
    "unknown_direction",
    "unknown_pattern",
    "vehicle_route_conflict",
    "stop_missing",
    "unknown_stop",
    "stop_not_on_pattern",
    "unknown_stop_name",
    "invalid_position",
    "no_stop_positions",
    "no_stop_within_tolerance",
    "stop_sequence_mismatch",
    "previous_stop_unknown",
    "previous_stop_mismatch",
    "availability_missing",
    "availability_precedes_event",
    "nonexistent_local_time",
    "ambiguous_local_time",
]
AmbiguousField = Literal["stop", "stop_sequence"]


class IdentityError(ValueError):
    """Configuration or payload defect; row-level data defects become outcomes instead."""


@dataclass(frozen=True)
class SourceEvent:
    """One source row before identity resolution; naive timestamps use the source clock."""

    event_id: str
    source_id: str
    event_at: datetime
    available_at: datetime | None = None
    route_id: str | None = None
    direction_id: str | None = None
    stop_id: str | None = None
    stop_name: str | None = None
    stop_sequence: int | None = None
    previous_stop_id: str | None = None
    vehicle_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    fix_at: datetime | None = None


@dataclass(frozen=True)
class Matched:
    kind: MatchKind
    route_id: str
    direction_id: str
    stop_id: str
    stop_sequence: int
    outcome: Literal["matched"] = "matched"


@dataclass(frozen=True)
class Unmatched:
    reason: UnmatchedReason
    outcome: Literal["unmatched"] = "unmatched"


@dataclass(frozen=True)
class Ambiguous:
    field: AmbiguousField
    candidates: tuple[str, ...]
    outcome: Literal["ambiguous"] = "ambiguous"


@dataclass(frozen=True)
class Stale:
    lag_seconds: float
    outcome: Literal["stale"] = "stale"


MatchResult = Matched | Unmatched | Ambiguous | Stale


@dataclass(frozen=True)
class AlignedTime:
    """Source and Europe/Moscow instants side by side; a day shift is never silent."""

    source_event_at: datetime
    event_at: datetime
    source_available_at: datetime | None
    available_at: datetime | None
    offset_seconds: int
    service_day_shifted: bool


@dataclass(frozen=True)
class UnresolvedLocalTime:
    """A naive source wall time that names zero or two instants in the source zone."""

    source_event_at: datetime
    source_available_at: datetime | None
    field: Literal["event_at", "available_at"]
    reason: LocalTimeReason


@dataclass(frozen=True)
class AlignedEvent:
    event_id: str
    source_id: str
    entity_version: str
    crosswalk_version: str
    time: AlignedTime | UnresolvedLocalTime
    match: MatchResult

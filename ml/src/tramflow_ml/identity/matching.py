"""Entity resolution through one crosswalk; ambiguity and conflict are outcomes, not guesses."""

from datetime import datetime

from tramflow_ml.identity.catalog import CanonicalCatalog, CanonicalPattern
from tramflow_ml.identity.config import AlignmentConfig
from tramflow_ml.identity.crosswalk import Crosswalk
from tramflow_ml.identity.geo import GeoPoint, stops_within
from tramflow_ml.identity.types import (
    Ambiguous,
    LocalTimeReason,
    Matched,
    MatchKind,
    MatchResult,
    SourceEvent,
    Stale,
    Unmatched,
)

_StopHit = tuple[str, MatchKind]


def match_event(
    catalog: CanonicalCatalog,
    crosswalk: Crosswalk,
    config: AlignmentConfig,
    event: SourceEvent,
    event_at: datetime,
    fix_at: datetime | LocalTimeReason | None,
) -> MatchResult:
    """Resolve route, direction, stop and visit for one event at its aligned instant."""
    pattern = _resolve_pattern(catalog, crosswalk, event)
    if isinstance(pattern, Unmatched):
        return pattern
    conflict = _vehicle_conflict(crosswalk, event.vehicle_id, pattern.route_id, event_at)
    if conflict is not None:
        return conflict
    hit = _resolve_stop(catalog, crosswalk, config, pattern, event, event_at, fix_at)
    if not isinstance(hit, tuple):
        return hit
    stop_id, kind = hit
    sequence = _resolve_sequence(crosswalk, config, pattern, stop_id, event)
    if not isinstance(sequence, int):
        return sequence
    return Matched(kind, pattern.route_id, pattern.direction_id, stop_id, sequence)


def _resolve_pattern(
    catalog: CanonicalCatalog, crosswalk: Crosswalk, event: SourceEvent
) -> CanonicalPattern | Unmatched:
    if event.route_id is None:
        return Unmatched("route_missing")
    route_id = crosswalk.routes.get(event.route_id)
    if route_id is None:
        return Unmatched("unknown_route")
    if event.direction_id is None:
        return Unmatched("direction_missing")
    direction_id = crosswalk.directions.get(event.direction_id)
    if direction_id is None:
        return Unmatched("unknown_direction")
    pattern = catalog.patterns.get((route_id, direction_id))
    return Unmatched("unknown_pattern") if pattern is None else pattern


def _vehicle_conflict(
    crosswalk: Crosswalk, vehicle_id: str | None, route_id: str, event_at: datetime
) -> Unmatched | None:
    if vehicle_id is None:
        return None
    assignment = crosswalk.assignment_at(vehicle_id, event_at)
    if assignment is not None and assignment.route_id != route_id:
        return Unmatched("vehicle_route_conflict")
    return None


def _resolve_stop(
    catalog: CanonicalCatalog,
    crosswalk: Crosswalk,
    config: AlignmentConfig,
    pattern: CanonicalPattern,
    event: SourceEvent,
    event_at: datetime,
    fix_at: datetime | LocalTimeReason | None,
) -> _StopHit | Unmatched | Ambiguous | Stale:
    if event.stop_id is not None:
        return _stop_by_id(crosswalk, pattern, event.stop_id)
    if event.stop_name is not None:
        return _stop_by_name(catalog, crosswalk, pattern, event.stop_name)
    if event.latitude is None or event.longitude is None:
        return Unmatched("stop_missing")
    try:
        position = GeoPoint(event.latitude, event.longitude)
    except ValueError:
        return Unmatched("invalid_position")
    return _stop_by_position(crosswalk, config, pattern, position, event_at, fix_at)


def _stop_by_id(
    crosswalk: Crosswalk, pattern: CanonicalPattern, source_stop_id: str
) -> _StopHit | Unmatched:
    stop_id = crosswalk.stops.get(source_stop_id)
    if stop_id is None:
        return Unmatched("unknown_stop")
    if stop_id not in pattern.stop_ids:
        return Unmatched("stop_not_on_pattern")
    return (stop_id, "exact_id")


def _stop_by_name(
    catalog: CanonicalCatalog, crosswalk: Crosswalk, pattern: CanonicalPattern, name: str
) -> _StopHit | Unmatched | Ambiguous:
    qualified = crosswalk.stop_names.get((name, pattern.route_id, pattern.direction_id))
    if qualified is not None:
        return (qualified, "name_route_direction")
    candidates = sorted({stop for stop in pattern.stop_ids if catalog.stops[stop].name == name})
    if not candidates:
        return Unmatched("unknown_stop_name")
    if len(candidates) > 1:
        return Ambiguous("stop", tuple(candidates))
    return (candidates[0], "name_route_direction")


def _stop_by_position(
    crosswalk: Crosswalk,
    config: AlignmentConfig,
    pattern: CanonicalPattern,
    position: GeoPoint,
    event_at: datetime,
    fix_at: datetime | LocalTimeReason | None,
) -> _StopHit | Unmatched | Ambiguous | Stale:
    if isinstance(fix_at, str):
        return Unmatched(fix_at)
    if fix_at is not None:
        lag = abs((fix_at - event_at).total_seconds())
        if lag > config.gps_staleness_seconds:
            return Stale(lag)
    candidates = {
        stop: crosswalk.stop_positions[stop]
        for stop in set(pattern.stop_ids)
        if stop in crosswalk.stop_positions
    }
    if not candidates:
        return Unmatched("no_stop_positions")
    hits = stops_within(position, candidates, config.geo_tolerance_metres)
    if not hits:
        return Unmatched("no_stop_within_tolerance")
    if len(hits) > 1:
        return Ambiguous("stop", tuple(stop for _, stop in hits))
    return (hits[0][1], "geo_nearest")


def _resolve_sequence(
    crosswalk: Crosswalk,
    config: AlignmentConfig,
    pattern: CanonicalPattern,
    stop_id: str,
    event: SourceEvent,
) -> int | Unmatched | Ambiguous:
    visits = pattern.visits(stop_id)
    if event.stop_sequence is not None:
        index = event.stop_sequence - config.stop_sequence_base
        return index if index in visits else Unmatched("stop_sequence_mismatch")
    if len(visits) == 1:
        return visits[0]
    if event.previous_stop_id is None:
        return Ambiguous("stop_sequence", _visit_labels(stop_id, visits))
    previous = crosswalk.stops.get(event.previous_stop_id)
    if previous is None:
        return Unmatched("previous_stop_unknown")
    following = tuple(i for i in visits if i > 0 and pattern.stop_ids[i - 1] == previous)
    if not following:
        return Unmatched("previous_stop_mismatch")
    if len(following) > 1:
        return Ambiguous("stop_sequence", _visit_labels(stop_id, following))
    return following[0]


def _visit_labels(stop_id: str, visits: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{stop_id}@{index}" for index in visits)

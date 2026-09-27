"""Explicit canonical-to-OSM mapping, without inferred trip order or direction."""

from dataclasses import dataclass
from enum import StrEnum


class GeometryMappingError(ValueError):
    """A mapping or forecast version cannot be used with this graph snapshot."""


@dataclass(frozen=True, slots=True)
class ForecastEntity:
    route_id: str
    direction_id: str
    stop_id: str


@dataclass(frozen=True, slots=True)
class RouteGeometryLink:
    route_id: str
    osm_route_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StopGeometryLink:
    entity: ForecastEntity
    osm_stop_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ServingGeometryLink:
    route_id: int
    stop_id: int
    direction_id: str
    entity: ForecastEntity


@dataclass(frozen=True, slots=True)
class GeometryCrosswalk:
    mapping_version: str
    entity_version: str
    graph_version: str
    routes: tuple[RouteGeometryLink, ...]
    stops: tuple[StopGeometryLink, ...]
    synthetic: bool
    serving_links: tuple[ServingGeometryLink, ...] = ()


class GeometryMatchStatus(StrEnum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class MappedForecastStop:
    entity: ForecastEntity
    status: GeometryMatchStatus
    reason: str
    candidate_osm_stop_ids: tuple[int, ...]
    osm_stop_id: int | None = None
    longitude: float | None = None
    latitude: float | None = None


@dataclass(frozen=True, slots=True)
class ForecastGeometry:
    entity_version: str
    mapping_version: str
    graph_version: str
    stops: tuple[MappedForecastStop, ...]
    matched_count: int
    unmatched_count: int
    ambiguous_count: int
    synthetic: bool
    route_semantics: str = "unordered_membership"


@dataclass(frozen=True, slots=True)
class ForecastMapPosition:
    stop_id: int
    direction_id: str | None
    status: GeometryMatchStatus
    reason: str
    position_kind: str
    osm_stop_id: int | None = None
    longitude: float | None = None
    latitude: float | None = None


@dataclass(frozen=True, slots=True)
class ForecastMap:
    run_id: str | None
    entity_version: str | None
    graph_version: str | None
    mapping_version: str | None
    synthetic: bool
    status: str
    reason: str
    matched_count: int
    unmatched_count: int
    ambiguous_count: int
    positions: tuple[ForecastMapPosition, ...]

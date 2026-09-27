"""Strict offline crosswalk artifact adapter; no implicit canonical/OSM coercion."""

import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.application.services.forecast_geometry import ForecastGeometryService
from app.application.services.forecast_map import ForecastMapService
from app.domain.forecast import ForecastSnapshot
from app.domain.forecast_geometry import (
    ForecastEntity,
    GeometryCrosswalk,
    GeometryMappingError,
    RouteGeometryLink,
    ServingGeometryLink,
    StopGeometryLink,
)
from app.domain.tram_graph import TramGraphDataError
from app.infrastructure.graph_artifacts import load_active_graph_snapshot
from app.infrastructure.tram_graph_validation import parse_network

Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
OsmId = Annotated[int, Field(gt=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class _Route(_StrictModel):
    route_id: Identifier
    osm_route_refs: list[Identifier]


class _Stop(_StrictModel):
    route_id: Identifier
    direction_id: Identifier
    stop_id: Identifier
    osm_stop_ids: list[OsmId]


class _Entity(_StrictModel):
    route_id: Identifier
    direction_id: Identifier
    stop_id: Identifier


class _ServingLink(_StrictModel):
    route_id: OsmId
    stop_id: OsmId
    direction_id: Identifier
    entity: _Entity


class _Artifact(_StrictModel):
    synthetic: bool
    schema_version: Literal["forecast-geometry.v1"]
    mapping_version: Identifier
    entity_version: Identifier
    graph_version: Identifier
    routes: list[_Route]
    stops: list[_Stop]
    serving_links: list[_ServingLink] = Field(default_factory=list)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GeometryMappingError("duplicate mapping JSON field")
        result[key] = value
    return result


def load_crosswalk(path: Path) -> GeometryCrosswalk:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        artifact = _Artifact.model_validate(raw)
    except (OSError, UnicodeError, ValueError, RecursionError, ValidationError) as exc:
        raise GeometryMappingError("invalid forecast geometry artifact") from exc
    return GeometryCrosswalk(
        artifact.mapping_version,
        artifact.entity_version,
        artifact.graph_version,
        tuple(
            RouteGeometryLink(row.route_id, tuple(row.osm_route_refs)) for row in artifact.routes
        ),
        tuple(
            StopGeometryLink(
                ForecastEntity(row.route_id, row.direction_id, row.stop_id), tuple(row.osm_stop_ids)
            )
            for row in artifact.stops
        ),
        artifact.synthetic,
        tuple(
            ServingGeometryLink(
                row.route_id,
                row.stop_id,
                row.direction_id,
                ForecastEntity(row.entity.route_id, row.entity.direction_id, row.entity.stop_id),
            )
            for row in artifact.serving_links
        ),
    )


def load_geometry_service(mapping_path: Path, graph_root: Path) -> ForecastGeometryService:
    """Only published, checksum-verified snapshots qualify for versioned joins."""
    version, graph, geo = load_active_graph_snapshot(graph_root)
    return ForecastGeometryService(load_crosswalk(mapping_path), parse_network(graph, geo), version)


class FileForecastMapProvider:
    def __init__(self, mapping_path: Path | None, graph_root: Path) -> None:
        self._mapping_path = mapping_path
        self._graph_root = graph_root
        self._service: ForecastMapService | None = None
        self._lock = asyncio.Lock()

    async def enrich(self, snapshot: ForecastSnapshot) -> ForecastSnapshot:
        if self._mapping_path is None:
            return ForecastMapService().enrich(snapshot)
        if self._service is None:
            async with self._lock:
                if self._service is None:
                    try:
                        mapper = await asyncio.to_thread(
                            load_geometry_service, self._mapping_path, self._graph_root
                        )
                        self._service = ForecastMapService(mapper)
                    except (GeometryMappingError, TramGraphDataError):
                        return ForecastMapService(
                            unavailable_reason="mapping_configuration_invalid"
                        ).enrich(snapshot)
        return self._service.enrich(snapshot)

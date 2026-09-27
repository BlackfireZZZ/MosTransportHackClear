"""Geometry enrichment of one immutable serving snapshot, without value inference."""

import math
from collections import Counter
from dataclasses import replace

from app.application.services.forecast_geometry import ForecastGeometryService
from app.domain.forecast import ForecastSnapshot
from app.domain.forecast_geometry import (
    ForecastMap,
    ForecastMapPosition,
    GeometryMappingError,
    GeometryMatchStatus,
)


class ForecastMapService:
    def __init__(
        self,
        mapper: ForecastGeometryService | None = None,
        unavailable_reason: str = "mapping_not_configured",
    ) -> None:
        self._mapper = mapper
        self._unavailable_reason = unavailable_reason
        self._bridge = (
            {
                (row.route_id, row.stop_id, row.direction_id): row.entity
                for row in mapper.crosswalk.serving_links
            }
            if mapper
            else {}
        )
        if mapper:
            links = mapper.crosswalk.serving_links
            if len(self._bridge) != len(links):
                raise GeometryMappingError("duplicate serving bridge key")
            known = {row.entity for row in mapper.crosswalk.stops}
            if any(row.entity not in known for row in links):
                raise GeometryMappingError("serving bridge references unknown canonical mapping")
            if len(set(self._bridge.values())) != len(self._bridge):
                raise GeometryMappingError("serving bridge aliases canonical entities")

    def enrich(self, snapshot: ForecastSnapshot) -> ForecastSnapshot:
        if snapshot.run is None:
            return replace(snapshot, map=None)
        keys = tuple(
            dict.fromkeys((row.stop_id, row.direction_id) for row in snapshot.stop_points or ())
        )
        run = snapshot.run
        reason = self._unavailable_reason
        mapper = self._mapper
        compatible = False
        if mapper:
            if run.identity_namespace != "serving-surrogate-integer":
                reason = "identity_namespace_unsupported"
            elif run.entity_version != mapper.crosswalk.entity_version:
                reason = "entity_version_mismatch"
            elif run.graph_version is None:
                reason = "forecast_graph_version_unavailable"
            elif run.graph_version != mapper.crosswalk.graph_version:
                reason = "graph_version_mismatch"
            elif mapper.synthetic and not run.synthetic:
                reason = "synthetic_mapping_for_real_forecast"
            else:
                compatible = True
                reason = "explicit_crosswalk"
        positions: list[ForecastMapPosition] = []
        stops = {stop.id: stop for stop in snapshot.stops}
        for stop_id, direction_id in keys:
            row = ForecastMapPosition(
                stop_id,
                direction_id,
                GeometryMatchStatus.UNMATCHED,
                reason,
                "unavailable",
            )
            if compatible and mapper and run:
                entity = (
                    self._bridge.get((snapshot.route.id, stop_id, direction_id))
                    if direction_id
                    else None
                )
                if entity is None:
                    row = replace(
                        row,
                        reason="missing_serving_bridge" if direction_id else "aggregated_direction",
                    )
                else:
                    mapped = mapper.map_stops(
                        [entity],
                        entity_version=run.entity_version,
                        graph_version=run.graph_version,
                    ).stops[0]
                    row = replace(
                        row,
                        status=mapped.status,
                        reason=mapped.reason,
                        position_kind="osm" if mapped.osm_stop_id is not None else "unavailable",
                        osm_stop_id=mapped.osm_stop_id,
                        longitude=mapped.longitude,
                        latitude=mapped.latitude,
                    )
            if row.position_kind == "unavailable" and run and run.synthetic:
                stop = stops.get(stop_id)
                if (
                    stop
                    and all(math.isfinite(v) for v in (stop.longitude, stop.latitude))
                    and (-180 <= stop.longitude <= 180 and -90 <= stop.latitude <= 90)
                ):
                    row = replace(
                        row,
                        position_kind="synthetic_demo",
                        longitude=stop.longitude,
                        latitude=stop.latitude,
                    )
            positions.append(row)
        counts = Counter(row.status for row in positions)
        matched = counts[GeometryMatchStatus.MATCHED]
        return replace(
            snapshot,
            map=ForecastMap(
                run_id=run.run_id if run else None,
                entity_version=run.entity_version if run else None,
                graph_version=run.graph_version if run else None,
                mapping_version=mapper.crosswalk.mapping_version if mapper else None,
                synthetic=bool(run and run.synthetic) or bool(mapper and mapper.synthetic),
                status="ready"
                if positions and matched == len(positions)
                else "partial"
                if matched
                else "unavailable",
                reason=reason,
                matched_count=matched,
                unmatched_count=counts[GeometryMatchStatus.UNMATCHED],
                ambiguous_count=counts[GeometryMatchStatus.AMBIGUOUS],
                positions=tuple(positions),
            ),
        )

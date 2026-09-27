"""Join a bounded forecast selection against one immutable mapping/graph pair."""

from collections import Counter
from collections.abc import Sequence

from app.domain.forecast_geometry import (
    ForecastEntity,
    ForecastGeometry,
    GeometryCrosswalk,
    GeometryMappingError,
    GeometryMatchStatus,
    MappedForecastStop,
)
from app.domain.tram_graph import TramNetwork


class ForecastGeometryService:
    def __init__(
        self, crosswalk: GeometryCrosswalk, network: TramNetwork, graph_version: str
    ) -> None:
        if not graph_version or crosswalk.graph_version != graph_version:
            raise GeometryMappingError("mapping graph version does not match loaded snapshot")
        if not crosswalk.entity_version or not crosswalk.mapping_version:
            raise GeometryMappingError("entity and mapping versions are required")
        if type(crosswalk.synthetic) is not bool:
            raise GeometryMappingError("mapping synthetic marker must be a boolean")
        self._crosswalk = crosswalk
        self._network = network
        self._routes = {item.route_id: item.osm_route_refs for item in crosswalk.routes}
        self._stops = {item.entity: item.osm_stop_ids for item in crosswalk.stops}
        if len(self._routes) != len(crosswalk.routes) or len(self._stops) != len(crosswalk.stops):
            raise GeometryMappingError("duplicate canonical mapping key")
        for refs in self._routes.values():
            if len(set(refs)) != len(refs) or any(not ref for ref in refs):
                raise GeometryMappingError("route candidates must be distinct nonempty refs")
        for entity, ids in self._stops.items():
            if not all((entity.route_id, entity.direction_id, entity.stop_id)):
                raise GeometryMappingError("route, direction and stop identifiers are required")
            if len(set(ids)) != len(ids) or any(type(i) is not int or i <= 0 for i in ids):
                raise GeometryMappingError("stop candidates must be distinct positive OSM ids")

    @property
    def crosswalk(self) -> GeometryCrosswalk:
        return self._crosswalk

    @property
    def synthetic(self) -> bool:
        return self._crosswalk.synthetic or self._network.metadata.synthetic

    def map_stops(
        self,
        entities: Sequence[ForecastEntity],
        *,
        entity_version: str,
        graph_version: str | None,
    ) -> ForecastGeometry:
        if entity_version != self._crosswalk.entity_version:
            raise GeometryMappingError("forecast entity version does not match mapping")
        if graph_version != self._crosswalk.graph_version:
            raise GeometryMappingError("forecast graph version does not match mapping")
        rows = tuple(self._map(entity) for entity in dict.fromkeys(entities))
        counts = Counter(row.status for row in rows)
        return ForecastGeometry(
            entity_version=entity_version,
            mapping_version=self._crosswalk.mapping_version,
            graph_version=graph_version,
            stops=rows,
            matched_count=counts[GeometryMatchStatus.MATCHED],
            unmatched_count=counts[GeometryMatchStatus.UNMATCHED],
            ambiguous_count=counts[GeometryMatchStatus.AMBIGUOUS],
            synthetic=self.synthetic,
        )

    def _map(self, entity: ForecastEntity) -> MappedForecastStop:
        ids = self._stops.get(entity, ())
        refs = self._routes.get(entity.route_id, ())
        reason = "missing_stop_mapping"
        status = GeometryMatchStatus.UNMATCHED
        if len(ids) > 1:
            reason, status = "multiple_stop_candidates", GeometryMatchStatus.AMBIGUOUS
        elif ids:
            stop = self._network.stops.get(ids[0])
            if not refs:
                reason = "missing_route_mapping"
            elif any(ref not in self._network.routes for ref in refs):
                reason = "unknown_osm_route"
            elif stop is None:
                reason = "unknown_osm_stop"
            elif not set(refs).intersection(stop.routes):
                reason = "stop_not_on_mapped_route"
            else:
                return MappedForecastStop(
                    entity,
                    GeometryMatchStatus.MATCHED,
                    "explicit_crosswalk",
                    ids,
                    stop.id,
                    stop.longitude,
                    stop.latitude,
                )
        return MappedForecastStop(entity, status, reason, ids)

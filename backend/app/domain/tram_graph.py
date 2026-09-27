"""The Moscow tram network as a framework-free domain model.

Three properties of the data drive every decision in this module and are documented
in `docs/tram-graph.md`:

* The graph is **directed**. A stop pair served both ways is two edges, and the two
  directions can differ in length where the tracks are not parallel.
* Undirected it falls into **two components** (694 and 162 stops) with no track
  between them. That is real geography, not an extraction artefact, so an
  unreachable pair is an ordinary answer and never an error.
* Route refs are **not all numeric** -- ``А``, ``1а``, ``39а``, ``47а``, ``т1`` and
  ``т2`` exist alongside ``1`` and ``16``. They are sorted and compared as strings.

Routing is a plain Dijkstra over ``length_m`` rather than a graph library: 856 nodes
and 919 edges do not justify a new production dependency, and the domain layer must
stay importable without one.
"""

import math
import statistics
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

Coordinate = tuple[float, float]


class GeometryQuality(StrEnum):
    PROVIDED = "provided"
    INFERRED = "inferred"
    SYNTHETIC = "synthetic"


class TramGraphDataError(RuntimeError):
    """The stored graph could not be turned into a usable network."""


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_coordinate(coordinate: Coordinate) -> None:
    if (
        not isinstance(coordinate, (tuple, list))
        or len(coordinate) != 2
        or not all(_finite_number(value) for value in coordinate)
    ):
        raise TramGraphDataError("coordinates must contain two finite numbers")
    longitude, latitude = coordinate
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise TramGraphDataError("coordinates are outside WGS84 ranges")


def _validate_routes(routes: tuple[str, ...]) -> None:
    if not isinstance(routes, tuple) or any(not isinstance(ref, str) for ref in routes):
        raise TramGraphDataError("routes must be a tuple of strings")


def route_sort_key(ref: str) -> tuple[int, int, str]:
    """Order refs 1, 2, 10 rather than 1, 10, 2, with the non-numeric ones last.

    The same ordering `scripts/fetch_tram_graph.py` uses, so refs read the same way
    in the API as in the committed CSVs. Never `int(ref)` without the digit guard.
    """
    try:
        return (0, int(ref), "") if ref.isdecimal() else (1, 0, ref)
    except ValueError as exc:
        raise TramGraphDataError("numeric route reference is too long") from exc


def sorted_refs(refs: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(refs), key=route_sort_key))


@dataclass(frozen=True, slots=True)
class GraphMetadata:
    """Provenance of the committed extract.

    `route_relations` is the PTv2 relation count (73: one relation per direction).
    It is deliberately not the number of distinct refs, which is what `routes` in
    `NetworkStats` counts.
    """

    synthetic: bool = False
    source: str | None = None
    license: str | None = None
    area: str | None = None
    osm_data_timestamp: str | None = None
    generated_at: str | None = None
    route_relations: int | None = None


@dataclass(frozen=True, slots=True)
class TramStop:
    id: int
    name: str
    latitude: float
    longitude: float
    routes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TramEdge:
    """One directed hop between consecutive stops, measured along the rails."""

    source: int
    target: int
    length_m: float
    routes: tuple[str, ...]


class NeighbourDirection(StrEnum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class StopNeighbour:
    id: int
    name: str
    length_m: float
    routes: tuple[str, ...]
    direction: NeighbourDirection


@dataclass(frozen=True, slots=True)
class StopDetail:
    stop: TramStop
    neighbours: tuple[StopNeighbour, ...]


@dataclass(frozen=True, slots=True)
class TramRoute:
    ref: str
    stop_ids: tuple[int, ...]
    edges: tuple[TramEdge, ...]
    length_m: float
    component: int


@dataclass(frozen=True, slots=True)
class RouteDetail:
    ref: str
    stops: tuple[TramStop, ...]
    edges: tuple[TramEdge, ...]
    length_m: float
    component: int


@dataclass(frozen=True, slots=True)
class NetworkComponent:
    size: int
    routes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SegmentLengthSummary:
    min: float
    median: float
    mean: float
    max: float


@dataclass(frozen=True, slots=True)
class NetworkStats:
    stops: int
    edges: int
    routes: int
    route_relations: int | None
    total_length_km: float
    generated_at: str | None
    osm_data_timestamp: str | None
    components: tuple[NetworkComponent, ...]
    degree_histogram: Mapping[str, int]
    segment_length_m: SegmentLengthSummary


@dataclass(frozen=True, slots=True)
class TrackSegment:
    """An edge polyline whose provenance is explicit, including inferred connectors."""

    source: int
    target: int
    length_m: float
    routes: tuple[str, ...]
    coordinates: tuple[Coordinate, ...]
    geometry_quality: GeometryQuality


@dataclass(frozen=True, slots=True)
class NetworkGeometry:
    metadata: GraphMetadata
    filtered_to_route: str | None
    stops: tuple[TramStop, ...]
    segments: tuple[TrackSegment, ...]


def _connected_components(
    stops: Mapping[int, TramStop],
    outgoing: Mapping[int, list[TramEdge]],
    incoming: Mapping[int, list[TramEdge]],
) -> tuple[tuple[int, ...], ...]:
    """Components of the undirected view, largest first, so 0 is the main network."""
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in stops:
        if start in seen:
            continue
        queue = deque([start])
        found: set[int] = set()
        while queue:
            current = queue.popleft()
            if current in found:
                continue
            found.add(current)
            for edge in outgoing.get(current, ()):
                if edge.target not in found:
                    queue.append(edge.target)
            for edge in incoming.get(current, ()):
                if edge.source not in found:
                    queue.append(edge.source)
        seen |= found
        components.append(tuple(sorted(found)))
    components.sort(key=len, reverse=True)
    return tuple(components)


def _index_routes(
    stops: Mapping[int, TramStop],
    edges: Sequence[TramEdge],
    component_of: Mapping[int, int],
) -> dict[str, TramRoute]:
    stop_ids_by_ref: dict[str, set[int]] = {}
    edges_by_ref: dict[str, list[TramEdge]] = {}
    for stop in stops.values():
        for ref in stop.routes:
            stop_ids_by_ref.setdefault(ref, set()).add(stop.id)
    for edge in edges:
        for ref in edge.routes:
            edges_by_ref.setdefault(ref, []).append(edge)

    routes: dict[str, TramRoute] = {}
    for ref in sorted(stop_ids_by_ref, key=route_sort_key):
        stop_ids = stop_ids_by_ref[ref]
        ref_edges = tuple(edges_by_ref.get(ref, ()))
        components = {component_of[stop_id] for stop_id in stop_ids}
        routes[ref] = TramRoute(
            ref=ref,
            stop_ids=tuple(sorted(stop_ids)),
            edges=ref_edges,
            length_m=round(sum(edge.length_m for edge in ref_edges), 1),
            component=min(components) if components else -1,
        )
    return routes


@dataclass(frozen=True, slots=True)
class TramNetwork:
    """The graph plus every index the read-only API needs, built once per process."""

    metadata: GraphMetadata
    stops: Mapping[int, TramStop]
    edges: tuple[TramEdge, ...]
    routes: Mapping[str, TramRoute]
    components: tuple[tuple[int, ...], ...]
    component_of: Mapping[int, int]
    outgoing: Mapping[int, tuple[TramEdge, ...]]
    incoming: Mapping[int, tuple[TramEdge, ...]]
    track_geometry: Mapping[tuple[int, int], tuple[Coordinate, ...]]

    @classmethod
    def build(
        cls,
        metadata: GraphMetadata,
        stops: Sequence[TramStop],
        edges: Sequence[TramEdge],
        track_geometry: Mapping[tuple[int, int], tuple[Coordinate, ...]],
    ) -> "TramNetwork":
        for stop in stops:
            if type(stop.id) is not int or not isinstance(stop.name, str):
                raise TramGraphDataError("stop id must be an integer and name must be text")
            _validate_coordinate((stop.longitude, stop.latitude))
            _validate_routes(stop.routes)
        stops_by_id = {stop.id: stop for stop in stops}
        if len(stops_by_id) != len(stops):
            raise TramGraphDataError("duplicate stop id")
        edge_keys: set[tuple[int, int]] = set()
        for edge in edges:
            key = (edge.source, edge.target)
            if type(edge.source) is not int or type(edge.target) is not int:
                raise TramGraphDataError("edge endpoints must be integers")
            if edge.source not in stops_by_id or edge.target not in stops_by_id:
                raise TramGraphDataError("edge references an unknown stop")
            if key in edge_keys:
                raise TramGraphDataError("duplicate directed edge")
            edge_keys.add(key)
            if not _finite_number(edge.length_m) or edge.length_m < 0:
                raise TramGraphDataError("edge length must be finite and nonnegative")
            _validate_routes(edge.routes)
        if not _finite_number(sum(edge.length_m for edge in edges)):
            raise TramGraphDataError("total edge length must be finite")
        for key, coordinates in track_geometry.items():
            if not isinstance(key, tuple) or len(key) != 2 or any(type(k) is not int for k in key):
                raise TramGraphDataError("geometry endpoints must be a pair of integers")
            if key not in edge_keys:
                raise TramGraphDataError("geometry references an unknown directed edge")
            if not isinstance(coordinates, (tuple, list)) or len(coordinates) < 2:
                raise TramGraphDataError("track geometry requires at least two positions")
            for coordinate in coordinates:
                _validate_coordinate(coordinate)
        outgoing: dict[int, list[TramEdge]] = {stop.id: [] for stop in stops}
        incoming: dict[int, list[TramEdge]] = {stop.id: [] for stop in stops}
        for edge in edges:
            outgoing.setdefault(edge.source, []).append(edge)
            incoming.setdefault(edge.target, []).append(edge)

        components = _connected_components(stops_by_id, outgoing, incoming)
        component_of = {
            stop_id: index for index, component in enumerate(components) for stop_id in component
        }
        return cls(
            metadata=metadata,
            stops=stops_by_id,
            edges=tuple(edges),
            routes=_index_routes(stops_by_id, edges, component_of),
            components=components,
            component_of=component_of,
            outgoing={key: tuple(value) for key, value in outgoing.items()},
            incoming={key: tuple(value) for key, value in incoming.items()},
            track_geometry=dict(track_geometry),
        )

    def stats(self) -> NetworkStats:
        lengths = [edge.length_m for edge in self.edges]
        degrees: Counter[int] = Counter(
            len(
                {edge.target for edge in self.outgoing.get(stop_id, ())}
                | {edge.source for edge in self.incoming.get(stop_id, ())}
            )
            for stop_id in self.stops
        )
        return NetworkStats(
            stops=len(self.stops),
            edges=len(self.edges),
            routes=len(self.routes),
            route_relations=self.metadata.route_relations,
            total_length_km=round(sum(lengths) / 1000, 2),
            generated_at=self.metadata.generated_at,
            osm_data_timestamp=self.metadata.osm_data_timestamp,
            components=tuple(
                NetworkComponent(
                    size=len(component),
                    routes=sorted_refs(
                        ref for stop_id in component for ref in self.stops[stop_id].routes
                    ),
                )
                for component in self.components
            ),
            degree_histogram={str(degree): degrees[degree] for degree in sorted(degrees)},
            segment_length_m=SegmentLengthSummary(
                min=round(min(lengths), 1) if lengths else 0.0,
                median=round(statistics.median(lengths), 1) if lengths else 0.0,
                mean=round(statistics.mean(lengths), 1) if lengths else 0.0,
                max=round(max(lengths), 1) if lengths else 0.0,
            ),
        )

    def search_stops(self, query: str | None, limit: int) -> tuple[TramStop, ...]:
        if not query:
            return tuple(self.stops.values())[:limit]
        needle = query.casefold()
        found: list[TramStop] = []
        for stop in self.stops.values():
            if needle in stop.name.casefold():
                found.append(stop)
                if len(found) >= limit:
                    break
        return tuple(found)

    def stop_detail(self, stop_id: int) -> StopDetail | None:
        stop = self.stops.get(stop_id)
        if stop is None:
            return None
        neighbours = [
            StopNeighbour(
                id=edge.target,
                name=self.stops[edge.target].name,
                length_m=edge.length_m,
                routes=edge.routes,
                direction=NeighbourDirection.OUT,
            )
            for edge in self.outgoing.get(stop_id, ())
        ]
        neighbours += [
            StopNeighbour(
                id=edge.source,
                name=self.stops[edge.source].name,
                length_m=edge.length_m,
                routes=edge.routes,
                direction=NeighbourDirection.IN,
            )
            for edge in self.incoming.get(stop_id, ())
        ]
        return StopDetail(stop=stop, neighbours=tuple(neighbours))

    def route_detail(self, ref: str) -> RouteDetail | None:
        route = self.routes.get(ref)
        if route is None:
            return None
        return RouteDetail(
            ref=route.ref,
            stops=tuple(self.stops[stop_id] for stop_id in route.stop_ids),
            edges=route.edges,
            length_m=route.length_m,
            component=route.component,
        )

    def geometry_quality(self, edge: TramEdge) -> GeometryQuality:
        if (edge.source, edge.target) not in self.track_geometry:
            return GeometryQuality.INFERRED
        return GeometryQuality.SYNTHETIC if self.metadata.synthetic else GeometryQuality.PROVIDED

    def track_coordinates(self, edge: TramEdge) -> tuple[Coordinate, ...]:
        """Coordinates only; consumers must inspect geometry_quality before rendering rails."""
        coordinates = self.track_geometry.get((edge.source, edge.target))
        if coordinates:
            return coordinates
        source, target = self.stops[edge.source], self.stops[edge.target]
        return ((source.longitude, source.latitude), (target.longitude, target.latitude))

    def geometry(self, ref: str | None) -> NetworkGeometry:
        stops = tuple(self.stops.values())
        edges = self.edges
        if ref is not None:
            stops = tuple(stop for stop in stops if ref in stop.routes)
            edges = tuple(edge for edge in edges if ref in edge.routes)
        return NetworkGeometry(
            metadata=self.metadata,
            filtered_to_route=ref,
            stops=stops,
            segments=tuple(
                TrackSegment(
                    source=edge.source,
                    target=edge.target,
                    length_m=edge.length_m,
                    routes=edge.routes,
                    coordinates=self.track_coordinates(edge),
                    geometry_quality=self.geometry_quality(edge),
                )
                for edge in edges
            ),
        )

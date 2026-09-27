from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.tram_graph import (
    GeometryQuality,
    NeighbourDirection,
    NetworkGeometry,
    RouteDetail,
    StopDetail,
    TramRoute,
)
from app.domain.tram_pathfinding import PathAbsence


class TramStopResponse(BaseModel):
    """`latitude`/`longitude`, matching the forecast slice's StopResponse.

    The committed graph files call these `lat`/`lon`; that spelling stops at the
    repository, so one API does not expose two names for one concept. Original
    docstring follows.

    These are the field names in
    `docs/tram-graph.md`, the committed GeoJSON and every map library."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="OSM stop_position node id")
    name: str
    latitude: float
    longitude: float
    routes: list[str]


class TramEdgeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source: int
    target: int
    length_m: float = Field(ge=0, description="Metres along the track, not as the crow flies")
    routes: list[str]


class StopNeighbourResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    length_m: float = Field(ge=0)
    routes: list[str]
    direction: NeighbourDirection


class StopDetailResponse(TramStopResponse):
    neighbours: list[StopNeighbourResponse]

    @classmethod
    def from_domain(cls, detail: StopDetail) -> "StopDetailResponse":
        return cls(
            id=detail.stop.id,
            name=detail.stop.name,
            latitude=detail.stop.latitude,
            longitude=detail.stop.longitude,
            routes=list(detail.stop.routes),
            neighbours=[
                StopNeighbourResponse.model_validate(neighbour) for neighbour in detail.neighbours
            ],
        )


class TramRouteResponse(BaseModel):
    ref: str = Field(description='Route ref as OSM carries it; not always numeric ("А", "т1")')
    stop_count: int = Field(ge=0)
    length_m: float = Field(ge=0)
    component: int = Field(ge=0, description="0 is the main network, 1 the northern one")

    @classmethod
    def from_domain(cls, route: TramRoute) -> "TramRouteResponse":
        return cls(
            ref=route.ref,
            stop_count=len(route.stop_ids),
            length_m=route.length_m,
            component=route.component,
        )


class RouteDetailResponse(BaseModel):
    ref: str
    stops: list[TramStopResponse]
    edges: list[TramEdgeResponse]
    length_m: float = Field(ge=0)
    component: int = Field(ge=0)

    @classmethod
    def from_domain(cls, route: RouteDetail) -> "RouteDetailResponse":
        return cls(
            ref=route.ref,
            stops=[TramStopResponse.model_validate(stop) for stop in route.stops],
            edges=[TramEdgeResponse.model_validate(edge) for edge in route.edges],
            length_m=route.length_m,
            component=route.component,
        )


class NetworkComponentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    size: int = Field(ge=0)
    routes: list[str]


class SegmentLengthSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    min: float
    median: float
    mean: float
    max: float


class NetworkStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stops: int
    edges: int = Field(description="Directed edges; a pair served both ways counts twice")
    routes: int = Field(description="Distinct route refs")
    route_relations: int | None = Field(
        default=None,
        description=(
            "PTv2 route relations in the OSM extract, one per direction. A larger "
            "number than `routes`, and not interchangeable with it."
        ),
    )
    total_length_km: float
    generated_at: str | None = None
    osm_data_timestamp: str | None = None
    components: list[NetworkComponentResponse] = Field(
        description="Undirected components, largest first; the network really is in two pieces"
    )
    degree_histogram: dict[str, int]
    segment_length_m: SegmentLengthSummaryResponse


class TramPathResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    found: bool
    reason_code: PathAbsence | None = Field(
        default=None,
        description=(
            "Why there is no such ride, as a value a client can branch on. Present with "
            "`found=false`, which is a 200. Prefer this over `reason` for anything the "
            "user sees: `reason` is English prose."
        ),
    )
    reason: str | None = Field(
        default=None,
        description="The same cause in English prose, for a human reading the API directly.",
    )
    stops: list[TramStopResponse]
    total_length_m: float = Field(ge=0)
    geometry: list[tuple[float, float]] = Field(
        description="[lon, lat] pairs; inferred paths contain straight endpoint connectors"
    )
    geometry_quality: GeometryQuality | None = None
    missing_geometry_edges: int = Field(default=0, ge=0)
    routes: list[str]


class GraphMetadataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    synthetic: bool = False
    source: str | None = None
    license: str | None = None
    area: str | None = None
    osm_data_timestamp: str | None = None
    generated_at: str | None = None
    route_relations: int | None = None


class GeoJsonMetadataResponse(GraphMetadataResponse):
    missing_geometry_edges: int = Field(default=0, ge=0)
    filtered_to_route: str | None = None


class PointGeometry(BaseModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float]


class LineStringGeometry(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]]


class StopFeatureProperties(BaseModel):
    id: int
    name: str
    routes: list[str]


class SegmentFeatureProperties(BaseModel):
    geometry_quality: GeometryQuality
    source: int
    target: int
    length_m: float
    routes: list[str]


class StopFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PointGeometry
    properties: StopFeatureProperties


class SegmentFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: LineStringGeometry
    properties: SegmentFeatureProperties


class TramGraphGeoJson(BaseModel):
    """A GeoJSON FeatureCollection: stops as Points, track segments as LineStrings."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    metadata: GeoJsonMetadataResponse
    features: list[StopFeature | SegmentFeature]

    @classmethod
    def from_domain(cls, geometry: NetworkGeometry) -> "TramGraphGeoJson":
        features: list[StopFeature | SegmentFeature] = [
            StopFeature(
                geometry=PointGeometry(coordinates=(stop.longitude, stop.latitude)),
                properties=StopFeatureProperties(
                    id=stop.id, name=stop.name, routes=list(stop.routes)
                ),
            )
            for stop in geometry.stops
        ]
        features += [
            SegmentFeature(
                geometry=LineStringGeometry(coordinates=list(segment.coordinates)),
                properties=SegmentFeatureProperties(
                    source=segment.source,
                    target=segment.target,
                    length_m=segment.length_m,
                    routes=list(segment.routes),
                    geometry_quality=segment.geometry_quality,
                ),
            )
            for segment in geometry.segments
        ]
        return cls(
            metadata=GeoJsonMetadataResponse(
                **GraphMetadataResponse.model_validate(geometry.metadata).model_dump(),
                filtered_to_route=geometry.filtered_to_route,
                missing_geometry_edges=sum(
                    segment.geometry_quality == GeometryQuality.INFERRED
                    for segment in geometry.segments
                ),
            ),
            features=features,
        )

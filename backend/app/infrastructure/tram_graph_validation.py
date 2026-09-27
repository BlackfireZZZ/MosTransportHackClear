"""Strict parsing shared by the file repository and offline graph publication."""

import math
from dataclasses import replace
from typing import Any

from app.domain.tram_graph import (
    Coordinate,
    GraphMetadata,
    TramEdge,
    TramGraphDataError,
    TramNetwork,
    TramStop,
)


def _object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TramGraphDataError(f"{location} must be an object")
    return value


def _array(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TramGraphDataError(f"{location} must be an array")
    return value


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TramGraphDataError(f"{location} must be an integer")
    return value


def _text(value: object, location: str) -> str:
    if not isinstance(value, str):
        raise TramGraphDataError(f"{location} must be a string")
    return value


def _number(value: object, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TramGraphDataError(f"{location} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise TramGraphDataError(f"{location} must be finite") from error
    if not math.isfinite(result):
        raise TramGraphDataError(f"{location} must be finite")
    return result


def _routes(value: object, location: str) -> tuple[str, ...]:
    return tuple(_text(ref, location) for ref in _array(value, location))


def _coordinate(value: object, location: str) -> Coordinate:
    position = _array(value, location)
    if len(position) != 2:
        raise TramGraphDataError(f"{location} must contain longitude and latitude")
    lon, lat = (_number(item, location) for item in position)
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise TramGraphDataError(f"{location} is outside WGS84 ranges")
    return lon, lat


def _metadata(document: dict[str, Any]) -> GraphMetadata:
    raw = _object(document.get("metadata", {}), "metadata")
    fields: dict[str, Any] = {}
    if "synthetic" in raw:
        if type(raw["synthetic"]) is not bool:
            raise TramGraphDataError("metadata.synthetic must be a boolean")
        fields["synthetic"] = raw["synthetic"]
    for field in ("source", "license", "area", "osm_data_timestamp", "generated_at"):
        if field in raw:
            fields[field] = _text(raw[field], f"metadata.{field}")
    if "routes_used" in raw:
        count = _integer(raw["routes_used"], "metadata.routes_used")
        if count < 0:
            raise TramGraphDataError("metadata.routes_used must be nonnegative")
        fields["route_relations"] = count
    return GraphMetadata(**fields)


def parse_network(document: object, geojson: object) -> TramNetwork:
    """Reject malformed pairs without coercing identifiers or numeric strings.

    Legacy extracts may omit metadata and per-edge geometry; missing geometry
    is explicitly marked inferred. Present features must be valid.
    """
    graph = _object(document, "graph")
    if "directed" in graph and graph["directed"] is not True:
        raise TramGraphDataError("graph.directed must be true")
    stops: list[TramStop] = []
    for index, value in enumerate(_array(graph.get("nodes"), "graph.nodes")):
        node = _object(value, f"nodes[{index}]")
        stops.append(
            TramStop(
                id=_integer(node.get("id"), "node.id"),
                name=_text(node.get("name"), "node.name"),
                latitude=_number(node.get("lat"), "node.lat"),
                longitude=_number(node.get("lon"), "node.lon"),
                routes=_routes(node.get("routes"), "node.routes"),
            )
        )
    edges: list[TramEdge] = []
    for index, value in enumerate(_array(graph.get("links"), "graph.links")):
        link = _object(value, f"links[{index}]")
        edges.append(
            TramEdge(
                source=_integer(link.get("source"), "link.source"),
                target=_integer(link.get("target"), "link.target"),
                length_m=_number(link.get("length_m"), "link.length_m"),
                routes=_routes(link.get("routes"), "link.routes"),
            )
        )
    collection = _object(geojson, "geojson")
    if collection.get("type") != "FeatureCollection":
        raise TramGraphDataError("geojson.type must be FeatureCollection")
    geometry_metadata = _metadata(collection)
    geometry: dict[tuple[int, int], tuple[Coordinate, ...]] = {}
    stop_ids = {stop.id for stop in stops}
    point_ids: set[int] = set()
    for index, value in enumerate(_array(collection.get("features"), "geojson.features")):
        feature = _object(value, f"features[{index}]")
        if feature.get("type") != "Feature":
            raise TramGraphDataError("geojson feature.type must be Feature")
        shape = _object(feature.get("geometry"), "feature.geometry")
        properties = _object(feature.get("properties"), "feature.properties")
        if shape.get("type") == "Point":
            stop_id = _integer(properties.get("id"), "point.id")
            if stop_id not in stop_ids or stop_id in point_ids:
                raise TramGraphDataError("point references an unknown or duplicate stop")
            point_ids.add(stop_id)
            _coordinate(shape.get("coordinates"), "point.coordinates")
            _text(properties.get("name"), "point.name")
            _routes(properties.get("routes"), "point.routes")
        elif shape.get("type") == "LineString":
            key = (
                _integer(properties.get("source"), "geometry.source"),
                _integer(properties.get("target"), "geometry.target"),
            )
            if key in geometry:
                raise TramGraphDataError("duplicate directed edge geometry")
            length = _number(properties.get("length_m"), "geometry.length_m")
            if length < 0:
                raise TramGraphDataError("geometry.length_m must be nonnegative")
            _routes(properties.get("routes"), "geometry.routes")
            geometry[key] = tuple(
                _coordinate(point, "geometry.coordinates")
                for point in _array(shape.get("coordinates"), "geometry.coordinates")
            )
        else:
            raise TramGraphDataError("graph features must be Point or LineString")
    metadata = _metadata(graph)
    metadata = replace(metadata, synthetic=metadata.synthetic or geometry_metadata.synthetic)
    return TramNetwork.build(metadata, stops, edges, geometry)

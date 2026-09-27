import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.domain.tram_graph import GraphMetadata, TramEdge, TramGraphDataError, TramNetwork, TramStop
from app.infrastructure.repositories.tram_graph import FileTramGraphRepository
from app.infrastructure.tram_graph_validation import parse_network


def documents() -> tuple[dict[str, Any], dict[str, Any]]:
    graph = {
        "directed": True,
        "nodes": [
            {"id": 1, "name": "A", "lat": 55.0, "lon": 37.0, "routes": ["1"]},
            {"id": 2, "name": "B", "lat": 56.0, "lon": 38.0, "routes": ["1"]},
        ],
        "links": [{"source": 1, "target": 2, "length_m": 0, "routes": ["1"]}],
    }
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": dict(graph["links"][0]),
                "geometry": {"type": "LineString", "coordinates": [[37, 55], [38, 56]]},
            }
        ],
    }
    return graph, geojson


@pytest.mark.parametrize("root", [None, [], "graph", 1, True])
@pytest.mark.parametrize("side", [0, 1])
def test_malformed_roots(root: object, side: int) -> None:
    pair: list[object] = list(documents())
    pair[side] = root
    with pytest.raises(TramGraphDataError):
        parse_network(*pair)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("nodes",), None),
        (("nodes",), {}),
        (("nodes", 0), None),
        (("links",), None),
        (("links",), {}),
        (("links", 0), []),
        (("directed",), False),
        (("directed",), 1),
        (("metadata",), []),
        (("metadata",), None),
        (("metadata",), {"routes_used": True}),
        (("metadata",), {"routes_used": -1}),
        (("metadata",), {"area": 5}),
        (("nodes", 0, "id"), True),
        (("nodes", 0, "id"), "1"),
        (("nodes", 0, "id"), 1.1),
        (("nodes", 0, "name"), None),
        (("nodes", 0, "routes"), "1"),
        (("nodes", 0, "routes"), [1]),
        (("nodes", 0, "lat"), 91),
        (("nodes", 0, "lon"), -181),
        (("nodes", 0, "lat"), "55"),
        (("nodes", 0, "lat"), True),
        (("nodes", 0, "lat"), float("nan")),
        (("nodes", 0, "lon"), float("inf")),
        (("nodes", 0, "lat"), 10**400),
        (("links", 0, "source"), 999),
        (("links", 0, "target"), True),
        (("links", 0, "target"), "2"),
        (("links", 0, "routes"), [False]),
        (("links", 0, "length_m"), -1),
        (("links", 0, "length_m"), True),
        (("links", 0, "length_m"), "1"),
        (("links", 0, "length_m"), float("nan")),
        (("links", 0, "length_m"), float("inf")),
    ],
)
def test_corrupt_graph_matrix(path: tuple[str | int, ...], value: object) -> None:
    graph, geojson = documents()
    current: Any = graph
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(TramGraphDataError):
        parse_network(graph, geojson)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("type",), "Feature"),
        (("features",), None),
        (("features",), {}),
        (("features", 0), None),
        (("features", 0, "type"), "NotFeature"),
        (("features", 0, "properties"), []),
        (("features", 0, "geometry"), None),
        (("features", 0, "geometry", "type"), "Polygon"),
        (("features", 0, "geometry", "coordinates"), []),
        (("features", 0, "geometry", "coordinates"), [[37, 55]]),
        (("features", 0, "geometry", "coordinates"), [[37, 55], [38]]),
        (("features", 0, "geometry", "coordinates"), [[37, 55], [38, 56, 0]]),
        (("features", 0, "geometry", "coordinates"), [[37, 55], [True, 56]]),
        (("features", 0, "geometry", "coordinates"), [[37, 55], [181, 56]]),
        (("features", 0, "geometry", "coordinates"), [[37, 55], [38, float("nan")]]),
        (("features", 0, "properties", "source"), 999),
        (("features", 0, "properties", "source"), True),
        (("features", 0, "properties", "length_m"), -1),
        (("features", 0, "properties", "routes"), "1"),
    ],
)
def test_corrupt_geojson_matrix(path: tuple[str | int, ...], value: object) -> None:
    graph, geojson = documents()
    current: Any = geojson
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(TramGraphDataError):
        parse_network(graph, geojson)


@pytest.mark.parametrize("collection", ["nodes", "links", "features"])
def test_duplicate_identifiers_are_not_silently_collapsed(collection: str) -> None:
    graph, geojson = documents()
    values = (geojson if collection == "features" else graph)[collection]
    values.append(values[0])
    with pytest.raises(TramGraphDataError, match="duplicate"):
        parse_network(graph, geojson)


def test_valid_zero_length_direction_and_missing_edge_geometry() -> None:
    graph, geojson = documents()
    graph["links"].append({"source": 2, "target": 1, "length_m": 5, "routes": ["1"]})
    network = parse_network(graph, geojson)
    assert [edge.length_m for edge in network.edges] == [0, 5]
    assert network.track_coordinates(network.edges[1]) == ((38, 56), (37, 55))


@pytest.mark.parametrize("failure", ["unknown", "duplicate", "coordinates", "properties"])
def test_corrupt_point_features(failure: str) -> None:
    graph, geojson = documents()
    point = {
        "type": "Feature",
        "properties": {"id": 1, "name": "A", "routes": ["1"]},
        "geometry": {"type": "Point", "coordinates": [37, 55]},
    }
    geojson["features"].append(point)
    if failure == "unknown":
        point["properties"]["id"] = 999
    elif failure == "duplicate":
        geojson["features"].append(point)
    elif failure == "coordinates":
        point["geometry"]["coordinates"] = [37, 91]
    else:
        point["properties"]["name"] = None
    with pytest.raises(TramGraphDataError):
        parse_network(graph, geojson)


def test_nondecimal_digit_route_ref_is_valid_text() -> None:
    graph, geojson = documents()
    graph["nodes"][0]["routes"] = ["²"]
    assert "²" in parse_network(graph, geojson).routes


def test_extremely_long_numeric_route_ref_is_a_graph_data_error() -> None:
    graph, geojson = documents()
    graph["nodes"][0]["routes"] = ["1" * 5000]
    with pytest.raises(TramGraphDataError, match="numeric route reference"):
        parse_network(graph, geojson)


@pytest.mark.parametrize(
    "geometry",
    [
        {(True, 2): ((37, 55), (38, 56))},
        {(1, 2): None},
        {(1, 2): (None, None)},
    ],
)
def test_domain_rejects_malformed_geometry(geometry: Any) -> None:
    stops = [TramStop(1, "A", 55, 37, ("1",)), TramStop(2, "B", 56, 38, ("1",))]
    with pytest.raises(TramGraphDataError):
        TramNetwork.build(GraphMetadata(), stops, [TramEdge(1, 2, 5, ("1",))], geometry)


@pytest.mark.parametrize("content", [b"\xff", b"{", b"null", b"[]"])
def test_repository_translates_invalid_files(tmp_path: Path, content: bytes) -> None:
    graph_path, geojson_path = tmp_path / "graph.json", tmp_path / "graph.geojson"
    graph_path.write_bytes(content)
    geojson_path.write_text(json.dumps(documents()[1]))
    with pytest.raises(TramGraphDataError):
        FileTramGraphRepository(graph_path, geojson_path).load()


@pytest.mark.asyncio
async def test_failed_repository_load_can_be_retried_after_repair(tmp_path: Path) -> None:
    graph, geojson = documents()
    graph_path, geojson_path = tmp_path / "graph.json", tmp_path / "graph.geojson"
    graph_path.write_text("[]")
    geojson_path.write_text(json.dumps(geojson))
    repository = FileTramGraphRepository(graph_path, geojson_path)
    with pytest.raises(TramGraphDataError):
        await repository.get_network()
    graph_path.write_text(json.dumps(graph))
    network = await repository.get_network()
    assert len(network.stops) == 2
    assert await repository.get_network() is network


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_stop",
        "duplicate_edge",
        "unknown_endpoint",
        "negative",
        "nonfinite",
        "coordinate",
        "boolean_id",
    ],
)
def test_domain_build_checks_invariants_independent_of_json(case: str) -> None:
    stops = [TramStop(1, "A", 55, 37, ("1",)), TramStop(2, "B", 56, 38, ("1",))]
    edges = [TramEdge(1, 2, 5, ("1",))]
    if case == "duplicate_stop":
        stops.append(stops[0])
    elif case == "duplicate_edge":
        edges.append(edges[0])
    elif case == "unknown_endpoint":
        edges[0] = replace(edges[0], target=3)
    elif case == "negative":
        edges[0] = replace(edges[0], length_m=-1)
    elif case == "nonfinite":
        edges[0] = replace(edges[0], length_m=float("nan"))
    elif case == "coordinate":
        stops[0] = replace(stops[0], latitude=91)
    else:
        stops[0] = replace(stops[0], id=True)
    with pytest.raises(TramGraphDataError):
        TramNetwork.build(GraphMetadata(), stops, edges, {})


@pytest.mark.parametrize("synthetic, expected", [(False, "provided"), (True, "synthetic")])
def test_provided_and_synthetic_geometry_quality(synthetic: bool, expected: str) -> None:
    from app.domain.tram_pathfinding import find_path
    from app.schemas.tram_graph import TramGraphGeoJson

    graph, geojson = documents()
    graph["metadata"] = {"synthetic": synthetic}
    network = parse_network(graph, geojson)
    assert network.geometry(None).segments[0].geometry_quality == expected
    assert find_path(network, 1, 2).geometry_quality == expected
    assert find_path(network, 1, 2).missing_geometry_edges == 0
    assert TramGraphGeoJson.from_domain(network.geometry(None)).metadata.synthetic == synthetic


def test_partial_geometry_counts_only_missing_edges() -> None:
    from app.domain.tram_pathfinding import find_path
    from app.schemas.tram_graph import TramGraphGeoJson

    graph, geojson = documents()
    graph["links"].append({"source": 2, "target": 1, "length_m": 5, "routes": ["1"]})
    network = parse_network(graph, geojson)
    assert TramGraphGeoJson.from_domain(network.geometry(None)).metadata.missing_geometry_edges == 1
    assert find_path(network, 1, 2).geometry_quality == "provided"
    assert find_path(network, 2, 1).geometry_quality == "inferred"


@pytest.mark.parametrize("value", ["true", 1, None])
def test_synthetic_metadata_requires_boolean(value: object) -> None:
    graph, geojson = documents()
    graph["metadata"] = {"synthetic": value}
    with pytest.raises(TramGraphDataError, match="synthetic must be a boolean"):
        parse_network(graph, geojson)


@pytest.mark.parametrize("graph_metadata", [{}, {"synthetic": False}, {"synthetic": True}])
def test_geometry_synthetic_marker_cannot_be_erased_by_graph_metadata(
    graph_metadata: dict[str, object],
) -> None:
    from app.domain.tram_pathfinding import find_path
    from app.schemas.tram_graph import TramGraphGeoJson

    graph, geojson = documents()
    graph["metadata"] = graph_metadata
    geojson["metadata"] = {"synthetic": True}
    network = parse_network(graph, geojson)
    assert network.metadata.synthetic is True
    assert network.geometry(None).segments[0].geometry_quality == "synthetic"
    assert find_path(network, 1, 2).geometry_quality == "synthetic"
    assert TramGraphGeoJson.from_domain(network.geometry(None)).metadata.synthetic is True

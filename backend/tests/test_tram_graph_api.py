"""Tram graph endpoints, exercised against the committed graph in `data/`.

The numbers below are the facts recorded in `docs/tram-graph.md` for the 2026-09-18
extract. They are asserted rather than recomputed: the point is to catch a loader or
an index that quietly changes the graph, so a test that derives its expectation from
the same code under test would prove nothing.
"""

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_tram_graph_repository
from app.core.config import settings
from app.domain.tram_graph import TramGraphDataError, TramNetwork
from app.infrastructure.repositories.tram_graph import FileTramGraphRepository
from app.main import app

BASE = "/api/v1/tram-graph"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def network() -> TramNetwork:
    return get_tram_graph_repository().load()


def test_stats_match_the_committed_extract(client: TestClient) -> None:
    body = client.get(f"{BASE}/stats").json()

    assert body["stops"] == 856
    assert body["edges"] == 919
    assert body["total_length_km"] == pytest.approx(377.0, abs=0.5)
    assert [component["size"] for component in body["components"]] == [694, 162]
    assert "6" in body["components"][1]["routes"]
    assert sum(body["degree_histogram"].values()) == 856
    assert body["degree_histogram"] == {"1": 40, "2": 665, "3": 136, "4": 15}
    assert body["generated_at"] and body["osm_data_timestamp"]
    segment = body["segment_length_m"]
    assert segment["min"] < segment["median"] <= segment["mean"] < segment["max"]


def test_stats_separate_route_refs_from_pt_v2_relations(client: TestClient) -> None:
    body = client.get(f"{BASE}/stats").json()

    # 73 PTv2 relations, one per direction, collapse to 38 distinct refs. Reporting
    # either number as the other is the mistake this pins.
    assert body["routes"] == 38
    assert body["route_relations"] == 73


def test_routes_sort_naturally_without_parsing_refs_as_numbers(client: TestClient) -> None:
    body = client.get(f"{BASE}/routes").json()
    refs = [route["ref"] for route in body]

    assert refs[:3] == ["1", "2", "4"]
    assert refs.index("10") > refs.index("2")
    assert refs.index("А") > refs.index("50")
    assert all(route["stop_count"] > 0 and route["length_m"] > 0 for route in body)
    assert {route["component"] for route in body} == {0, 1}


@pytest.mark.parametrize("ref", ["1", "А", "т1", "39а"])
def test_route_detail(client: TestClient, ref: str) -> None:
    response = client.get(f"{BASE}/routes/{quote(ref)}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ref"] == ref
    assert body["stops"] and body["edges"]
    assert all(ref in stop["routes"] for stop in body["stops"])
    assert all(ref in edge["routes"] for edge in body["edges"])
    assert body["length_m"] == pytest.approx(
        sum(edge["length_m"] for edge in body["edges"]), abs=0.5
    )


def test_unknown_route_is_404_and_does_not_conflate_the_two_counts(client: TestClient) -> None:
    response = client.get(f"{BASE}/routes/999")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "999" in detail
    # The message must not offer the relation count as if it were a list of refs.
    assert "route relations" in detail and "per direction" in detail


def test_stop_search_is_case_folded(client: TestClient) -> None:
    body = client.get(f"{BASE}/stops", params={"q": "курский"}).json()

    assert body
    assert all("курский" in stop["name"].casefold() for stop in body)
    assert set(body[0]) == {"id", "name", "latitude", "longitude", "routes"}
    assert len(client.get(f"{BASE}/stops").json()) == 50
    assert len(client.get(f"{BASE}/stops", params={"limit": 3}).json()) == 3


def test_stop_detail_lists_both_directions(client: TestClient, network: TramNetwork) -> None:
    stop_id = 472373305  # Курский вокзал
    body = client.get(f"{BASE}/stops/{stop_id}").json()

    assert body["name"] == "Курский вокзал"
    assert body["neighbours"]
    assert {neighbour["direction"] for neighbour in body["neighbours"]} <= {"in", "out"}
    for neighbour in body["neighbours"]:
        assert neighbour["id"] in network.stops
        assert neighbour["length_m"] > 0


def test_unknown_stop_is_404(client: TestClient) -> None:
    response = client.get(f"{BASE}/stops/1")

    assert response.status_code == 404
    assert "1" in response.json()["detail"]


def test_edges(client: TestClient) -> None:
    assert len(client.get(f"{BASE}/edges").json()) == 919

    filtered = client.get(f"{BASE}/edges", params={"route": "т1"}).json()
    assert filtered and all("т1" in edge["routes"] for edge in filtered)
    assert client.get(f"{BASE}/edges", params={"route": "nope"}).status_code == 404


def test_geojson(client: TestClient) -> None:
    body = client.get(f"{BASE}/geojson").json()

    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 856 + 919
    assert body["metadata"]["filtered_to_route"] is None

    one = client.get(f"{BASE}/geojson", params={"route": "А"}).json()
    assert 0 < len(one["features"]) < len(body["features"])
    assert one["metadata"]["filtered_to_route"] == "А"
    assert all("А" in feature["properties"]["routes"] for feature in one["features"])
    lines = [f for f in one["features"] if f["geometry"]["type"] == "LineString"]
    assert lines and len(lines[0]["geometry"]["coordinates"]) >= 2
    assert client.get(f"{BASE}/geojson", params={"route": "nope"}).status_code == 404


def test_path_follows_the_track_geometry(client: TestClient) -> None:
    body = client.get(f"{BASE}/path", params={"from": 1377188043, "to": 472373305}).json()

    assert body["found"] is True
    assert body["reason"] is None
    assert body["stops"][0]["id"] == 1377188043
    assert body["stops"][-1]["name"] == "Курский вокзал"
    assert body["total_length_m"] == pytest.approx(32014, abs=50)
    # More coordinates than stops, because the polyline follows the rails.
    assert len(body["geometry"]) > len(body["stops"])
    assert all(len(point) == 2 for point in body["geometry"])
    assert body["routes"]


def test_path_across_components_is_an_answer_not_an_error(
    client: TestClient, network: TramNetwork
) -> None:
    source = network.components[0][0]
    target = network.components[1][0]

    response = client.get(f"{BASE}/path", params={"from": source, "to": target})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["stops"] == [] and body["geometry"] == []
    assert "different parts" in body["reason"]


def test_path_from_unknown_stop_is_an_answer_not_an_error(client: TestClient) -> None:
    response = client.get(f"{BASE}/path", params={"from": 1, "to": 472373305})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert "unknown stop id 1" in body["reason"]


def test_path_to_self_is_a_zero_length_ride(client: TestClient) -> None:
    body = client.get(f"{BASE}/path", params={"from": 472373305, "to": 472373305}).json()

    assert body["found"] is True
    assert body["total_length_m"] == 0
    assert [stop["id"] for stop in body["stops"]] == [472373305]
    assert len(body["geometry"]) == 1


def test_missing_geometry_file_is_an_error_not_straight_lines(tmp_path: Path) -> None:
    """Both graph files come from one script run; one without the other is broken.

    The old behaviour returned 200 with straight-line geometry, so a deployment
    missing the GeoJSON looked healthy while drawing trams through buildings.
    """
    graph_json = settings.tram_graph_json
    repository = FileTramGraphRepository(graph_json, tmp_path / "absent.geojson")

    with pytest.raises(TramGraphDataError) as error:
        repository.load()

    assert "absent.geojson" in str(error.value)


def test_http_geometry_quality_matches_complete_committed_extract(client: TestClient) -> None:
    body = client.get(f"{BASE}/geojson").json()
    assert body["metadata"]["missing_geometry_edges"] == 0
    assert body["metadata"]["synthetic"] is False
    lines = [item for item in body["features"] if item["geometry"]["type"] == "LineString"]
    assert lines and all(item["properties"]["geometry_quality"] == "provided" for item in lines)
    path = client.get(f"{BASE}/path", params={"from": 1377188043, "to": 472373305}).json()
    assert path["found"]
    assert path["geometry_quality"] == "provided"
    assert path["missing_geometry_edges"] == 0

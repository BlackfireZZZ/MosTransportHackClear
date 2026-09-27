"""Domain rules of the tram network, on a hand-built graph.

These are the properties that are easy to get wrong and hard to see in the real
extract: directed edges with different lengths per direction, refs that are not
numbers, and a second component that has to stay unreachable.
"""

import pytest

from app.domain.tram_graph import (
    GraphMetadata,
    NeighbourDirection,
    TramEdge,
    TramNetwork,
    TramStop,
    route_sort_key,
    sorted_refs,
)
from app.domain.tram_pathfinding import PathAbsence, find_path


def stop(stop_id: int, name: str, *routes: str) -> TramStop:
    return TramStop(id=stop_id, name=name, latitude=55.0, longitude=37.0, routes=routes)


@pytest.fixture
def network() -> TramNetwork:
    # 1 -> 2 -> 3 is the main line; 3 -> 1 closes it the long way round. 9 and 10 are
    # a separate island, like the real northern network around Timiryazevskaya.
    stops = [
        stop(1, "Первая", "1", "т1"),
        stop(2, "Вторая", "1"),
        stop(3, "Третья", "1", "т1"),
        stop(9, "Остров", "А"),
        stop(10, "Остров-2", "А"),
    ]
    edges = [
        TramEdge(source=1, target=2, length_m=100.0, routes=("1",)),
        TramEdge(source=2, target=3, length_m=100.0, routes=("1",)),
        # Same pair, other way round, and longer: the tracks are not parallel.
        TramEdge(source=3, target=2, length_m=250.0, routes=("1",)),
        TramEdge(source=3, target=1, length_m=1000.0, routes=("т1",)),
        TramEdge(source=9, target=10, length_m=50.0, routes=("А",)),
    ]
    return TramNetwork.build(GraphMetadata(route_relations=7), stops, edges, {})


def test_refs_are_sorted_as_strings_with_numbers_first() -> None:
    assert sorted_refs(["10", "2", "т1", "А", "1а", "1"]) == ("1", "2", "10", "1а", "А", "т1")
    assert route_sort_key("39а")[0] == 1  # never int("39а")


def test_components_rank_the_main_network_first(network: TramNetwork) -> None:
    assert [len(component) for component in network.components] == [3, 2]
    assert network.component_of[1] == 0
    assert network.component_of[9] == 1


def test_route_relations_are_not_the_ref_count(network: TramNetwork) -> None:
    stats = network.stats()

    assert stats.routes == 3
    assert stats.route_relations == 7


def test_path_uses_the_length_of_the_direction_travelled(network: TramNetwork) -> None:
    there = find_path(network, 1, 3)
    back = find_path(network, 3, 2)

    assert there.found and there.total_length_m == 200
    assert back.found and back.total_length_m == 250


def test_path_prefers_the_cheaper_chain_over_the_shorter_one(network: TramNetwork) -> None:
    # 3 -> 1 exists directly at 1000 m; there is no cheaper way round.
    result = find_path(network, 3, 1)

    assert [stop.id for stop in result.stops] == [3, 1]
    assert result.total_length_m == 1000
    assert result.routes == ("т1",)


def test_unreachable_component_is_a_successful_absence(network: TramNetwork) -> None:
    result = find_path(network, 1, 9)

    assert result.found is False
    assert result.stops == () and result.geometry == ()
    assert "different parts of the tram network" in (result.reason or "")


def test_neighbours_carry_the_direction(network: TramNetwork) -> None:
    detail = network.stop_detail(2)

    assert detail is not None
    directions = {(neighbour.id, neighbour.direction) for neighbour in detail.neighbours}
    assert (3, NeighbourDirection.OUT) in directions
    assert (1, NeighbourDirection.IN) in directions
    assert (3, NeighbourDirection.IN) in directions


def test_geometry_falls_back_to_the_straight_line(network: TramNetwork) -> None:
    geometry = network.geometry(None)

    assert len(geometry.segments) == 5
    assert all(len(segment.coordinates) == 2 for segment in geometry.segments)


def test_absence_reasons_carry_a_code_a_client_can_branch_on(network: TramNetwork) -> None:
    """`reason` is English prose; a UI in another language needs the code instead."""
    assert find_path(network, 999, 1).reason_code is PathAbsence.UNKNOWN_STOP
    assert find_path(network, 1, 9).reason_code is PathAbsence.DIFFERENT_COMPONENTS
    # 9 -> 10 is the only edge on the island, so the return leg has no track.
    assert find_path(network, 10, 9).reason_code is PathAbsence.WRONG_DIRECTION
    assert find_path(network, 1, 3).reason_code is None


def test_every_absence_sets_both_code_and_prose(network: TramNetwork) -> None:
    for path in (
        find_path(network, 999, 1),
        find_path(network, 1, 9),
        find_path(network, 10, 9),
    ):
        assert path.found is False
        assert path.reason_code is not None
        assert path.reason


def test_missing_geometry_quality_is_explicit_in_path_and_geojson(network: TramNetwork) -> None:
    from app.schemas.tram_graph import TramGraphGeoJson, TramPathResponse

    path = TramPathResponse.model_validate(find_path(network, 1, 3))
    assert path.geometry_quality == "inferred"
    assert path.missing_geometry_edges == 2
    body = TramGraphGeoJson.from_domain(network.geometry(None)).model_dump()
    assert body["metadata"]["missing_geometry_edges"] == 5
    lines = [f for f in body["features"] if f["geometry"]["type"] == "LineString"]
    assert all(f["properties"]["geometry_quality"] == "inferred" for f in lines)
    absent = find_path(network, 1, 9)
    assert absent.geometry_quality is None
    assert absent.missing_geometry_edges == 0

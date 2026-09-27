import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.application.services.forecast_geometry import ForecastGeometryService
from app.domain.forecast_geometry import (
    ForecastEntity,
    GeometryMappingError,
    GeometryMatchStatus,
    RouteGeometryLink,
)
from app.domain.tram_graph import GraphMetadata, TramNetwork, TramStop
from app.infrastructure import graph_artifacts
from app.infrastructure.forecast_geometry import load_crosswalk, load_geometry_service

FIXTURE = Path(__file__).parent / "fixtures" / "forecast_geometry.v1.json"
DATA = Path(__file__).resolve().parents[2] / "data"


def network() -> TramNetwork:
    return TramNetwork.build(
        GraphMetadata(),
        [
            TramStop(101, "Same name", 55.7, 37.5, ("А",)),
            TramStop(202, "Same name", 55.71, 37.51, ("А",)),
            TramStop(303, "Same name", 55.72, 37.52, ("2",)),
        ],
        [],
        {},
    )


def entity(stop: str, direction: str = "out") -> ForecastEntity:
    return ForecastEntity("canonical-route", direction, stop)


def service() -> ForecastGeometryService:
    return ForecastGeometryService(load_crosswalk(FIXTURE), network(), "fixture-graph.v1")


def test_fixture_preserves_directional_duplicates_and_unresolved_counts() -> None:
    entities = [
        entity("same-name"),
        entity("same-name", "in"),
        entity("ambiguous"),
        entity("missing-node"),
        entity("unmapped"),
        entity("wrong-route"),
    ]
    result = service().map_stops(
        entities + [entities[0]],
        entity_version="synthetic-entities.v1",
        graph_version="fixture-graph.v1",
    )
    assert (result.matched_count, result.ambiguous_count, result.unmatched_count) == (2, 1, 3)
    assert len(result.stops) == 6
    assert [row.osm_stop_id for row in result.stops] == [101, 202, None, None, None, None]
    assert [(row.longitude, row.latitude) for row in result.stops[:2]] == [
        (37.5, 55.7),
        (37.51, 55.71),
    ]
    assert [row.reason for row in result.stops[2:]] == [
        "multiple_stop_candidates",
        "unknown_osm_stop",
        "missing_stop_mapping",
        "stop_not_on_mapped_route",
    ]
    assert result.stops[2].candidate_osm_stop_ids == (101, 202)
    assert result.mapping_version == "synthetic-map.v1"
    assert result.entity_version == "synthetic-entities.v1"
    assert result.graph_version == "fixture-graph.v1"
    assert result.route_semantics == "unordered_membership"


def test_names_numeric_ids_and_unknown_directions_never_infer_mapping() -> None:
    result = service().map_stops(
        [entity("Same name"), entity("101"), entity("same-name", "unknown")],
        entity_version="synthetic-entities.v1",
        graph_version="fixture-graph.v1",
    )
    assert result.unmatched_count == 3
    assert all(row.osm_stop_id is None for row in result.stops)


@pytest.mark.parametrize("graph_version", [None, "new-graph"])
def test_run_graph_mismatch_fails_closed(graph_version: str | None) -> None:
    with pytest.raises(GeometryMappingError, match="forecast graph version"):
        service().map_stops([], entity_version="synthetic-entities.v1", graph_version=graph_version)


def test_entity_version_and_loaded_snapshot_mismatch_fail_closed() -> None:
    with pytest.raises(GeometryMappingError, match="forecast entity version"):
        service().map_stops([], entity_version="other", graph_version="fixture-graph.v1")
    with pytest.raises(GeometryMappingError, match="loaded snapshot"):
        ForecastGeometryService(load_crosswalk(FIXTURE), network(), "other")


def test_missing_route_mapping_and_unknown_osm_route_are_visible() -> None:
    original = load_crosswalk(FIXTURE)
    for routes, reason in [
        ((), "missing_route_mapping"),
        ((RouteGeometryLink("canonical-route", ("unknown",)),), "unknown_osm_route"),
    ]:
        mapper = ForecastGeometryService(
            replace(original, routes=routes), network(), original.graph_version
        )
        row = mapper.map_stops(
            [entity("same-name")],
            entity_version=original.entity_version,
            graph_version=original.graph_version,
        ).stops[0]
        assert row.reason == reason
        assert row.status == GeometryMatchStatus.UNMATCHED


def test_duplicate_keys_are_rejected_instead_of_last_write_wins() -> None:
    original = load_crosswalk(FIXTURE)
    with pytest.raises(GeometryMappingError, match="duplicate canonical"):
        ForecastGeometryService(
            replace(original, stops=original.stops * 2), network(), original.graph_version
        )


@pytest.mark.parametrize("value", [True, "101", -1, 0, 1.1])
def test_osm_ids_are_strict_positive_integers(tmp_path: Path, value: object) -> None:
    raw = json.loads(FIXTURE.read_text())
    raw["stops"][0]["osm_stop_ids"] = [value]
    path = tmp_path / "map.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(GeometryMappingError):
        load_crosswalk(path)


def test_json_duplicate_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text('{"schema_version":"forecast-geometry.v1","schema_version":"other"}')
    with pytest.raises(GeometryMappingError):
        load_crosswalk(path)


def test_published_snapshot_fixture_joins_real_coordinates(tmp_path: Path) -> None:
    def write(directory: Path) -> None:
        for name in graph_artifacts.ARTIFACT_NAMES:
            (directory / name).write_bytes((DATA / name).read_bytes())

    version = graph_artifacts.publish_graph(tmp_path, write)
    actual_version, graph, _ = graph_artifacts.load_active_graph_snapshot(tmp_path)
    assert actual_version == version
    node = next(node for node in graph["nodes"] if node["routes"])
    raw = json.loads(FIXTURE.read_text())
    raw["graph_version"] = version
    raw["routes"][0]["osm_route_refs"] = [node["routes"][0]]
    raw["stops"] = [raw["stops"][0]]
    raw["stops"][0]["osm_stop_ids"] = [node["id"]]
    path = tmp_path / "map.json"
    path.write_text(json.dumps(raw))
    result = load_geometry_service(path, tmp_path).map_stops(
        [entity("same-name")],
        entity_version=raw["entity_version"],
        graph_version=version,
    )
    assert result.matched_count == 1
    assert result.stops[0].longitude == node["lon"]
    assert result.stops[0].latitude == node["lat"]


def test_snapshot_version_is_from_the_single_manifest_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write(directory: Path) -> None:
        for name in graph_artifacts.ARTIFACT_NAMES:
            (directory / name).write_bytes((DATA / name).read_bytes())

    version = graph_artifacts.publish_graph(tmp_path, write)
    active = tmp_path / graph_artifacts.STORE_NAME / graph_artifacts.MANIFEST_NAME
    read = graph_artifacts._bytes

    def switch_after_read(path: Path) -> bytes:
        content = read(path)
        if path == active:
            active.write_text("{}")
        return content

    monkeypatch.setattr(graph_artifacts, "_bytes", switch_after_read)
    loaded_version, graph, _ = graph_artifacts.load_active_graph_snapshot(tmp_path)
    assert loaded_version == version
    assert len(graph["nodes"]) == 856


@pytest.mark.parametrize("synthetic", [True, False])
def test_mapping_synthetic_provenance_is_preserved(tmp_path: Path, synthetic: bool) -> None:
    raw = json.loads(FIXTURE.read_text())
    raw["synthetic"] = synthetic
    path = tmp_path / "map.json"
    path.write_text(json.dumps(raw))
    crosswalk = load_crosswalk(path)
    assert crosswalk.synthetic is synthetic
    result = ForecastGeometryService(crosswalk, network(), crosswalk.graph_version).map_stops(
        [entity("same-name")],
        entity_version=crosswalk.entity_version,
        graph_version=crosswalk.graph_version,
    )
    assert result.synthetic is synthetic


@pytest.mark.parametrize("value", [None, "true", 1])
def test_mapping_synthetic_marker_requires_explicit_boolean(tmp_path: Path, value: object) -> None:
    raw = json.loads(FIXTURE.read_text())
    if value is None:
        del raw["synthetic"]
    else:
        raw["synthetic"] = value
    path = tmp_path / "map.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(GeometryMappingError):
        load_crosswalk(path)

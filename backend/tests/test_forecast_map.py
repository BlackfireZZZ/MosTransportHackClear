import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.application.services.forecast_geometry import ForecastGeometryService
from app.application.services.forecast_map import ForecastMapService
from app.domain.forecast import (
    ForecastHorizon,
    ForecastPoint,
    ForecastRunMetadata,
    ForecastSnapshot,
    RouteSummary,
    StopForecastPoint,
    StopLoad,
)
from app.domain.forecast_geometry import (
    ForecastEntity,
    GeometryCrosswalk,
    GeometryMappingError,
    RouteGeometryLink,
    ServingGeometryLink,
    StopGeometryLink,
)
from app.domain.tram_graph import GraphMetadata, TramNetwork, TramStop
from app.infrastructure.forecast_geometry import FileForecastMapProvider, load_crosswalk
from app.schemas.forecast import ForecastResponse

INSTANT = datetime(2026, 10, 1, tzinfo=UTC)
ENTITY = ForecastEntity("route-a", "out", "stop-a")


def snapshot(synthetic: bool = True) -> ForecastSnapshot:
    return ForecastSnapshot(
        RouteSummary(1, "demo", "fixture", "#000000"),
        ForecastHorizon.DAY,
        INSTANT,
        "model",
        [ForecastPoint(INSTANT, 10, 8, 12, None)],
        [StopLoad(7, "duplicate name", 55.7, 37.6, 10, None, 1)],
        stop_points=[
            StopForecastPoint(7, INSTANT, None, "out", 10, 8, 12, None, "stop_bucket_direction")
        ],
        run=ForecastRunMetadata(
            "run",
            "data",
            "source",
            "features",
            "entities",
            "calendar",
            "graph",
            "synthetic_boardings" if synthetic else "boardings",
            "event_count",
            synthetic,
            INSTANT,
            INSTANT,
            None,
            None,
        ),
    )


def mapping(
    *, synthetic: bool = False, candidates: tuple[int, ...] = (701,)
) -> ForecastGeometryService:
    crosswalk = GeometryCrosswalk(
        "map",
        "entities",
        "graph",
        (RouteGeometryLink("route-a", ("А",)),),
        (StopGeometryLink(ENTITY, candidates),),
        synthetic,
        (ServingGeometryLink(1, 7, "out", ENTITY),),
    )
    network = TramNetwork.build(
        GraphMetadata(),
        [
            TramStop(701, "different name", 55.75, 37.65, ("А",)),
            TramStop(702, "different name", 55.76, 37.66, ("А",)),
        ],
        [],
        {},
    )
    return ForecastGeometryService(crosswalk, network, "graph")


def test_explicit_bridge_preserves_one_run_and_values_without_name_or_id_guess() -> None:
    source = snapshot(False)
    result = ForecastMapService(mapping()).enrich(source)
    assert result.points is source.points
    assert result.stop_points is source.stop_points
    assert result.run is source.run
    assert result.map is not None
    assert (result.map.run_id, result.map.mapping_version, result.map.graph_version) == (
        "run",
        "map",
        "graph",
    )
    assert result.map.status == "ready"
    point = result.map.positions[0]
    assert (point.stop_id, point.osm_stop_id, point.longitude, point.latitude) == (
        7,
        701,
        37.65,
        55.75,
    )
    assert point.position_kind == "osm"
    assert ForecastResponse.model_validate(result).map is not None


@pytest.mark.parametrize("synthetic", [False, True])
def test_missing_mapping_uses_only_explicit_synthetic_demo_positions(synthetic: bool) -> None:
    result = ForecastMapService().enrich(snapshot(synthetic))
    assert result.map is not None
    assert result.map.reason == "mapping_not_configured"
    assert result.map.unmatched_count == 1
    assert result.map.matched_count == 0
    row = result.map.positions[0]
    assert row.osm_stop_id is None
    assert row.position_kind == ("synthetic_demo" if synthetic else "unavailable")
    assert row.longitude == (37.6 if synthetic else None)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("entity_version", "other", "entity_version_mismatch"),
        ("graph_version", "other", "graph_version_mismatch"),
        ("graph_version", None, "forecast_graph_version_unavailable"),
        ("identity_namespace", "canonical", "identity_namespace_unsupported"),
    ],
)
def test_version_mismatch_does_not_join(field: str, value: str | None, reason: str) -> None:
    source = snapshot(False)
    assert source.run
    source = replace(source, run=replace(source.run, **{field: value}))
    result = ForecastMapService(mapping()).enrich(source)
    assert result.map and result.map.reason == reason
    assert result.map.positions[0].longitude is None


def test_no_serving_bridge_guess_or_aggregated_direction_guess() -> None:
    source = snapshot(False)
    assert source.stop_points
    for changes in [{"stop_id": 701}, {"direction_id": None}, {"direction_id": "return"}]:
        result = ForecastMapService(mapping()).enrich(
            replace(source, stop_points=[replace(source.stop_points[0], **changes)])
        )
        assert result.map and result.map.unmatched_count == 1
        assert result.map.positions[0].osm_stop_id is None


def test_ambiguous_matches_are_counted_and_never_choose_a_candidate() -> None:
    result = ForecastMapService(mapping(candidates=(701, 702))).enrich(snapshot(False))
    assert result.map and result.map.ambiguous_count == 1
    assert result.map.positions[0].longitude is None


def test_synthetic_crosswalk_cannot_locate_real_forecasts() -> None:
    result = ForecastMapService(mapping(synthetic=True)).enrich(snapshot(False))
    assert result.map and result.map.reason == "synthetic_mapping_for_real_forecast"
    assert result.map.positions[0].position_kind == "unavailable"


def test_repeated_buckets_do_not_duplicate_geometry_and_empty_stays_empty() -> None:
    source = snapshot()
    source = replace(source, stop_points=(source.stop_points or []) * 2)
    result = ForecastMapService().enrich(source)
    assert result.map and len(result.map.positions) == 1
    result = ForecastMapService().enrich(replace(source, stop_points=[]))
    assert result.map and result.map.positions == ()


async def test_bad_config_preserves_values_and_exposes_failure(tmp_path: Path) -> None:
    source = snapshot(False)
    provider = FileForecastMapProvider(tmp_path / "absent.json", tmp_path)
    result = await provider.enrich(source)
    assert result.points is source.points
    assert result.map and result.map.reason == "mapping_configuration_invalid"
    assert result.map.positions[0].longitude is None


def test_bridge_json_is_explicit_and_duplicate_keys_rejected(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "forecast_geometry.v1.json"
    raw = json.loads(fixture.read_text())
    raw["serving_links"] = [
        {
            "route_id": 1,
            "stop_id": 7,
            "direction_id": "out",
            "entity": {
                "route_id": "canonical-route",
                "stop_id": "same-name",
                "direction_id": "out",
            },
        }
    ]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(raw))
    assert load_crosswalk(path).serving_links[0].entity.stop_id == "same-name"
    mapper = mapping()
    with pytest.raises(GeometryMappingError, match="duplicate serving"):
        ForecastMapService(
            ForecastGeometryService(
                replace(mapper.crosswalk, serving_links=mapper.crosswalk.serving_links * 2),
                TramNetwork.build(GraphMetadata(), [], [], {}),
                "graph",
            )
        )
    raw["serving_links"][0]["stop_id"] = "7"
    path.write_text(json.dumps(raw))
    with pytest.raises(GeometryMappingError):
        load_crosswalk(path)


def test_synthetic_graph_provenance_cannot_be_hidden_by_mapping() -> None:
    base = mapping()
    graph = TramNetwork.build(
        GraphMetadata(synthetic=True),
        [
            TramStop(701, "demo", 55.75, 37.65, ("А",)),
        ],
        [],
        {},
    )
    mapper = ForecastGeometryService(base.crosswalk, graph, "graph")
    assert mapper.map_stops([ENTITY], entity_version="entities", graph_version="graph").synthetic
    result = ForecastMapService(mapper).enrich(snapshot(False))
    assert result.map and result.map.reason == "synthetic_mapping_for_real_forecast"
    assert result.map.positions[0].longitude is None


async def test_configured_provider_loads_bridge_and_verified_snapshot_together(
    tmp_path: Path,
) -> None:
    from app.infrastructure.graph_artifacts import (
        ARTIFACT_NAMES,
        load_active_graph_snapshot,
        publish_graph,
    )

    data = Path(__file__).resolve().parents[2] / "data"

    def write(directory: Path) -> None:
        for name in ARTIFACT_NAMES:
            (directory / name).write_bytes((data / name).read_bytes())

    version = publish_graph(tmp_path, write)
    _, graph, _ = load_active_graph_snapshot(tmp_path)
    node = next(node for node in graph["nodes"] if node["routes"])
    fixture = Path(__file__).parent / "fixtures" / "forecast_geometry.v1.json"
    raw = json.loads(fixture.read_text())
    raw["graph_version"] = version
    raw["routes"] = [{"route_id": "canonical-route", "osm_route_refs": [node["routes"][0]]}]
    raw["stops"] = [
        {
            "route_id": "canonical-route",
            "direction_id": "out",
            "stop_id": "canonical-stop",
            "osm_stop_ids": [node["id"]],
        }
    ]
    raw["serving_links"] = [
        {
            "route_id": 1,
            "stop_id": 7,
            "direction_id": "out",
            "entity": {
                "route_id": "canonical-route",
                "direction_id": "out",
                "stop_id": "canonical-stop",
            },
        }
    ]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(raw))
    source = snapshot()
    assert source.run
    source = replace(
        source, run=replace(source.run, graph_version=version, entity_version=raw["entity_version"])
    )
    result = await FileForecastMapProvider(path, tmp_path).enrich(source)
    assert result.map and result.map.status == "ready"
    assert result.map.positions[0].osm_stop_id == node["id"]
    assert result.map.positions[0].longitude == node["lon"]
    assert result.map.run_id == source.run.run_id

"""Route projections preserve counts through the existing backend response seam."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.application.services.forecast_map import ForecastMapService  # noqa: E402
from app.domain.forecast import (  # noqa: E402
    ForecastHorizon,
    ForecastPoint,
    ForecastRunMetadata,
    ForecastSnapshot,
    ResolvedForecastSelection,
    RouteSummary,
)
from app.schemas.forecast import ForecastResponse  # noqa: E402

from tramflow_ml.route_models.artifact import RouteForecast, RoutePoint  # noqa: E402

MOSCOW = ZoneInfo("Europe/Moscow")
ORIGIN = datetime(2025, 11, 1, tzinfo=MOSCOW)
ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
SERVING_IDS = {route: 1000 + index for index, route in enumerate(ROUTES)}


@pytest.fixture(scope="module")
def artifact() -> RouteForecast:
    return RouteForecast(
        schema_version="route-forecast.experimental.v1",
        run_id="compatibility-fixture",
        dataset_id="constructed-counts-fixture",
        source_hash="a" * 64,
        source_version="fixture-v1",
        feature_version="route-features-fixture-v1",
        model_version="route-model-fixture-v1",
        forecast_origin=ORIGIN,
        data_cutoff=ORIGIN,
        generated_at=ORIGIN,
        points=[
            RoutePoint(
                route=route,
                bucket_start=ORIGIN + timedelta(hours=hour),
                predicted=float(route + hour % 24) + 0.25,
            )
            for route in ROUTES
            for hour in range(61 * 24)
        ],
    )


@pytest.mark.parametrize(
    ("horizon", "start", "days", "buckets"),
    [
        ("day", ORIGIN, 1, 24),
        ("day", datetime(2025, 12, 31, tzinfo=MOSCOW), 1, 24),
        ("month", ORIGIN, 30, 30),
        ("month", datetime(2025, 12, 1, tzinfo=MOSCOW), 31, 31),
    ],
)
def test_route_projection_survives_existing_http_schema(
    artifact: RouteForecast, horizon: str, start: datetime, days: int, buckets: int
) -> None:
    source_route = 17
    end = start + timedelta(days=days)
    projected = artifact.project(route=source_route, start=start, horizon=horizon)
    run = ForecastRunMetadata(
        run_id=artifact.run_id,
        dataset_id=artifact.dataset_id,
        source_version=artifact.source_version,
        feature_version=artifact.feature_version,
        entity_version="explicit-route-fixture-v1",
        calendar_version="moscow-midnight.v1",
        graph_version=None,
        target=artifact.target,
        unit=artifact.unit,
        synthetic=artifact.synthetic,
        forecast_origin=artifact.forecast_origin,
        data_cutoff=artifact.data_cutoff,
        interval_level=None,
        interval_method=None,
    )
    snapshot = ForecastSnapshot(
        route=RouteSummary(SERVING_IDS[source_route], str(source_route), "Route 17", "#d9342b"),
        horizon=ForecastHorizon(horizon),
        generated_at=artifact.generated_at,
        model_version=artifact.model_version,
        points=[ForecastPoint(p.timestamp, p.predicted, None, None, None) for p in projected],
        stops=[],
        stop_points=[],
        run=run,
        selection=ResolvedForecastSelection(start, end, None, None, "route_bucket"),
    )
    response = ForecastResponse.model_validate(ForecastMapService().enrich(snapshot))
    restored = ForecastResponse.model_validate_json(response.model_dump_json())

    expected = sum(
        p.predicted
        for p in artifact.points
        if p.route == source_route and start <= p.bucket_start < end
    )
    assert restored == response
    assert len(restored.points) == buckets
    assert sum(p.predicted_passengers for p in restored.points) == expected
    assert [p.timestamp for p in restored.points] == [p.timestamp for p in projected]
    assert restored.route.id == SERVING_IDS[source_route] != source_route
    assert restored.route.number == str(source_route)
    assert restored.model_version == artifact.model_version
    assert restored.generated_at == artifact.generated_at
    assert restored.run is not None
    assert restored.run.run_id == artifact.run_id
    assert restored.run.dataset_id == artifact.dataset_id
    assert restored.run.feature_version == artifact.feature_version
    assert restored.run.source_version == artifact.source_version
    assert restored.run.data_cutoff == artifact.data_cutoff
    assert restored.run.forecast_origin == artifact.forecast_origin
    assert (restored.run.target, restored.run.unit, restored.run.synthetic) == (
        "validation_count",
        "event_count",
        False,
    )
    assert restored.stops == [] and restored.stop_points == []
    assert restored.peak_load_percent is None
    assert all(
        p.capacity is None and p.lower_bound is None and p.upper_bound is None
        for p in restored.points
    )
    assert restored.map is not None
    assert restored.map.status == "unavailable" and restored.map.positions == []
    assert restored.selection is not None
    assert restored.selection.aggregation_key == "route_bucket"
    assert restored.selection.stop_id is None and restored.selection.direction_id is None


def test_route_projection_rejects_unsupported_year(artifact: RouteForecast) -> None:
    with pytest.raises(ValueError):
        artifact.project(route=17, start=ORIGIN, horizon="year")


@pytest.mark.parametrize("fault", ["duplicate", "missing", "naive", "negative", "nan"])
def test_route_artifact_rejects_invalid_grid(artifact: RouteForecast, fault: str) -> None:
    payload = artifact.model_dump(mode="python")
    points = list(payload["points"])
    if fault == "duplicate":
        points[-1] = points[0]
    elif fault == "missing":
        points.pop()
    elif fault == "naive":
        points[0]["bucket_start"] = ORIGIN.replace(tzinfo=None)
    else:
        points[0]["predicted"] = -1.0 if fault == "negative" else float("nan")
    payload["points"] = points
    with pytest.raises(ValueError):
        RouteForecast.model_validate(payload)


@pytest.mark.parametrize(
    ("start", "horizon"),
    [
        (ORIGIN + timedelta(days=1), "month"),
        (ORIGIN + timedelta(hours=1), "day"),
        (ORIGIN.replace(tzinfo=None), "day"),
        (ORIGIN - timedelta(days=1), "day"),
        (ORIGIN + timedelta(days=61), "day"),
        (datetime(2026, 1, 1, tzinfo=MOSCOW), "month"),
    ],
)
def test_route_projection_rejects_invalid_window(
    artifact: RouteForecast, start: datetime, horizon: str
) -> None:
    with pytest.raises(ValueError):
        artifact.project(route=17, start=start, horizon=horizon)

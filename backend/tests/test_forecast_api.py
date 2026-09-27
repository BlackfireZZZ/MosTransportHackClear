from collections.abc import Iterator
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_forecast_service
from app.application.services.forecast import ForecastService
from app.domain.forecast import (
    ForecastHorizon,
    ForecastPoint,
    ForecastSelection,
    ForecastSnapshot,
    RouteStopSummary,
    RouteSummary,
    StopLoad,
)
from app.main import create_app


class ForecastRepositoryFixture:
    def __init__(self) -> None:
        self.route = RouteSummary(id=1, number="Т1", name="Тестовый маршрут", color="#d9342b")
        self.snapshots = {
            horizon: ForecastSnapshot(
                route=self.route,
                horizon=horizon,
                generated_at=datetime(2026, 9, 19, tzinfo=UTC),
                model_version="synthetic-http-test-v1",
                points=[
                    ForecastPoint(datetime(2026, 9, 20, tzinfo=UTC), 120, 100, 140, 240),
                    ForecastPoint(datetime(2026, 9, 21, tzinfo=UTC), 180, 150, 210, 180),
                ],
                stops=[StopLoad(10, "Остановка", 55.75, 37.62, 90, 50, 1)],
            )
            for horizon in ForecastHorizon
        }
        self.calls: list[tuple[int, ForecastHorizon]] = []

    async def list_routes(self) -> list[RouteSummary]:
        return [self.route]

    async def list_stops(self, route_id: int) -> list[RouteStopSummary] | None:
        return [] if route_id == self.route.id else None

    async def get_snapshot(
        self, route_id: int, horizon: ForecastHorizon, selection: ForecastSelection | None = None
    ) -> ForecastSnapshot | None:
        self.calls.append((route_id, horizon))
        return self.snapshots.get(horizon) if route_id == self.route.id else None


@pytest.fixture
def repository() -> ForecastRepositoryFixture:
    return ForecastRepositoryFixture()


@pytest.fixture
def client(repository: ForecastRepositoryFixture) -> Iterator[TestClient]:
    application = create_app()
    application.dependency_overrides[get_forecast_service] = lambda: ForecastService(repository)
    try:
        with TestClient(application) as test_client:
            yield test_client
    finally:
        application.dependency_overrides.clear()


@pytest.mark.parametrize("horizon", list(ForecastHorizon))
def test_forecast_response_contract(
    client: TestClient, repository: ForecastRepositoryFixture, horizon: ForecastHorizon
) -> None:
    response = client.get("/api/v1/forecasts", params={"route_id": 1, "horizon": horizon})

    assert response.status_code == 200
    assert response.json() == {
        "run": None,
        "map": None,
        "selection": None,
        "stop_points": None,
        "route": {"id": 1, "number": "Т1", "name": "Тестовый маршрут", "color": "#d9342b"},
        "horizon": horizon.value,
        "generated_at": "2026-09-19T00:00:00Z",
        "model_version": "synthetic-http-test-v1",
        "peak_passengers": 180,
        "peak_load_percent": 100,
        "points": [
            {
                "timestamp": "2026-09-20T00:00:00Z",
                "predicted_passengers": 120,
                "lower_bound": 100,
                "upper_bound": 140,
                "capacity": 240,
            },
            {
                "timestamp": "2026-09-21T00:00:00Z",
                "predicted_passengers": 180,
                "lower_bound": 150,
                "upper_bound": 210,
                "capacity": 180,
            },
        ],
        "stops": [
            {
                "id": 10,
                "name": "Остановка",
                "latitude": 55.75,
                "longitude": 37.62,
                "predicted_passengers": 90,
                "load_percent": 50,
                "sequence": 1,
            }
        ],
    }
    assert repository.calls == [(1, horizon)]


def test_missing_published_stop_catalog_is_explicit(client: TestClient) -> None:
    response = client.get("/api/v1/routes/1/stops")
    assert response.status_code == 409
    assert "estimated GTFS stops" in response.json()["detail"]
    assert client.get("/api/v1/routes/999/stops").status_code == 404


@pytest.mark.parametrize("horizon", list(ForecastHorizon))
def test_zero_change_scenario_is_identity(
    client: TestClient, repository: ForecastRepositoryFixture, horizon: ForecastHorizon
) -> None:
    before = deepcopy(repository.snapshots)
    baseline = client.get("/api/v1/forecasts", params={"route_id": 1, "horizon": horizon}).json()
    response = client.post("/api/v1/scenarios/evaluate", json={"route_id": 1, "horizon": horizon})

    assert response.status_code == 200
    assert response.json() == {
        "baseline_peak_load_percent": 100,
        "scenario_peak_load_percent": 100,
        "passenger_delta": 0,
        "capacity_delta": 0,
        "affected_stops": baseline["stops"],
        "solver_version": "network-flow-baseline-v1",
    }
    assert repository.snapshots == before
    assert repository.calls == [(1, horizon), (1, horizon)]


@pytest.mark.parametrize("horizon", list(ForecastHorizon))
def test_changed_scenario_preserves_baseline(
    client: TestClient, repository: ForecastRepositoryFixture, horizon: ForecastHorizon
) -> None:
    before = deepcopy(repository.snapshots)
    params = {"route_id": 1, "horizon": horizon}
    baseline = client.get("/api/v1/forecasts", params=params).json()
    response = client.post(
        "/api/v1/scenarios/evaluate",
        json={
            **params,
            "additional_vehicles": 5,
            "interval_change_percent": 20,
            "demand_change_percent": 50,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["baseline_peak_load_percent"] == 100
    assert result["scenario_peak_load_percent"] == pytest.approx(125)
    assert result["passenger_delta"] == pytest.approx(150)
    assert result["capacity_delta"] == pytest.approx(20)
    assert result["affected_stops"] == [
        {**baseline["stops"][0], "predicted_passengers": 135, "load_percent": 62.5}
    ]
    assert repository.snapshots == before
    assert client.get("/api/v1/forecasts", params=params).json() == baseline


def test_default_horizon_is_day(client: TestClient, repository: ForecastRepositoryFixture) -> None:
    assert client.get("/api/v1/forecasts", params={"route_id": 1}).json()["horizon"] == "day"
    assert client.post("/api/v1/scenarios/evaluate", json={"route_id": 1}).status_code == 200
    assert repository.calls == [(1, ForecastHorizon.DAY), (1, ForecastHorizon.DAY)]


@pytest.mark.parametrize("horizon", list(ForecastHorizon))
@pytest.mark.parametrize("missing", ["route", "snapshot"])
def test_missing_baseline_returns_404(
    client: TestClient,
    repository: ForecastRepositoryFixture,
    horizon: ForecastHorizon,
    missing: str,
) -> None:
    route_id = 999 if missing == "route" else 1
    if missing == "snapshot":
        del repository.snapshots[horizon]
    params = {"route_id": route_id, "horizon": horizon}
    forecast = client.get("/api/v1/forecasts", params=params)
    scenario = client.post("/api/v1/scenarios/evaluate", json=params)

    assert forecast.status_code == scenario.status_code == 404
    assert forecast.json() == {"detail": "Forecast not found for the selected route and horizon"}
    assert scenario.json() == {"detail": "Baseline forecast not found for the scenario"}


@pytest.mark.parametrize(
    "params,field",
    [
        ({}, "route_id"),
        ({"route_id": 0}, "route_id"),
        ({"route_id": -1}, "route_id"),
        ({"route_id": "abc"}, "route_id"),
        ({"route_id": 1.5}, "route_id"),
        ({"route_id": 1, "horizon": "week"}, "horizon"),
    ],
)
def test_invalid_forecast_query_does_not_read_repository(
    client: TestClient, repository: ForecastRepositoryFixture, params: dict, field: str
) -> None:
    response = client.get("/api/v1/forecasts", params=params)
    assert response.status_code == 422
    assert any(error["loc"] == ["query", field] for error in response.json()["detail"])
    assert repository.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("route_id", 0),
        ("route_id", -1),
        ("route_id", "abc"),
        ("route_id", 1.5),
        ("horizon", "week"),
        ("additional_vehicles", -6),
        ("additional_vehicles", 21),
        ("additional_vehicles", 1.5),
        ("interval_change_percent", -75.01),
        ("interval_change_percent", 200.01),
        ("demand_change_percent", -90.01),
        ("demand_change_percent", 300.01),
        ("interval_change_percent", "NaN"),
        ("demand_change_percent", "Infinity"),
    ],
)
def test_invalid_scenario_does_not_read_repository(
    client: TestClient, repository: ForecastRepositoryFixture, field: str, value: object
) -> None:
    response = client.post("/api/v1/scenarios/evaluate", json={"route_id": 1, field: value})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", field] for error in response.json()["detail"])
    assert repository.calls == []


def test_missing_scenario_route_is_rejected(
    client: TestClient, repository: ForecastRepositoryFixture
) -> None:
    response = client.post("/api/v1/scenarios/evaluate", json={})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "route_id"]
    assert repository.calls == []


@pytest.mark.parametrize(
    "vehicles,interval,demand,load,passengers,capacity",
    [
        (-5, 200, -90, 40, -270, -75),
        (20, -75, 300, 400 / 3.35, 900, 235),
    ],
)
def test_inclusive_scenario_bounds_and_capacity_floor(
    client: TestClient,
    vehicles: int,
    interval: float,
    demand: float,
    load: float,
    passengers: float,
    capacity: float,
) -> None:
    response = client.post(
        "/api/v1/scenarios/evaluate",
        json={
            "route_id": 1,
            "additional_vehicles": vehicles,
            "interval_change_percent": interval,
            "demand_change_percent": demand,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["scenario_peak_load_percent"] == pytest.approx(load)
    assert result["passenger_delta"] == pytest.approx(passengers)
    assert result["capacity_delta"] == pytest.approx(capacity)


@pytest.mark.parametrize(
    "params",
    [
        {"start": "2026-09-20T00:00:00Z"},
        {"end": "2026-09-21T00:00:00Z"},
        {"start": "2026-09-20T00:00:00", "end": "2026-09-21T00:00:00"},
        {"start": "2026-09-20T00:00:00Z", "end": "2026-09-20T00:00:00Z"},
        {"start": "2026-09-20T00:00:00Z", "end": "2026-09-22T00:00:00Z"},
        {"direction_id": " "},
        {"stop_id": 0},
    ],
)
def test_invalid_bounded_selection_is_rejected_before_query(
    client: TestClient,
    repository: ForecastRepositoryFixture,
    params: dict,
) -> None:
    response = client.get("/api/v1/forecasts", params={"route_id": 1, **params})
    assert response.status_code == 422
    assert not repository.calls

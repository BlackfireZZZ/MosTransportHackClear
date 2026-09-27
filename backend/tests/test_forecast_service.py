from datetime import UTC, datetime

import pytest

from app.application.services.forecast import ForecastService
from app.domain.forecast import (
    ForecastHorizon,
    ForecastPoint,
    ForecastSelection,
    ForecastSnapshot,
    RouteSummary,
    ScenarioParameters,
    StopLoad,
)


class FakeForecastRepository:
    def __init__(self, snapshot: ForecastSnapshot) -> None:
        self.snapshot = snapshot

    async def list_routes(self) -> list[RouteSummary]:
        return [self.snapshot.route]

    async def get_snapshot(
        self, route_id: int, horizon: ForecastHorizon, selection: ForecastSelection | None = None
    ) -> ForecastSnapshot | None:
        if route_id != self.snapshot.route.id or horizon != self.snapshot.horizon:
            return None
        return self.snapshot


@pytest.fixture
def snapshot() -> ForecastSnapshot:
    return ForecastSnapshot(
        route=RouteSummary(id=1, number="Т1", name="Тестовый маршрут", color="#d9342b"),
        horizon=ForecastHorizon.DAY,
        generated_at=datetime(2026, 9, 19, tzinfo=UTC),
        model_version="test-v1",
        points=[
            ForecastPoint(
                timestamp=datetime(2026, 9, 20, 8, tzinfo=UTC),
                predicted_passengers=180,
                lower_bound=160,
                upper_bound=200,
                capacity=180,
            )
        ],
        stops=[
            StopLoad(
                id=1,
                name="Остановка",
                latitude=55.75,
                longitude=37.62,
                predicted_passengers=90,
                load_percent=50,
                sequence=1,
            )
        ],
    )


async def test_scenario_reduces_load_when_vehicle_is_added(snapshot: ForecastSnapshot) -> None:
    service = ForecastService(FakeForecastRepository(snapshot))

    result = await service.evaluate_scenario(
        route_id=1,
        horizon=ForecastHorizon.DAY,
        parameters=ScenarioParameters(additional_vehicles=2),
    )

    assert result is not None
    assert result.scenario_peak_load_percent < result.baseline_peak_load_percent
    assert result.capacity_delta == pytest.approx(16)


async def test_scenario_propagates_demand_change_to_stops(snapshot: ForecastSnapshot) -> None:
    service = ForecastService(FakeForecastRepository(snapshot))

    result = await service.evaluate_scenario(
        route_id=1,
        horizon=ForecastHorizon.DAY,
        parameters=ScenarioParameters(demand_change_percent=25),
    )

    assert result is not None
    assert result.affected_stops[0].predicted_passengers == pytest.approx(112.5)
    assert result.passenger_delta == pytest.approx(45)

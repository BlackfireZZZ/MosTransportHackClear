from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import ForecastServiceDep
from app.domain.forecast import (
    ForecastDataConflict,
    ForecastHorizon,
    ForecastQueryError,
    ForecastSelection,
    ScenarioParameters,
)
from app.schemas.forecast import (
    ForecastResponse,
    RouteResponse,
    RouteStopResponse,
    ScenarioRequest,
    ScenarioResponse,
)

router = APIRouter(tags=["forecast"])


@router.get("/routes", response_model=list[RouteResponse])
async def list_routes(service: ForecastServiceDep) -> list[RouteResponse]:
    routes = await service.list_routes()
    return [RouteResponse.model_validate(route) for route in routes]


@router.get("/forecasts", response_model=ForecastResponse)
async def get_forecast(
    service: ForecastServiceDep,
    route_id: int = Query(gt=0),
    horizon: ForecastHorizon = ForecastHorizon.DAY,
    stop_id: int | None = Query(default=None, gt=0),
    direction_id: str | None = Query(default=None, min_length=1, max_length=128),
    start: datetime | None = None,
    end: datetime | None = None,
) -> ForecastResponse:
    try:
        selection = ForecastSelection(stop_id, direction_id, start, end)
        snapshot = await service.get_forecast(route_id, horizon, selection)
    except ForecastQueryError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ForecastDataConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No forecast points for the selected route, horizon and interval"
                if start is not None
                else "Forecast not found for the selected route and horizon"
            ),
        )
    return ForecastResponse.model_validate(snapshot)


@router.get("/routes/{route_id}/stops", response_model=list[RouteStopResponse])
async def list_route_stops(route_id: int, service: ForecastServiceDep) -> list[RouteStopResponse]:
    try:
        stops = await service.list_stops(route_id)
    except ForecastQueryError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if stops is None:
        raise HTTPException(status_code=404, detail="Route not found")
    if not stops:
        raise HTTPException(
            status_code=409,
            detail="No published stop catalog for this route; estimated GTFS stops and "
            "directions are available through POST /api/v1/planning/forecast",
        )
    return [RouteStopResponse.model_validate(stop) for stop in stops]


@router.post("/scenarios/evaluate", response_model=ScenarioResponse)
async def evaluate_scenario(
    request: ScenarioRequest, service: ForecastServiceDep
) -> ScenarioResponse:
    result = await service.evaluate_scenario(
        request.route_id,
        request.horizon,
        ScenarioParameters(
            additional_vehicles=request.additional_vehicles,
            interval_change_percent=request.interval_change_percent,
            demand_change_percent=request.demand_change_percent,
        ),
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Baseline forecast not found for the scenario",
        )
    return ScenarioResponse.model_validate(result)

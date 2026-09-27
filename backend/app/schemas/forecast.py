from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.forecast import ForecastHorizon


class RouteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    number: str
    name: str
    color: str


class ForecastPointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    predicted_passengers: float = Field(ge=0)
    lower_bound: float | None = Field(ge=0)
    upper_bound: float | None = Field(ge=0)
    capacity: float | None = Field(ge=0)


class StopLoadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    latitude: float
    longitude: float
    predicted_passengers: float = Field(ge=0)
    load_percent: float | None = Field(ge=0)
    sequence: int = Field(ge=1)


class ForecastRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    dataset_id: str
    source_version: str
    feature_version: str
    entity_version: str
    calendar_version: str
    graph_version: str | None
    target: str
    unit: str
    synthetic: bool
    forecast_origin: datetime
    data_cutoff: datetime
    interval_level: float | None
    interval_method: str | None
    identity_namespace: str


class ForecastSelectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    start: datetime
    end: datetime
    stop_id: int | None
    direction_id: str | None
    aggregation_key: str
    interval_aggregation: str


class RouteStopResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    latitude: float
    longitude: float
    sequence: int


class StopForecastPointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stop_id: int
    timestamp: datetime
    bucket_end: datetime | None
    direction_id: str | None
    predicted_passengers: float
    lower_bound: float | None
    upper_bound: float | None
    capacity: float | None
    aggregation_scope: str


class ForecastMapPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stop_id: int
    direction_id: str | None
    status: Literal["matched", "unmatched", "ambiguous"]
    reason: str
    position_kind: Literal["osm", "synthetic_demo", "unavailable"]
    osm_stop_id: int | None
    longitude: float | None
    latitude: float | None


class ForecastMapResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str | None
    entity_version: str | None
    graph_version: str | None
    mapping_version: str | None
    synthetic: bool
    status: Literal["ready", "partial", "unavailable"]
    reason: str
    matched_count: int
    unmatched_count: int
    ambiguous_count: int
    positions: list[ForecastMapPositionResponse]


class ForecastResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    route: RouteResponse
    horizon: ForecastHorizon
    generated_at: datetime
    model_version: str
    peak_passengers: float
    peak_load_percent: float | None
    points: list[ForecastPointResponse]
    stops: list[StopLoadResponse]
    stop_points: list[StopForecastPointResponse] | None = None
    run: ForecastRunResponse | None = None
    selection: ForecastSelectionResponse | None = None
    map: ForecastMapResponse | None = None


class ScenarioRequest(BaseModel):
    route_id: int = Field(gt=0)
    horizon: ForecastHorizon = ForecastHorizon.DAY
    additional_vehicles: int = Field(default=0, ge=-5, le=20)
    interval_change_percent: float = Field(default=0, ge=-75, le=200)
    demand_change_percent: float = Field(default=0, ge=-90, le=300)


class ScenarioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    baseline_peak_load_percent: float | None
    scenario_peak_load_percent: float | None
    passenger_delta: float
    capacity_delta: float
    affected_stops: list[StopLoadResponse]
    solver_version: str

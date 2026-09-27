from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Factor(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    multiplier: float = Field(default=1, ge=0, le=3)


class Factors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weather: Factor = Field(default_factory=Factor)
    calendar: Factor = Field(default_factory=Factor)
    events: Factor = Field(default_factory=Factor)
    traffic: Factor = Field(default_factory=Factor)


class SourceFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calendar: bool = False
    weather: bool = False
    traffic: bool = False
    events: bool = False


class PlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_mode: Literal["approved", "external_experiment", "stop_model"] = "approved"
    source_enabled: SourceFlags = Field(default_factory=SourceFlags)
    route: str = Field(min_length=1, max_length=10)
    start_date: date
    horizon: Literal["day", "month", "year"] = "day"
    stop_id: str | None = Field(default=None, max_length=80)
    direction: str | None = Field(default=None, max_length=30)
    factors: Factors = Field(default_factory=Factors)


class PlanningStop(BaseModel):
    stop_id: str
    name: str
    direction: str
    latitude: float | None
    longitude: float | None


class SpatialPoint(PlanningStop):
    baseline: float
    scenario: float


class PlanningPoint(BaseModel):
    timestamp: datetime
    bucket_end: datetime
    basis: Literal["competition_period", "scenario_projection"]
    baseline: float
    scenario: float
    route_baseline: float
    route_scenario: float
    route_unallocated_baseline: float
    route_unallocated_scenario: float
    unallocated_baseline: float
    unallocated_scenario: float
    spatial: list[SpatialPoint]


class PlanningSource(BaseModel):
    id: str
    label: str
    status: str
    detail: str
    url: str | None


class PlanningResponse(PlanningRequest):
    experimental_evaluation: str | None
    run_id: str
    model_version: str
    generated_at: str
    unit: Literal["event_count"]
    timezone: Literal["Europe/Moscow"]
    qualitative: bool
    warnings: list[str]
    sources: list[PlanningSource]
    points: list[PlanningPoint]
    stops: list[PlanningStop]
    routes: list[str]
    provenance: dict[str, str]

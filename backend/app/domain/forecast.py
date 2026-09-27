import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from app.domain.forecast_geometry import ForecastMap


class ForecastHorizon(StrEnum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


@dataclass(frozen=True, slots=True)
class RouteSummary:
    id: int
    number: str
    name: str
    color: str


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    timestamp: datetime
    predicted_passengers: float
    lower_bound: float | None
    upper_bound: float | None
    capacity: float | None


@dataclass(frozen=True, slots=True)
class StopLoad:
    id: int
    name: str
    latitude: float
    longitude: float
    predicted_passengers: float
    load_percent: float | None
    sequence: int


class ForecastQueryError(ValueError):
    """Invalid or excessive forecast selection."""


class ForecastDataConflict(RuntimeError):
    """Stored rows cannot be combined without double counting."""


WINDOW_LIMITS = {
    ForecastHorizon.DAY: timedelta(days=1),
    ForecastHorizon.MONTH: timedelta(days=31),
    ForecastHorizon.YEAR: timedelta(days=366),
}
BUCKET_LIMITS = {ForecastHorizon.DAY: 24, ForecastHorizon.MONTH: 31, ForecastHorizon.YEAR: 12}


def require_additive_target(target: str, unit: str, source_count: int) -> None:
    if source_count > 1 and (target, unit) not in {
        ("synthetic_boardings", "event_count"),
        ("validation_count", "event_count"),
        ("boarding_count", "passengers"),
    }:
        raise ForecastDataConflict(
            "forecast target cannot be summed across sources or time buckets"
        )


def finite_load_percent(predicted: float, capacity: float | None) -> float | None:
    if capacity is None or capacity <= 0:
        return None
    result = predicted / capacity * 100
    if not math.isfinite(result):
        raise ForecastDataConflict("load percentage exceeds finite range")
    return result


@dataclass(frozen=True, slots=True)
class ForecastSelection:
    stop_id: int | None = None
    direction_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None

    def validate(self, horizon: ForecastHorizon) -> None:
        if self.stop_id is not None and self.stop_id <= 0:
            raise ForecastQueryError("stop_id must be positive")
        if self.direction_id is not None and (
            not self.direction_id.strip() or len(self.direction_id) > 128
        ):
            raise ForecastQueryError("direction_id must contain 1 to 128 characters")
        if (self.start is None) != (self.end is None):
            raise ForecastQueryError("start and end must be supplied together")
        if self.start is not None and self.end is not None:
            if any(
                value.tzinfo is None or value.utcoffset() is None
                for value in (self.start, self.end)
            ):
                raise ForecastQueryError("start and end must include timezone offsets")
            try:
                duration = self.end.astimezone(UTC) - self.start.astimezone(UTC)
            except (OverflowError, ValueError) as error:
                raise ForecastQueryError("window timestamps exceed supported range") from error
            if not timedelta(0) < duration <= WINDOW_LIMITS[horizon]:
                raise ForecastQueryError("window must be positive and within the horizon limit")


@dataclass(frozen=True, slots=True)
class ResolvedForecastSelection:
    start: datetime
    end: datetime
    stop_id: int | None
    direction_id: str | None
    aggregation_key: str
    interval_aggregation: str = "single_source_or_unavailable"


@dataclass(frozen=True, slots=True)
class RouteStopSummary:
    id: int
    name: str
    latitude: float
    longitude: float
    sequence: int


@dataclass(frozen=True, slots=True)
class ForecastRunMetadata:
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
    identity_namespace: str = "serving-surrogate-integer"


@dataclass(frozen=True, slots=True)
class StopForecastPoint:
    stop_id: int
    timestamp: datetime
    bucket_end: datetime | None
    direction_id: str | None
    predicted_passengers: float
    lower_bound: float | None
    upper_bound: float | None
    capacity: float | None
    aggregation_scope: str


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    route: RouteSummary
    horizon: ForecastHorizon
    generated_at: datetime
    model_version: str
    points: list[ForecastPoint]
    stops: list[StopLoad]
    stop_points: list[StopForecastPoint] | None = None
    run: ForecastRunMetadata | None = None
    selection: ResolvedForecastSelection | None = None
    map: ForecastMap | None = None

    @property
    def peak_passengers(self) -> float:
        return max((point.predicted_passengers for point in self.points), default=0)

    @property
    def peak_load_percent(self) -> float | None:
        if not self.points or any(
            point.capacity is None or point.capacity <= 0 for point in self.points
        ):
            return None
        percentages = [
            finite_load_percent(point.predicted_passengers, point.capacity)
            for point in self.points
            if point.capacity
        ]
        return max(value for value in percentages if value is not None)


@dataclass(frozen=True, slots=True)
class ScenarioParameters:
    additional_vehicles: int = 0
    interval_change_percent: float = 0
    demand_change_percent: float = 0


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    baseline_peak_load_percent: float | None
    scenario_peak_load_percent: float | None
    passenger_delta: float
    capacity_delta: float
    affected_stops: list[StopLoad]
    solver_version: str = "network-flow-baseline-v1"

"""Strict v1 contracts for dataset manifests and published forecast artifacts.

This module is a repository-level reference validator. It is intentionally not
imported by the online worker yet: the publication task will decide how this
contract is packaged at the producer/consumer boundary.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.calendar_v1 import forecast_buckets

CalendarVersion = Literal["moscow-midnight.v1"]
SchemaVersion = Literal["forecast.v1"]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$")]
FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ForecastHorizon(StrEnum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class BucketGranularity(StrEnum):
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class ForecastTarget(StrEnum):
    SYNTHETIC_BOARDINGS = "synthetic_boardings"
    VALIDATION_COUNT = "validation_count"
    BOARDING_COUNT = "boarding_count"
    ONBOARD_LOAD = "onboard_load"


class AggregationUnit(StrEnum):
    EVENT_COUNT = "event_count"
    PASSENGERS = "passengers"
    VEHICLES = "vehicles"


class DatasetManifest(ContractModel):
    schema_version: SchemaVersion
    dataset_id: Identifier
    source_version: Identifier
    source_hash: Annotated[str, Field(min_length=16, max_length=128, pattern=r"^[a-fA-F0-9]+$")]
    date_from: datetime
    date_to: datetime
    timezone: Literal["Europe/Moscow"] = "Europe/Moscow"
    feature_version: Identifier
    target: ForecastTarget
    unit: AggregationUnit
    entity_version: Identifier
    calendar_version: CalendarVersion = "moscow-midnight.v1"
    availability_policy: Literal["event-and-availability.v1"]
    synthetic: bool

    @model_validator(mode="after")
    def validate_manifest(self) -> "DatasetManifest":
        _require_aware(self.date_from, "date_from")
        _require_aware(self.date_to, "date_to")
        if self.date_to.astimezone(UTC) <= self.date_from.astimezone(UTC):
            raise ValueError("date_to must be after date_from")
        if self.synthetic != (self.target == ForecastTarget.SYNTHETIC_BOARDINGS):
            raise ValueError("dataset synthetic flag must agree with synthetic_boardings target")
        if self.target == ForecastTarget.ONBOARD_LOAD and self.unit != AggregationUnit.PASSENGERS:
            raise ValueError("onboard_load requires passengers unit")
        expected_unit = {
            ForecastTarget.SYNTHETIC_BOARDINGS: AggregationUnit.EVENT_COUNT,
            ForecastTarget.VALIDATION_COUNT: AggregationUnit.EVENT_COUNT,
            ForecastTarget.BOARDING_COUNT: AggregationUnit.PASSENGERS,
            ForecastTarget.ONBOARD_LOAD: AggregationUnit.PASSENGERS,
        }[self.target]
        if self.unit != expected_unit:
            raise ValueError(f"{self.target.value} requires {expected_unit.value} unit")
        return self


class ForecastPoint(ContractModel):
    route_id: Identifier
    direction_id: Identifier
    stop_id: Identifier
    bucket_start: datetime
    bucket_end: datetime
    predicted: FiniteNonNegative
    lower_bound: FiniteNonNegative | None = None
    upper_bound: FiniteNonNegative | None = None

    @model_validator(mode="after")
    def validate_point(self) -> "ForecastPoint":
        _require_aware(self.bucket_start, "bucket_start")
        _require_aware(self.bucket_end, "bucket_end")
        if self.bucket_end.astimezone(UTC) <= self.bucket_start.astimezone(UTC):
            raise ValueError("bucket_end must be after bucket_start")
        bounds = (self.lower_bound, self.upper_bound)
        if any(bound is None for bound in bounds) and any(bound is not None for bound in bounds):
            raise ValueError("lower_bound and upper_bound must be both present or absent")
        if self.lower_bound is not None and self.upper_bound is not None:
            if not self.lower_bound <= self.predicted <= self.upper_bound:
                raise ValueError("predicted must be inside uncertainty bounds")
        return self


class ForecastArtifact(ContractModel):
    schema_version: SchemaVersion
    run_id: Identifier
    dataset_id: Identifier
    target: ForecastTarget
    unit: AggregationUnit
    synthetic: bool
    horizon: ForecastHorizon
    bucket_granularity: BucketGranularity
    forecast_origin: datetime
    generated_at: datetime
    data_cutoff: datetime
    source_version: Identifier
    feature_version: Identifier
    entity_version: Identifier
    calendar_version: CalendarVersion = "moscow-midnight.v1"
    model_version: Identifier
    graph_version: Identifier | None = None
    interval_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None
    interval_method: Identifier | None = None
    points: list[ForecastPoint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_artifact(self) -> "ForecastArtifact":
        for name in ("forecast_origin", "generated_at", "data_cutoff"):
            _require_aware(getattr(self, name), name)
        if self.data_cutoff.astimezone(UTC) > self.forecast_origin.astimezone(UTC):
            raise ValueError("data_cutoff cannot be after forecast_origin")
        if self.generated_at.astimezone(UTC) < self.forecast_origin.astimezone(UTC):
            raise ValueError("generated_at cannot be before forecast_origin")
        if self.synthetic != (self.target == ForecastTarget.SYNTHETIC_BOARDINGS):
            raise ValueError("artifact synthetic flag must agree with synthetic_boardings target")
        expected_unit = {
            ForecastTarget.SYNTHETIC_BOARDINGS: AggregationUnit.EVENT_COUNT,
            ForecastTarget.VALIDATION_COUNT: AggregationUnit.EVENT_COUNT,
            ForecastTarget.BOARDING_COUNT: AggregationUnit.PASSENGERS,
            ForecastTarget.ONBOARD_LOAD: AggregationUnit.PASSENGERS,
        }[self.target]
        if self.unit != expected_unit:
            raise ValueError(f"{self.target.value} requires {expected_unit.value} unit")
        expected_granularity = {
            ForecastHorizon.DAY: BucketGranularity.HOURLY,
            ForecastHorizon.MONTH: BucketGranularity.DAILY,
            ForecastHorizon.YEAR: BucketGranularity.MONTHLY,
        }[self.horizon]
        if self.bucket_granularity != expected_granularity:
            raise ValueError(
                f"{self.horizon.value} horizon requires {expected_granularity.value} buckets"
            )
        if (self.interval_level is None) != (self.interval_method is None):
            raise ValueError("interval_level and interval_method must be both present or absent")
        expected_buckets = {
            (start.astimezone(UTC), end.astimezone(UTC))
            for start, end in forecast_buckets(self.forecast_origin, self.horizon.value)
        }
        by_entity: dict[tuple[str, str, str], set[tuple[datetime, datetime]]] = {}
        for point in self.points:
            key = (point.route_id, point.direction_id, point.stop_id)
            window = (point.bucket_start.astimezone(UTC), point.bucket_end.astimezone(UTC))
            windows = by_entity.setdefault(key, set())
            if window in windows:
                raise ValueError("forecast points must have unique entity/bucket keys")
            if window not in expected_buckets:
                raise ValueError("bucket must align with the calendar horizon")
            windows.add(window)
            if (point.lower_bound is not None) != (self.interval_level is not None):
                raise ValueError("point bounds must agree with interval metadata")
        if any(windows != expected_buckets for windows in by_entity.values()):
            raise ValueError("each represented entity must cover the complete horizon")
        return self


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    try:
        value.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("instant exceeds the supported datetime range") from error


def validate_forecast_json(
    payload: str, *, manifest: DatasetManifest | None = None
) -> ForecastArtifact:
    """Validate an artifact, and dataset identity when supplied by its consumer."""
    artifact = ForecastArtifact.model_validate_json(payload)
    if manifest is not None:
        for field in (
            "dataset_id",
            "source_version",
            "feature_version",
            "entity_version",
            "calendar_version",
            "target",
            "unit",
            "synthetic",
        ):
            if getattr(artifact, field) != getattr(manifest, field):
                raise ValueError(f"artifact {field} does not match manifest")
        if not (
            manifest.date_from.astimezone(UTC)
            <= artifact.data_cutoff.astimezone(UTC)
            <= manifest.date_to.astimezone(UTC)
        ):
            raise ValueError("artifact cutoff must fall within the manifest date range")
    return artifact

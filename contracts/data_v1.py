"""Normalized fixture rows; organizer columns and identity mappings remain unknown."""

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from contracts.forecast_v1 import ContractModel, Identifier

CountTarget = Literal["synthetic_boardings", "validation_count"]
NonNegativeInt = Annotated[int, Field(ge=0)]


def _aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    try:
        value.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("instant exceeds the supported datetime range") from error


class Stop(ContractModel):
    id: Identifier
    name: Annotated[str, Field(min_length=1)]


class RoutePattern(ContractModel):
    route_id: Identifier
    direction_id: Identifier
    stop_ids: list[Identifier] = Field(min_length=1)


class EntityCatalog(ContractModel):
    schema_version: Literal["data.v1"]
    entity_version: Identifier
    routes: list[Identifier] = Field(min_length=1)
    stops: list[Stop] = Field(min_length=1)
    patterns: list[RoutePattern] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        stop_ids = [stop.id for stop in self.stops]
        if len(set(self.routes)) != len(self.routes) or len(set(stop_ids)) != len(stop_ids):
            raise ValueError("duplicate entity ID")
        keys = [(pattern.route_id, pattern.direction_id) for pattern in self.patterns]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate route/direction pattern")
        for pattern in self.patterns:
            if pattern.route_id not in self.routes or not set(pattern.stop_ids) <= set(stop_ids):
                raise ValueError("pattern references unknown route or stop")
        return self

    def validate_location(
        self,
        entity_version: str,
        route_id: str,
        direction_id: str,
        stop_id: str,
        stop_sequence: int | None = None,
    ) -> None:
        """Sequence is a zero-based visit index, including repeated visits to one stop."""
        if entity_version != self.entity_version:
            raise ValueError("entity_version does not match catalog")
        pattern = next(
            (p for p in self.patterns if (p.route_id, p.direction_id) == (route_id, direction_id)),
            None,
        )
        if pattern is None or stop_id not in pattern.stop_ids:
            raise ValueError("unknown route/direction/stop combination")
        if stop_sequence is not None and (
            not 0 <= stop_sequence < len(pattern.stop_ids)
            or pattern.stop_ids[stop_sequence] != stop_id
        ):
            raise ValueError("stop_sequence does not identify this stop visit")


class LocatedRow(ContractModel):
    schema_version: Literal["data.v1"]
    entity_version: Identifier
    route_id: Identifier
    direction_id: Identifier
    stop_id: Identifier

    def validate_entities(self, catalog: EntityCatalog) -> None:
        catalog.validate_location(
            self.entity_version,
            self.route_id,
            self.direction_id,
            self.stop_id,
        )


class EventRow(LocatedRow):
    event_id: Identifier
    source_version: Identifier
    event_at: datetime
    available_at: datetime
    synthetic: bool
    vehicle_id: Identifier
    stop_sequence: NonNegativeInt

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        _aware(self.event_at)
        _aware(self.available_at)
        if self.available_at.astimezone(UTC) < self.event_at.astimezone(UTC):
            raise ValueError("available_at cannot precede event_at")
        return self

    def visible_at(self, cutoff: datetime) -> bool:
        """Half-open event history also excludes information unavailable at cutoff."""
        _aware(cutoff)
        return self.event_at.astimezone(UTC) < cutoff.astimezone(
            UTC
        ) and self.available_at.astimezone(UTC) <= cutoff.astimezone(UTC)

    def validate_entities(self, catalog: EntityCatalog) -> None:
        catalog.validate_location(
            self.entity_version,
            self.route_id,
            self.direction_id,
            self.stop_id,
            self.stop_sequence,
        )


class ValidationEvent(EventRow):
    target: CountTarget
    unit: Literal["event_count"]

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.synthetic != (self.target == "synthetic_boardings"):
            raise ValueError("target must match synthetic provenance")
        return self


class TelemetryEvent(EventRow):
    latitude: Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
    longitude: Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]


class ObservedAggregate(LocatedRow):
    source_version: Identifier
    target: CountTarget
    unit: Literal["event_count"]
    synthetic: bool
    bucket_start: datetime
    bucket_end: datetime
    coverage: Literal["observed", "missing"]
    value: NonNegativeInt | None

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        _aware(self.bucket_start)
        _aware(self.bucket_end)
        if self.bucket_end.astimezone(UTC) <= self.bucket_start.astimezone(UTC):
            raise ValueError("bucket must be a nonempty half-open interval")
        if self.synthetic != (self.target == "synthetic_boardings"):
            raise ValueError("target must match synthetic provenance")
        if (self.coverage == "missing") != (self.value is None):
            raise ValueError("missing requires null; observed requires a nonnegative count")
        return self

"""Scored observations and the slice axes cut from them.

A ``ScoredPoint`` is one forecast already compared with its outcome: the layer never
predicts. Its slice axes are derived, not supplied, so a slice cannot be relabelled to
avoid a verdict. ``None`` is the only representation of missing, as in ``features``:
an absent interval, an absent baseline and an absent capacity are absent, never a
default that would be reported as a measurement.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from tramflow_ml.features.records import (
    GRANULARITY_FOR_HORIZON,
    MOSCOW,
    EntityKey,
    FeatureError,
    Horizon,
    require_aware,
)

DayPart = Literal["morning_peak", "evening_peak", "offpeak"]
SliceAxis = Literal[
    "overall",
    "horizon",
    "fold",
    "route",
    "direction",
    "stop",
    "entity",
    "daypart",
    "route_daypart",
    "entity_daypart",
    "event",
]
SLICE_AXES: frozenset[str] = frozenset(
    {
        "overall",
        "horizon",
        "fold",
        "route",
        "direction",
        "stop",
        "entity",
        "daypart",
        "route_daypart",
        "entity_daypart",
        "event",
    }
)
SINGLE_ORIGIN_AXES: frozenset[str] = frozenset({"fold"})
"""Axes whose slices are one origin by construction, so the fold floor cannot apply.

A per-fold slice exists precisely to expose a model that collapsed on one origin. Gating
it on having two origins would make the axis unable to report the thing it was added for.
"""

MORNING_PEAK_HOURS: frozenset[int] = frozenset({7, 8, 9})
EVENING_PEAK_HOURS: frozenset[int] = frozenset({17, 18, 19})
HOURLY = "hourly"
OVERALL_VALUE = "all"
AXIS_SEPARATOR = "|"


class SliceError(ValueError):
    """Input that makes a slice report unsafe to produce or meaningless to read."""


def _aware(value: datetime, name: str) -> datetime:
    try:
        return require_aware(value, name)
    except FeatureError as error:
        raise SliceError(str(error)) from error


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SliceError(f"{name} must be a number")
    if not math.isfinite(value):
        raise SliceError(f"{name} must be finite")


def _identifier(value: str, name: str) -> None:
    if not value:
        raise SliceError(f"{name} must not be empty")
    if AXIS_SEPARATOR in value:
        raise SliceError(f"{name} must not contain {AXIS_SEPARATOR!r}: it joins slice values")


@dataclass(frozen=True, slots=True, order=True)
class SliceKey:
    """One group of points, named by the axis it was cut on and the value it carries."""

    axis: SliceAxis
    value: str

    def __str__(self) -> str:
        return f"{self.axis}={self.value}"

    def to_dict(self) -> dict[str, str]:
        return {"axis": self.axis, "value": self.value}


@dataclass(frozen=True, slots=True)
class IntervalBounds:
    """A central prediction interval and the nominal level it claims."""

    lower: float
    upper: float
    level: float
    method: str

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "level"):
            _finite(getattr(self, name), f"interval {name}")
        if self.lower < 0 or self.upper < self.lower:
            raise SliceError("interval bounds must be non-negative and ordered")
        if not 0 < self.level < 1:
            raise SliceError("interval level must lie strictly between 0 and 1")
        _identifier(self.method, "interval method")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def covers(self, actual: float) -> bool:
        return self.lower <= actual <= self.upper


@dataclass(frozen=True, slots=True)
class ScoredPoint:
    """One forecast beside its outcome, with everything a slice verdict needs."""

    entity: EntityKey | None
    horizon: Horizon
    bucket_start: datetime
    fold_id: str
    target: str
    unit: str
    event: str | None
    actual: float
    predicted: float
    baseline: float | None
    interval: IntervalBounds | None
    capacity: float | None

    def __post_init__(self) -> None:
        _aware(self.bucket_start, "bucket_start")
        if self.horizon not in GRANULARITY_FOR_HORIZON:
            raise SliceError(f"unsupported horizon: {self.horizon}")
        for name in ("fold_id", "target", "unit"):
            _identifier(getattr(self, name), name)
        if self.event is not None:
            _identifier(self.event, "event")
        self._validate_values()
        self._validate_entity()
        if self.interval is not None and not self.interval.covers(self.predicted):
            raise SliceError("the prediction must lie inside its own interval")

    def _validate_entity(self) -> None:
        entity = self.entity
        if entity is None:
            return
        _identifier(entity.route_id, "route_id")
        _identifier(entity.direction_id, "direction_id")
        _identifier(entity.stop_id, "stop_id")

    def _validate_values(self) -> None:
        for name in ("actual", "predicted"):
            value: float = getattr(self, name)
            _finite(value, name)
            if value < 0:
                raise SliceError(f"{name} must be non-negative")
        if self.baseline is not None:
            _finite(self.baseline, "baseline")
            if self.baseline < 0:
                raise SliceError("baseline must be non-negative")
        if self.capacity is not None:
            _finite(self.capacity, "capacity")
            if self.capacity <= 0:
                raise SliceError("a known capacity must be positive; unknown is None")

    @property
    def daypart(self) -> DayPart | None:
        """Defined only where the bucket is an hour; a month has no evening peak."""
        if GRANULARITY_FOR_HORIZON[self.horizon] != HOURLY:
            return None
        hour = self.bucket_start.astimezone(MOSCOW).hour
        if hour in MORNING_PEAK_HOURS:
            return "morning_peak"
        if hour in EVENING_PEAK_HOURS:
            return "evening_peak"
        return "offpeak"

    @property
    def sort_key(self) -> tuple[str, str, str, str, str]:
        """Canonical order, so the summed floats do not depend on the caller's order."""
        entity = self.entity
        return (
            self.horizon,
            "" if entity is None else entity.route_id,
            "" if entity is None else f"{entity.direction_id}{AXIS_SEPARATOR}{entity.stop_id}",
            self.bucket_start.astimezone(MOSCOW).isoformat(),
            self.fold_id,
        )

    @property
    def keys(self) -> tuple[SliceKey, ...]:
        keys = [
            SliceKey("overall", OVERALL_VALUE),
            SliceKey("horizon", self.horizon),
            SliceKey("fold", self.fold_id),
        ]
        if self.event is not None:
            keys.append(SliceKey("event", self.event))
        daypart = self.daypart
        if daypart is not None:
            keys.append(SliceKey("daypart", daypart))
        keys.extend(self._entity_keys(daypart))
        return tuple(sorted(keys))

    def _entity_keys(self, daypart: DayPart | None) -> list[SliceKey]:
        entity = self.entity
        if entity is None:
            return []
        route, direction, stop = entity.route_id, entity.direction_id, entity.stop_id
        keys = [
            SliceKey("route", route),
            SliceKey("direction", AXIS_SEPARATOR.join((route, direction))),
            SliceKey("stop", stop),
            SliceKey("entity", AXIS_SEPARATOR.join((route, direction, stop))),
        ]
        if daypart is not None:
            keys.append(SliceKey("route_daypart", AXIS_SEPARATOR.join((route, daypart))))
            keys.append(
                SliceKey("entity_daypart", AXIS_SEPARATOR.join((route, direction, stop, daypart)))
            )
        return keys

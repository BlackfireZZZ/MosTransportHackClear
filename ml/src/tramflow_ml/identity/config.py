"""Per-source clocks and matching thresholds; every tolerance is configuration, not code."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tramflow_ml.identity._payload import (
    optional_int,
    optional_mapping,
    require_float,
    require_mapping,
)
from tramflow_ml.identity.types import IdentityError

MOSCOW = ZoneInfo("Europe/Moscow")
DEFAULT_GEO_TOLERANCE_METRES = 50.0
DEFAULT_GPS_STALENESS_SECONDS = 60.0


class ConfigError(IdentityError):
    """Alignment configuration is malformed."""


@dataclass(frozen=True)
class SourceClock:
    """Naive source timestamps are read in ``timezone``; ``offset_seconds`` is added after."""

    timezone: str = "Europe/Moscow"
    offset_seconds: int = 0
    availability_lag_seconds: int | None = None

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ConfigError(f"unknown timezone {self.timezone!r}") from error
        if self.availability_lag_seconds is not None and self.availability_lag_seconds < 0:
            raise ConfigError("availability_lag_seconds must be non-negative")

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], path: str) -> Self:
        timezone = payload.get("timezone", "Europe/Moscow")
        if not isinstance(timezone, str) or not timezone:
            raise ConfigError(f"{path}.timezone must be a non-empty string")
        lag = payload.get("availability_lag_seconds")
        return cls(
            timezone=timezone,
            offset_seconds=optional_int(payload, "offset_seconds", path, 0),
            availability_lag_seconds=(
                None if lag is None else optional_int(payload, "availability_lag_seconds", path, 0)
            ),
        )


@dataclass(frozen=True)
class AlignmentConfig:
    clocks: Mapping[str, SourceClock] = field(default_factory=dict)
    geo_tolerance_metres: float = DEFAULT_GEO_TOLERANCE_METRES
    gps_staleness_seconds: float = DEFAULT_GPS_STALENESS_SECONDS
    stop_sequence_base: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.geo_tolerance_metres) or self.geo_tolerance_metres <= 0:
            raise ConfigError("geo_tolerance_metres must be a positive finite number")
        if not math.isfinite(self.gps_staleness_seconds) or self.gps_staleness_seconds < 0:
            raise ConfigError("gps_staleness_seconds must be a non-negative finite number")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        root = require_mapping(payload, "config")
        clocks = {
            source_id: SourceClock.from_dict(
                require_mapping(item, f"config.clocks[{source_id!r}]"),
                f"config.clocks[{source_id!r}]",
            )
            for source_id, item in optional_mapping(root, "clocks", "config").items()
        }
        defaults: dict[str, object] = {
            "geo_tolerance_metres": DEFAULT_GEO_TOLERANCE_METRES,
            "gps_staleness_seconds": DEFAULT_GPS_STALENESS_SECONDS,
        }
        merged = {**defaults, **root}
        return cls(
            clocks=clocks,
            geo_tolerance_metres=require_float(merged, "geo_tolerance_metres", "config"),
            gps_staleness_seconds=require_float(merged, "gps_staleness_seconds", "config"),
            stop_sequence_base=optional_int(root, "stop_sequence_base", "config", 0),
        )

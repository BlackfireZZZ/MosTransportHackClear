"""Normalization ``normalization.v1``: adapted rows into the ``data.v1`` event shape."""

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tramflow_ml.ingestion.records import (
    MOSCOW,
    ColumnAdapter,
    IngestionError,
    Rejection,
    StreamName,
)

SCHEMA_VERSION = "data.v1"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]+$")
IDENTIFIER_MAX_LENGTH = 128
TARGETS = frozenset({"synthetic_boardings", "validation_count"})
SYNTHETIC_TARGET = "synthetic_boardings"
UNIT = "event_count"
COORDINATE_LIMITS: Mapping[str, float] = {"latitude": 90.0, "longitude": 180.0}
IDENTIFIER_FIELDS = frozenset(
    {
        "entity_version",
        "source_version",
        "event_id",
        "route_id",
        "direction_id",
        "stop_id",
        "vehicle_id",
    }
)
TIMESTAMP_FIELDS = ("event_at", "available_at")
COMMON_FIELDS: tuple[str, ...] = (
    "schema_version",
    "entity_version",
    "source_version",
    "event_id",
    "route_id",
    "direction_id",
    "stop_id",
    "stop_sequence",
    "vehicle_id",
    "event_at",
    "available_at",
    "synthetic",
)
STREAM_FIELDS: Mapping[StreamName, tuple[str, ...]] = {
    "validations": (*COMMON_FIELDS, "target", "unit"),
    "telemetry": (*COMMON_FIELDS, "latitude", "longitude"),
}
_MISSING = object()
_TRUE_STRINGS = frozenset({"true", "1"})
_FALSE_STRINGS = frozenset({"false", "0"})


class CatalogIndex:
    """Route/direction patterns from ``entities.json``; small, fully in memory."""

    def __init__(
        self, entity_version: str, patterns: Mapping[tuple[str, str], tuple[str, ...]]
    ) -> None:
        self.entity_version = entity_version
        self.patterns = dict(patterns)

    @classmethod
    def load(cls, path: Path) -> "CatalogIndex":
        try:
            payload = json.loads(path.read_bytes())
            entity_version = payload["entity_version"]
            patterns = {
                (pattern["route_id"], pattern["direction_id"]): tuple(pattern["stop_ids"])
                for pattern in payload["patterns"]
            }
        except (ValueError, KeyError, TypeError) as error:
            raise IngestionError(f"{path.name}: not a data.v1 entity catalog") from error
        well_typed = isinstance(entity_version, str) and all(
            isinstance(stop, str) for stops in patterns.values() for stop in stops
        )
        if not well_typed or not patterns:
            raise IngestionError(f"{path.name}: catalog needs string ids and patterns")
        return cls(entity_version, patterns)

    def locate(
        self, entity_version: str, route_id: str, direction_id: str, stop_id: str, sequence: int
    ) -> str | None:
        """Return a rejection detail, or ``None`` when the location is in the catalog."""
        if entity_version != self.entity_version:
            return "entity_version does not match catalog"
        stops = self.patterns.get((route_id, direction_id))
        if stops is None:
            return "unknown route/direction"
        if stop_id not in stops:
            return "unknown stop for route/direction"
        if sequence >= len(stops) or stops[sequence] != stop_id:
            return "stop_sequence does not identify this stop visit"
        return None


def _identifier(name: str, raw: object) -> str | Rejection:
    if not isinstance(raw, str):
        return Rejection("invalid_type", f"{name} must be a string")
    if len(raw) > IDENTIFIER_MAX_LENGTH or not IDENTIFIER_PATTERN.match(raw):
        return Rejection("invalid_value", f"{name} is not a valid identifier")
    return raw


def _literal(name: str, raw: object, allowed: frozenset[str]) -> str | Rejection:
    if not isinstance(raw, str):
        return Rejection("invalid_type", f"{name} must be a string")
    if raw not in allowed:
        return Rejection("invalid_value", f"{name} must be one of {sorted(allowed)}")
    return raw


def _sequence(name: str, raw: object, coerce: bool) -> int | Rejection:
    if coerce and isinstance(raw, str):
        raw = int(raw) if raw.isdigit() else raw
    if type(raw) is not int:
        return Rejection("invalid_type", f"{name} must be an integer")
    if raw < 0:
        return Rejection("invalid_value", f"{name} must be >= 0")
    return raw


def _boolean(name: str, raw: object, coerce: bool) -> bool | Rejection:
    if coerce and isinstance(raw, str):
        lowered = raw.lower()
        raw = True if lowered in _TRUE_STRINGS else False if lowered in _FALSE_STRINGS else raw
    if type(raw) is not bool:
        return Rejection("invalid_type", f"{name} must be a boolean")
    return raw


def _coordinate(name: str, raw: object, coerce: bool) -> float | Rejection:
    if coerce and isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError:
            return Rejection("invalid_type", f"{name} must be a number")
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        return Rejection("invalid_type", f"{name} must be a number")
    value = float(raw)
    limit = COORDINATE_LIMITS[name]
    if not math.isfinite(value) or not -limit <= value <= limit:
        return Rejection("invalid_value", f"{name} must be finite within ±{limit:g}")
    return value


class Normalizer:
    def __init__(
        self,
        adapter: ColumnAdapter,
        catalog: CatalogIndex,
        stream: StreamName,
        coerce_strings: bool,
    ) -> None:
        self.adapter = adapter
        self.catalog = catalog
        self.stream = stream
        self.fields = STREAM_FIELDS[stream]
        self.coerce_strings = coerce_strings
        try:
            self.zone = ZoneInfo(adapter.assume_timezone) if adapter.assume_timezone else None
        except ZoneInfoNotFoundError as error:
            raise IngestionError(f"unknown assume_timezone {adapter.assume_timezone!r}") from error

    def normalize(self, source: Mapping[str, object]) -> dict[str, object] | Rejection:
        values: dict[str, object] = {}
        for name in self.fields:
            raw = self._lookup(source, name)
            if raw is _MISSING:
                return Rejection("missing_field", name)
            checked = self._check(name, raw)
            if isinstance(checked, Rejection):
                return checked
            values[name] = checked
        return self._relations(values)

    def _lookup(self, source: Mapping[str, object], name: str) -> object:
        column = self.adapter.source_column(name)
        if column in source:
            return source[column]
        return self.adapter.constants.get(name, _MISSING)

    def _check(self, name: str, raw: object) -> object:
        if name in IDENTIFIER_FIELDS:
            return _identifier(name, raw)
        if name == "schema_version":
            return _literal(name, raw, frozenset({SCHEMA_VERSION}))
        if name == "target":
            return _literal(name, raw, TARGETS)
        if name == "unit":
            return _literal(name, raw, frozenset({UNIT}))
        if name == "stop_sequence":
            return _sequence(name, raw, self.coerce_strings)
        if name == "synthetic":
            return _boolean(name, raw, self.coerce_strings)
        if name in TIMESTAMP_FIELDS:
            return self._timestamp(name, raw)
        return _coordinate(name, raw, self.coerce_strings)

    def _timestamp(self, name: str, raw: object) -> datetime | Rejection:
        if not isinstance(raw, str):
            return Rejection("invalid_type", f"{name} must be a string")
        try:
            parsed = (
                datetime.strptime(raw, self.adapter.timestamp_format)
                if self.adapter.timestamp_format
                else datetime.fromisoformat(raw)
            )
        except ValueError as error:
            return Rejection("invalid_timestamp", f"{name}: {error}")
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            if self.zone is None:
                return Rejection("invalid_timestamp", f"{name} has no timezone")
            parsed = parsed.replace(tzinfo=self.zone)
        try:
            return parsed.astimezone(MOSCOW)
        except OverflowError:
            return Rejection("invalid_timestamp", f"{name} exceeds the supported range")

    def _relations(self, values: dict[str, object]) -> dict[str, object] | Rejection:
        event_at, available_at = values["event_at"], values["available_at"]
        assert isinstance(event_at, datetime) and isinstance(available_at, datetime)
        if available_at < event_at:
            return Rejection("availability_before_event", "available_at precedes event_at")
        if self.stream == "validations" and values["synthetic"] != (
            values["target"] == SYNTHETIC_TARGET
        ):
            return Rejection("invalid_value", "target must match synthetic provenance")
        sequence = values["stop_sequence"]
        assert isinstance(sequence, int)
        detail = self.catalog.locate(
            str(values["entity_version"]),
            str(values["route_id"]),
            str(values["direction_id"]),
            str(values["stop_id"]),
            sequence,
        )
        if detail is not None:
            return Rejection("unknown_entity", detail)
        return {
            **values,
            "event_at": event_at.isoformat(),
            "available_at": available_at.isoformat(),
        }

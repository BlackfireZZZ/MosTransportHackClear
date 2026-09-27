"""Canonical entities read from the entities.json shape; ids are the only identity key."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from tramflow_ml.identity._payload import require_mapping, require_sequence, require_str
from tramflow_ml.identity.types import IdentityError


class CatalogError(IdentityError):
    """The catalog payload is malformed or internally inconsistent."""


@dataclass(frozen=True)
class CanonicalStop:
    id: str
    name: str


@dataclass(frozen=True)
class CanonicalPattern:
    route_id: str
    direction_id: str
    stop_ids: tuple[str, ...]

    def visits(self, stop_id: str) -> tuple[int, ...]:
        """Zero-based visit indexes of one stop; a loop pattern may return several."""
        return tuple(index for index, visited in enumerate(self.stop_ids) if visited == stop_id)


@dataclass(frozen=True)
class CanonicalCatalog:
    entity_version: str
    routes: frozenset[str]
    stops: Mapping[str, CanonicalStop]
    patterns: Mapping[tuple[str, str], CanonicalPattern]

    @property
    def direction_ids(self) -> frozenset[str]:
        return frozenset(direction for _, direction in self.patterns)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        root = require_mapping(payload, "catalog")
        if root.get("schema_version") != "data.v1":
            raise CatalogError("catalog.schema_version must be data.v1")
        routes = _string_list(root, "routes")
        stops = _stops(root)
        patterns = _patterns(root)
        if len(set(routes)) != len(routes):
            raise CatalogError("catalog.routes contains a duplicate id")
        for key, pattern in patterns.items():
            if pattern.route_id not in routes:
                raise CatalogError(f"pattern {key} references unknown route")
            unknown = sorted(set(pattern.stop_ids) - stops.keys())
            if unknown:
                raise CatalogError(f"pattern {key} references unknown stops {unknown}")
        return cls(
            require_str(root, "entity_version", "catalog"), frozenset(routes), stops, patterns
        )


def _string_list(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    items = require_sequence(payload.get(key), f"catalog.{key}")
    if not items or any(not isinstance(item, str) or not item for item in items):
        raise CatalogError(f"catalog.{key} must be a non-empty array of ids")
    return tuple(item for item in items if isinstance(item, str))


def _stops(payload: Mapping[str, object]) -> dict[str, CanonicalStop]:
    stops: dict[str, CanonicalStop] = {}
    for index, item in enumerate(require_sequence(payload.get("stops"), "catalog.stops")):
        entry = require_mapping(item, f"catalog.stops[{index}]")
        stop = CanonicalStop(
            require_str(entry, "id", f"catalog.stops[{index}]"),
            require_str(entry, "name", f"catalog.stops[{index}]"),
        )
        if stop.id in stops:
            raise CatalogError(f"catalog.stops repeats id {stop.id}")
        stops[stop.id] = stop
    if not stops:
        raise CatalogError("catalog.stops must not be empty")
    return stops


def _patterns(payload: Mapping[str, object]) -> dict[tuple[str, str], CanonicalPattern]:
    patterns: dict[tuple[str, str], CanonicalPattern] = {}
    for index, item in enumerate(require_sequence(payload.get("patterns"), "catalog.patterns")):
        path = f"catalog.patterns[{index}]"
        entry = require_mapping(item, path)
        stop_ids = require_sequence(entry.get("stop_ids"), f"{path}.stop_ids")
        if not stop_ids or any(not isinstance(stop, str) or not stop for stop in stop_ids):
            raise CatalogError(f"{path}.stop_ids must be a non-empty array of ids")
        pattern = CanonicalPattern(
            require_str(entry, "route_id", path),
            require_str(entry, "direction_id", path),
            tuple(stop for stop in stop_ids if isinstance(stop, str)),
        )
        key = (pattern.route_id, pattern.direction_id)
        if key in patterns:
            raise CatalogError(f"catalog.patterns repeats route/direction {key}")
        patterns[key] = pattern
    if not patterns:
        raise CatalogError("catalog.patterns must not be empty")
    return patterns

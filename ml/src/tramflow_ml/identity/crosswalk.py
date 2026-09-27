"""Versioned source-to-canonical tables; every canonical id is checked at load time."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from tramflow_ml.identity._payload import (
    optional_mapping,
    optional_sequence,
    optional_str,
    require_float,
    require_mapping,
    require_str,
    str_mapping,
)
from tramflow_ml.identity.catalog import CanonicalCatalog
from tramflow_ml.identity.geo import GeoPoint
from tramflow_ml.identity.types import IdentityError


class CrosswalkError(IdentityError):
    """The crosswalk names entities or intervals the catalog cannot vouch for."""


@dataclass(frozen=True)
class VehicleAssignment:
    """Half-open [valid_from, valid_to) interval; an open end means still assigned."""

    vehicle_id: str
    route_id: str
    valid_from: datetime
    valid_to: datetime | None

    def covers(self, instant: datetime) -> bool:
        moment = instant.astimezone(UTC)
        if moment < self.valid_from.astimezone(UTC):
            return False
        return self.valid_to is None or moment < self.valid_to.astimezone(UTC)


@dataclass(frozen=True)
class Crosswalk:
    crosswalk_version: str
    entity_version: str
    routes: Mapping[str, str]
    directions: Mapping[str, str]
    stops: Mapping[str, str]
    stop_names: Mapping[tuple[str, str, str], str]
    stop_positions: Mapping[str, GeoPoint]
    vehicles: Mapping[str, tuple[VehicleAssignment, ...]]

    def assignment_at(self, vehicle_id: str, instant: datetime) -> VehicleAssignment | None:
        return next(
            (item for item in self.vehicles.get(vehicle_id, ()) if item.covers(instant)), None
        )


def load_crosswalk(payload: Mapping[str, object], catalog: CanonicalCatalog) -> Crosswalk:
    root = require_mapping(payload, "crosswalk")
    entity_version = require_str(root, "entity_version", "crosswalk")
    if entity_version != catalog.entity_version:
        raise CrosswalkError(
            f"crosswalk.entity_version {entity_version!r} != catalog {catalog.entity_version!r}"
        )
    routes = str_mapping(root, "routes", "crosswalk")
    directions = str_mapping(root, "directions", "crosswalk")
    stops = str_mapping(root, "stops", "crosswalk")
    _check_targets(routes.values(), catalog.routes, "routes")
    _check_targets(directions.values(), catalog.direction_ids, "directions")
    _check_targets(stops.values(), catalog.stops.keys(), "stops")
    return Crosswalk(
        crosswalk_version=require_str(root, "crosswalk_version", "crosswalk"),
        entity_version=entity_version,
        routes=MappingProxyType(routes),
        directions=MappingProxyType(directions),
        stops=MappingProxyType(stops),
        stop_names=MappingProxyType(
            _stop_names(optional_sequence(root, "stop_names", "crosswalk"), catalog)
        ),
        stop_positions=MappingProxyType(
            _positions(optional_mapping(root, "stop_positions", "crosswalk"), catalog)
        ),
        vehicles=MappingProxyType(
            _vehicles(optional_sequence(root, "vehicles", "crosswalk"), catalog)
        ),
    )


def identity_crosswalk(catalog: CanonicalCatalog, crosswalk_version: str) -> Crosswalk:
    """Map every canonical id to itself for sources already keyed by catalog ids."""
    return Crosswalk(
        crosswalk_version=crosswalk_version,
        entity_version=catalog.entity_version,
        routes=MappingProxyType({route: route for route in sorted(catalog.routes)}),
        directions=MappingProxyType(
            {direction: direction for direction in sorted(catalog.direction_ids)}
        ),
        stops=MappingProxyType({stop: stop for stop in sorted(catalog.stops)}),
        stop_names=MappingProxyType({}),
        stop_positions=MappingProxyType({}),
        vehicles=MappingProxyType({}),
    )


def _check_targets(targets: Iterable[str], known: Iterable[str], table: str) -> None:
    unknown = sorted(set(targets) - set(known))
    if unknown:
        raise CrosswalkError(f"crosswalk.{table} maps to unknown canonical ids {unknown}")


def _stop_names(
    entries: Sequence[object], catalog: CanonicalCatalog
) -> dict[tuple[str, str, str], str]:
    table: dict[tuple[str, str, str], str] = {}
    for index, item in enumerate(entries):
        path = f"crosswalk.stop_names[{index}]"
        entry = require_mapping(item, path)
        key = (
            require_str(entry, "name", path),
            require_str(entry, "route_id", path),
            require_str(entry, "direction_id", path),
        )
        stop_id = require_str(entry, "stop_id", path)
        pattern = catalog.patterns.get(key[1:])
        if pattern is None or stop_id not in pattern.stop_ids:
            raise CrosswalkError(f"{path} names a stop that is not on pattern {key[1:]}")
        if key in table:
            raise CrosswalkError(f"{path} repeats name/route/direction {key}")
        table[key] = stop_id
    return table


def _positions(payload: Mapping[str, object], catalog: CanonicalCatalog) -> dict[str, GeoPoint]:
    _check_targets(payload.keys(), catalog.stops.keys(), "stop_positions")
    positions: dict[str, GeoPoint] = {}
    for stop_id, item in payload.items():
        path = f"crosswalk.stop_positions[{stop_id!r}]"
        entry = require_mapping(item, path)
        try:
            positions[stop_id] = GeoPoint(
                require_float(entry, "latitude", path), require_float(entry, "longitude", path)
            )
        except ValueError as error:
            raise CrosswalkError(f"{path}: {error}") from error
    return positions


def _instant(entry: Mapping[str, object], key: str, path: str) -> datetime | None:
    raw = optional_str(entry, key, path)
    if raw is None:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise CrosswalkError(f"{path}.{key} is not an ISO-8601 timestamp") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise CrosswalkError(f"{path}.{key} must be timezone-aware")
    return value


def _assignment(item: object, index: int, catalog: CanonicalCatalog) -> VehicleAssignment:
    path = f"crosswalk.vehicles[{index}]"
    entry = require_mapping(item, path)
    route_id = require_str(entry, "route_id", path)
    if route_id not in catalog.routes:
        raise CrosswalkError(f"{path}.route_id {route_id!r} is not a canonical route")
    valid_from = _instant(entry, "valid_from", path)
    if valid_from is None:
        raise CrosswalkError(f"{path}.valid_from is required")
    valid_to = _instant(entry, "valid_to", path)
    if valid_to is not None and valid_to.astimezone(UTC) <= valid_from.astimezone(UTC):
        raise CrosswalkError(f"{path} interval must have valid_from before valid_to")
    return VehicleAssignment(require_str(entry, "vehicle_id", path), route_id, valid_from, valid_to)


def _vehicles(
    entries: Sequence[object], catalog: CanonicalCatalog
) -> dict[str, tuple[VehicleAssignment, ...]]:
    grouped: dict[str, list[VehicleAssignment]] = {}
    for index, item in enumerate(entries):
        assignment = _assignment(item, index, catalog)
        grouped.setdefault(assignment.vehicle_id, []).append(assignment)
    table: dict[str, tuple[VehicleAssignment, ...]] = {}
    for vehicle_id, items in sorted(grouped.items()):
        ordered = tuple(sorted(items, key=lambda item: item.valid_from.astimezone(UTC)))
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if earlier.valid_to is None or earlier.valid_to > later.valid_from:
                raise CrosswalkError(f"crosswalk.vehicles overlap for vehicle {vehicle_id!r}")
        table[vehicle_id] = ordered
    return table

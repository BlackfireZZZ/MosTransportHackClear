"""Hand-built catalog and crosswalk shared by the identity tests; no test lives here."""

import copy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tramflow_ml.identity import (
    AlignmentConfig,
    CanonicalCatalog,
    SourceClock,
    SourceEvent,
    load_crosswalk,
)

MOSCOW = ZoneInfo("Europe/Moscow")
ENTITY_VERSION = "entities.test"
CROSSWALK_VERSION = "crosswalk.test.1"
SQUARE = "Площадь"
PARK = "Парк"
DEPOT = "Депо"
EVENT_AT = datetime(2024, 3, 1, 8, 0, tzinfo=MOSCOW)

CATALOG = {
    "schema_version": "data.v1",
    "entity_version": ENTITY_VERSION,
    "routes": ["route:1", "route:2"],
    "stops": [
        {"id": "stop:A1", "name": SQUARE},
        {"id": "stop:A2", "name": SQUARE},
        {"id": "stop:B", "name": PARK},
        {"id": "stop:C", "name": DEPOT},
    ],
    "patterns": [
        {
            "route_id": "route:1",
            "direction_id": "dir:0",
            "stop_ids": ["stop:A1", "stop:B", "stop:A2", "stop:B"],
        },
        {
            "route_id": "route:1",
            "direction_id": "dir:1",
            "stop_ids": ["stop:B", "stop:A2", "stop:B", "stop:A1"],
        },
        {
            "route_id": "route:2",
            "direction_id": "dir:0",
            "stop_ids": ["stop:C", "stop:B", "stop:C", "stop:B"],
        },
    ],
}

CROSSWALK = {
    "crosswalk_version": CROSSWALK_VERSION,
    "entity_version": ENTITY_VERSION,
    "routes": {"1": "route:1", "2": "route:2"},
    "directions": {"A": "dir:0", "B": "dir:1"},
    "stops": {"S-A1": "stop:A1", "S-A2": "stop:A2", "S-B": "stop:B", "S-C": "stop:C"},
    "stop_names": [
        {"name": SQUARE, "route_id": "route:1", "direction_id": "dir:1", "stop_id": "stop:A2"}
    ],
    "stop_positions": {
        "stop:A1": {"latitude": 55.75, "longitude": 37.60},
        "stop:A2": {"latitude": 55.76, "longitude": 37.60},
        "stop:B": {"latitude": 55.77, "longitude": 37.60},
        "stop:C": {"latitude": 55.78, "longitude": 37.60},
    },
    "vehicles": [
        {
            "vehicle_id": "V1",
            "route_id": "route:1",
            "valid_from": "2024-03-01T00:00:00+03:00",
            "valid_to": "2024-03-01T12:00:00+03:00",
        },
        {
            "vehicle_id": "V1",
            "route_id": "route:2",
            "valid_from": "2024-03-01T12:00:00+03:00",
            "valid_to": None,
        },
    ],
}


def catalog():
    return CanonicalCatalog.from_dict(copy.deepcopy(CATALOG))


def crosswalk(**overrides):
    payload = {**copy.deepcopy(CROSSWALK), **overrides}
    return load_crosswalk(payload, catalog())


def config(**overrides):
    clocks = {"validations": SourceClock(), "telemetry": SourceClock()}
    return AlignmentConfig(clocks=clocks, **overrides)


def event(**overrides):
    fields = {
        "event_id": "e1",
        "source_id": "validations",
        "event_at": EVENT_AT,
        "available_at": EVENT_AT + timedelta(minutes=1),
        "route_id": "1",
        "direction_id": "A",
    }
    return SourceEvent(**{**fields, **overrides})

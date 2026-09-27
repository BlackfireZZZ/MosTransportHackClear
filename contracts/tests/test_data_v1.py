from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from contracts.data_v1 import EntityCatalog, ObservedAggregate, TelemetryEvent, ValidationEvent

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": "data.v1",
        "entity_version": "entities.1",
        "routes": ["r1"],
        "stops": [{"id": "a", "name": "Same name"}, {"id": "b", "name": "Same name"}],
        "patterns": [{"route_id": "r1", "direction_id": "out", "stop_ids": ["a", "b", "a"]}],
    }


def event_payload() -> dict[str, Any]:
    return {
        "schema_version": "data.v1",
        "entity_version": "entities.1",
        "route_id": "r1",
        "direction_id": "out",
        "stop_id": "a",
        "event_id": "event1",
        "source_version": "src1",
        "event_at": NOW,
        "available_at": NOW + timedelta(minutes=5),
        "synthetic": True,
        "vehicle_id": "vehicle1",
        "stop_sequence": 2,
        "target": "synthetic_boardings",
        "unit": "event_count",
    }


def aggregate_payload() -> dict[str, Any]:
    return {
        "schema_version": "data.v1",
        "entity_version": "entities.1",
        "route_id": "r1",
        "direction_id": "out",
        "stop_id": "a",
        "source_version": "src1",
        "target": "synthetic_boardings",
        "unit": "event_count",
        "synthetic": True,
        "bucket_start": NOW,
        "bucket_end": NOW + timedelta(hours=1),
        "coverage": "observed",
        "value": 0,
    }


def test_duplicate_names_and_repeated_visits_are_distinct() -> None:
    catalog = EntityCatalog.model_validate(catalog_payload())
    for sequence in (0, 2):
        row = ValidationEvent.model_validate(event_payload() | {"stop_sequence": sequence})
        row.validate_entities(catalog)
    with pytest.raises(ValueError, match="stop_sequence"):
        ValidationEvent.model_validate(event_payload() | {"stop_sequence": 1}).validate_entities(
            catalog
        )


@pytest.mark.parametrize(
    "change",
    [
        {"routes": ["r1", "r1"]},
        {"stops": [{"id": "a", "name": "A"}, {"id": "a", "name": "B"}]},
        {"patterns": [{"route_id": "unknown", "direction_id": "out", "stop_ids": ["a"]}]},
        {"patterns": [{"route_id": "r1", "direction_id": "out", "stop_ids": ["unknown"]}]},
        {"patterns": catalog_payload()["patterns"] * 2},
    ],
)
def test_invalid_catalog(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EntityCatalog.model_validate(catalog_payload() | change)


@pytest.mark.parametrize(
    "change",
    [
        {"entity_version": "other"},
        {"route_id": "unknown"},
        {"direction_id": "in"},
        {"stop_id": "unknown"},
        {"stop_sequence": 3},
    ],
)
def test_invalid_entity_reference(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ValidationEvent.model_validate(event_payload() | change).validate_entities(
            EntityCatalog.model_validate(catalog_payload())
        )


def test_availability_and_event_cutoff_are_independent() -> None:
    row = ValidationEvent.model_validate(event_payload())
    assert not row.visible_at(NOW + timedelta(minutes=4))
    assert row.visible_at(NOW + timedelta(minutes=5))
    immediate = ValidationEvent.model_validate(event_payload() | {"available_at": NOW})
    assert not immediate.visible_at(NOW)
    assert immediate.visible_at(NOW + timedelta(microseconds=1))
    assert immediate.visible_at(NOW.astimezone(timezone(timedelta(hours=3))) + timedelta(seconds=1))
    with pytest.raises(ValueError, match="timezone-aware"):
        row.visible_at(NOW.replace(tzinfo=None))


@pytest.mark.parametrize(
    "change",
    [
        {"event_at": NOW.replace(tzinfo=None)},
        {"available_at": NOW.replace(tzinfo=None)},
        {"available_at": NOW - timedelta(seconds=1)},
        {"synthetic": False},
        {"target": "validation_count"},
        {"target": "onboard_load"},
        {"unit": "passengers"},
        {"stop_sequence": -1},
        {"stop_sequence": True},
        {"passenger_id": "forbidden"},
        {"schema_version": "data.v2"},
    ],
)
def test_invalid_validation_event(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ValidationEvent.model_validate(event_payload() | change)


def test_real_validation_target() -> None:
    row = ValidationEvent.model_validate(
        event_payload() | {"synthetic": False, "target": "validation_count"}
    )
    assert row.unit == "event_count"


@pytest.mark.parametrize(
    "latitude,longitude",
    [
        (91.0, 0.0),
        (-91.0, 0.0),
        (0.0, 181.0),
        (0.0, -181.0),
        (float("nan"), 0.0),
        (0.0, float("inf")),
    ],
)
def test_telemetry_coordinate_bounds(latitude: float, longitude: float) -> None:
    payload = event_payload()
    del payload["target"], payload["unit"]
    with pytest.raises(ValidationError):
        TelemetryEvent.model_validate(payload | {"latitude": latitude, "longitude": longitude})


def test_telemetry_identity_and_availability() -> None:
    payload = event_payload()
    del payload["target"], payload["unit"]
    row = TelemetryEvent.model_validate(payload | {"latitude": 55.75, "longitude": 37.6})
    row.validate_entities(EntityCatalog.model_validate(catalog_payload()))
    assert row.visible_at(NOW + timedelta(minutes=5))
    assert not row.visible_at(NOW)


def test_missing_is_not_zero() -> None:
    zero = ObservedAggregate.model_validate(aggregate_payload())
    missing = ObservedAggregate.model_validate(
        aggregate_payload() | {"coverage": "missing", "value": None}
    )
    assert zero.value == 0
    assert missing.value is None
    zero.validate_entities(EntityCatalog.model_validate(catalog_payload()))


@pytest.mark.parametrize(
    "change",
    [
        {"coverage": "missing"},
        {"value": None},
        {"value": -1},
        {"value": 1.5},
        {"value": True},
        {"bucket_end": NOW},
        {"bucket_start": NOW.replace(tzinfo=None)},
        {"bucket_end": NOW.replace(tzinfo=None)},
        {"target": "onboard_load"},
        {"unit": "passengers"},
        {"synthetic": False},
        {"unexpected": 1},
    ],
)
def test_invalid_observation(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ObservedAggregate.model_validate(aggregate_payload() | change)

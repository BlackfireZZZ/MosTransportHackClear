"""Read the ingestion output back as observations, and enumerate catalog entities."""

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from tramflow_ml.features.records import EntityKey, FeatureError, Observation, require_aware
from tramflow_ml.identity import CanonicalCatalog

REQUIRED_FIELDS = ("route_id", "direction_id", "stop_id", "event_at", "available_at")
COUNT_FIELDS = ("target", "unit")


def observation_from_row(row: Mapping[str, object], where: str) -> Observation:
    """One ``data.v1`` validation line; a defect stops the run instead of being dropped."""
    values = {name: _text(row, name, where) for name in (*REQUIRED_FIELDS, *COUNT_FIELDS)}
    return Observation(
        entity=EntityKey(values["route_id"], values["direction_id"], values["stop_id"]),
        event_at=_timestamp(values["event_at"], "event_at", where),
        available_at=_timestamp(values["available_at"], "available_at", where),
        target=values["target"],
        unit=values["unit"],
    )


def _text(row: Mapping[str, object], name: str, where: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise FeatureError(f"{where}: {name} must be a non-empty string")
    return value


def _timestamp(raw: str, name: str, where: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as error:
        raise FeatureError(f"{where}: {name} is not an ISO-8601 timestamp") from error
    return require_aware(parsed, f"{where}: {name}")


def load_observations(path: Path) -> tuple[Observation, ...]:
    """Read ``validations.jsonl`` from an ingestion run; telemetry has no count target."""
    observations: list[Observation] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{number}"
            try:
                payload = json.loads(line)
            except ValueError as error:
                raise FeatureError(f"{where}: not valid JSON") from error
            if not isinstance(payload, dict):
                raise FeatureError(f"{where}: line must be a JSON object")
            observations.append(observation_from_row(payload, where))
    return tuple(observations)


def entity_keys(catalog: CanonicalCatalog) -> tuple[EntityKey, ...]:
    """Every route/direction/stop the catalog can produce, in a stable order."""
    keys = {
        EntityKey(pattern.route_id, pattern.direction_id, stop_id)
        for pattern in catalog.patterns.values()
        for stop_id in pattern.stop_ids
    }
    return tuple(sorted(keys))


def load_entity_keys(path: Path) -> tuple[EntityKey, ...]:
    try:
        payload = json.loads(path.read_bytes())
    except ValueError as error:
        raise FeatureError(f"{path.name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise FeatureError(f"{path.name} must be a JSON object")
    return entity_keys(CanonicalCatalog.from_dict(payload))

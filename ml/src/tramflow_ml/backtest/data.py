"""The inputs one backtest stands on, and the fingerprint that identifies them.

"The same config produces the same manifest" is vacuous without "and the same data", so
the data is fingerprinted too. Timestamps are normalised to UTC before hashing, because
two spellings of one instant are the same datum and must not produce two hashes.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from types import MappingProxyType

from tramflow_ml.backtest.records import BacktestError
from tramflow_ml.features import (
    CapacityRecord,
    CoverageCalendar,
    EntityKey,
    Observation,
    encode,
)

DATA_DIGEST_SCHEMA = "backtest-data.v1"


def _instant(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _observation_payload(observation: Observation) -> dict[str, str]:
    return {
        **observation.entity.to_dict(),
        "event_at": _instant(observation.event_at),
        "available_at": _instant(observation.available_at),
        "target": observation.target,
        "unit": observation.unit,
    }


def _coverage_payload(coverage: CoverageCalendar) -> dict[str, object]:
    return {
        "dates": [day.isoformat() for day in sorted(coverage.dates)],
        "published_at": {
            day.isoformat(): _instant(instant)
            for day, instant in sorted(coverage.published_at.items())
        },
    }


def _capacity_payload(capacities: Mapping[EntityKey, CapacityRecord]) -> list[dict[str, object]]:
    return [
        {
            **entity.to_dict(),
            "value": capacities[entity].value,
            "available_at": _instant(capacities[entity].available_at),
        }
        for entity in sorted(capacities)
    ]


@dataclass(frozen=True, slots=True)
class BacktestData:
    """Entities, observations and the coverage statement every fold is cut from.

    Entities are stored sorted and deduplicated, and capacities are copied, so a caller
    that keeps its own containers cannot change a constructed input set afterwards.
    """

    entities: tuple[EntityKey, ...]
    observations: tuple[Observation, ...]
    coverage: CoverageCalendar
    capacities: Mapping[EntityKey, CapacityRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entities:
            raise BacktestError("a backtest needs at least one entity")
        object.__setattr__(self, "entities", tuple(sorted(set(self.entities))))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "capacities", MappingProxyType(dict(self.capacities)))

    @classmethod
    def of(
        cls,
        entities: Iterable[EntityKey],
        observations: Iterable[Observation],
        coverage: CoverageCalendar,
        capacities: Mapping[EntityKey, CapacityRecord] | None = None,
    ) -> "BacktestData":
        return cls(tuple(entities), tuple(observations), coverage, dict(capacities or {}))

    @property
    def data_hash(self) -> str:
        """Order-independent SHA-256 over the observations, coverage and capacities.

        Observations are hashed as sorted canonical lines rather than in input order: the
        feature layer's result does not depend on the order events arrive in, so neither
        may the fingerprint that claims to identify that result's input.
        """
        digest = sha256()
        digest.update(encode({"schema": DATA_DIGEST_SCHEMA}))
        digest.update(encode({"entities": [entity.to_dict() for entity in self.entities]}))
        for line in sorted(encode(_observation_payload(item)) for item in self.observations):
            digest.update(line)
            digest.update(b"\n")
        digest.update(encode(_coverage_payload(self.coverage)))
        digest.update(encode({"capacities": _capacity_payload(self.capacities)}))
        return digest.hexdigest()

"""Assemble one deterministic feature table for one horizon and one cutoff."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from tramflow_ml.features.aggregate import AggregateIndex, aggregate
from tramflow_ml.features.calendar_features import calendar_feature_names, calendar_features
from tramflow_ml.features.coverage import CoverageCalendar, CoverageView
from tramflow_ml.features.history import (
    lag_features,
    lag_names,
    rolling_features,
    rolling_names,
    seasonal_features,
    seasonal_names,
)
from tramflow_ml.features.periods import horizon_buckets
from tramflow_ml.features.policy import HorizonPolicy
from tramflow_ml.features.records import (
    FEATURE_VERSION,
    Bucket,
    CapacityRecord,
    EntityKey,
    FeatureError,
    FeatureHeader,
    FeatureRow,
    FeatureTable,
    FeatureValue,
    Observation,
    validate_count_pair,
)

HORIZON_INDEX = "horizon_index"
ENTITY_CAPACITY = "entity_capacity"


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    """One horizon at one origin.

    ``capacities`` is copied on construction, so a caller that keeps its own dict cannot
    change a built request afterwards. Every capacity carries its own availability
    instant and is resolved at the cutoff like any other input.
    """

    policy: HorizonPolicy
    origin: datetime
    entities: tuple[EntityKey, ...]
    observations: tuple[Observation, ...]
    coverage: CoverageCalendar
    target: str
    unit: str
    capacities: Mapping[EntityKey, CapacityRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entities:
            raise FeatureError("a feature request needs at least one entity")
        validate_count_pair(self.target, self.unit)
        stamped: dict[EntityKey, CapacityRecord] = {}
        for entity, record in self.capacities.items():
            if not isinstance(record, CapacityRecord):
                raise FeatureError(
                    f"capacity for {entity.stop_id} must be a CapacityRecord carrying its "
                    "own availability instant; an unstamped capacity cannot be placed "
                    "relative to a cutoff"
                )
            stamped[entity] = record
        object.__setattr__(self, "capacities", MappingProxyType(stamped))

    def capacity_at(self, entity: EntityKey, cutoff: datetime) -> FeatureValue:
        """Missing when unknown and when known only after the cutoff."""
        record = self.capacities.get(entity)
        return None if record is None else record.value_at(cutoff)


def feature_names(policy: HorizonPolicy) -> tuple[str, ...]:
    """The column order of every table this policy produces."""
    return (
        *calendar_feature_names(policy.granularity),
        HORIZON_INDEX,
        *lag_names(policy),
        *rolling_names(policy),
        *seasonal_names(policy),
        ENTITY_CAPACITY,
    )


@dataclass(frozen=True, slots=True)
class _Tables:
    """What every row of one build shares."""

    request: FeatureRequest
    history: AggregateIndex
    labels: AggregateIndex
    cutoff: datetime
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PerEntity:
    """What every bucket of one entity shares; rolling windows are constant per entity."""

    entity: EntityKey
    rolling: Mapping[str, FeatureValue]
    capacity: FeatureValue


def build_features(request: FeatureRequest) -> FeatureTable:
    """History is the cutoff-visible slice; labels come from the full series separately."""
    policy = request.policy
    cutoff = policy.cutoff(request.origin)
    visible = tuple(
        observation for observation in request.observations if observation.visible_at(cutoff)
    )
    tables = _Tables(
        request=request,
        history=_index(request, visible, request.coverage.as_of(cutoff)),
        labels=_index(request, request.observations, request.coverage.complete()),
        cutoff=cutoff,
        names=feature_names(policy),
    )
    buckets = horizon_buckets(request.origin, policy.horizon)
    entities = tuple(sorted(set(request.entities)))
    rows: list[FeatureRow] = []
    for entity in entities:
        per_entity = _PerEntity(
            entity=entity,
            rolling=rolling_features(tables.history, entity, policy, cutoff),
            capacity=request.capacity_at(entity, cutoff),
        )
        rows.extend(
            _row(tables, per_entity, bucket, index) for index, bucket in enumerate(buckets)
        )
    header = _header(request, cutoff, tables.names, len(entities), len(buckets))
    return FeatureTable(header, tuple(rows))


def _index(
    request: FeatureRequest, observations: tuple[Observation, ...], coverage: CoverageView
) -> AggregateIndex:
    return aggregate(
        observations, request.policy.granularity, coverage, request.target, request.unit
    )


def _header(
    request: FeatureRequest,
    cutoff: datetime,
    names: tuple[str, ...],
    entities: int,
    buckets: int,
) -> FeatureHeader:
    return {
        "feature_version": FEATURE_VERSION,
        "policy": request.policy.name,
        "horizon": request.policy.horizon,
        "granularity": request.policy.granularity,
        "timezone": "Europe/Moscow",
        "forecast_origin": request.origin.isoformat(),
        "cutoff": cutoff.isoformat(),
        "target": request.target,
        "unit": request.unit,
        "feature_names": list(names),
        "entities": entities,
        "buckets": buckets,
    }


def _row(tables: _Tables, per_entity: _PerEntity, bucket: Bucket, index: int) -> FeatureRow:
    request = tables.request
    policy = request.policy
    entity = per_entity.entity
    values: dict[str, FeatureValue] = {
        **calendar_features(bucket, policy.granularity),
        HORIZON_INDEX: float(index),
        **lag_features(tables.history, entity, bucket.start, policy, tables.cutoff),
        **per_entity.rolling,
        **seasonal_features(tables.history, entity, bucket.start, policy, tables.cutoff),
        ENTITY_CAPACITY: per_entity.capacity,
    }
    if tuple(values) != tables.names:
        raise FeatureError("feature values disagree with the declared column order")
    label = tables.labels.cell(entity, bucket.start)
    return FeatureRow(
        entity=entity,
        bucket=bucket,
        cutoff=tables.cutoff,
        target=request.target,
        unit=request.unit,
        target_value=label.value,
        target_coverage=label.coverage,
        target_covered_units=label.covered_units,
        target_total_units=label.total_units,
        features=MappingProxyType(values),
    )

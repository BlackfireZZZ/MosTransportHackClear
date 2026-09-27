"""Lags, rolling and seasonal statistics, every one of them clipped at the cutoff."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from tramflow_ml.features.aggregate import AggregateIndex
from tramflow_ml.features.periods import bucket_from_start, bucket_start_of, step
from tramflow_ml.features.policy import HorizonPolicy
from tramflow_ml.features.records import EntityKey, FeatureValue, spans_multiple_dates

ROLLING_STATISTICS = ("sum", "mean", "max", "coverage")
SEASONAL_STATISTICS = ("mean", "coverage")
UNITS = "units"


def carries_unit_ratio(policy: HorizonPolicy) -> bool:
    """A covered-units ratio is carried only where a bucket spans several civil dates.

    At hourly and daily granularity a bucket is exactly one date, so the ratio would be
    1.0 whenever the value exists and would say nothing the value does not already say.
    At monthly granularity it separates a complete month from a month with one day of data.
    """
    return spans_multiple_dates(policy.granularity)


def lag_name(policy: HorizonPolicy, steps: int) -> str:
    return f"lag_{steps}{policy.suffix}"


def rolling_prefix(policy: HorizonPolicy, window: int) -> str:
    return f"roll_{window}{policy.suffix}"


def seasonal_prefix(policy: HorizonPolicy) -> str:
    return f"seasonal_{policy.season_periods}x{policy.season_step}{policy.suffix}"


def _named(prefix: str, statistics: tuple[str, ...], with_units: bool) -> tuple[str, ...]:
    names = tuple(f"{prefix}_{statistic}" for statistic in statistics)
    return (*names, f"{prefix}_{UNITS}") if with_units else names


def lag_names(policy: HorizonPolicy) -> tuple[str, ...]:
    units = carries_unit_ratio(policy)
    return tuple(
        name
        for steps in policy.lags
        for name in _lag_columns(lag_name(policy, steps), units)
    )


def _lag_columns(name: str, with_units: bool) -> tuple[str, ...]:
    return (name, f"{name}_{UNITS}") if with_units else (name,)


def rolling_names(policy: HorizonPolicy) -> tuple[str, ...]:
    units = carries_unit_ratio(policy)
    return tuple(
        name
        for window in policy.rolling_windows
        for name in _named(rolling_prefix(policy, window), ROLLING_STATISTICS, units)
    )


def seasonal_names(policy: HorizonPolicy) -> tuple[str, ...]:
    return _named(seasonal_prefix(policy), SEASONAL_STATISTICS, carries_unit_ratio(policy))


def available_value(
    index: AggregateIndex, entity: EntityKey, start: datetime, cutoff: datetime
) -> FeatureValue:
    """A bucket still running at the cutoff is missing, never a partial sum."""
    if not bucket_from_start(start, index.granularity).ends_by(cutoff):
        return None
    cell = index.cell(entity, start)
    return None if cell.value is None else float(cell.value)


def unit_counts(
    index: AggregateIndex, start: datetime, cutoff: datetime
) -> tuple[int, int]:
    """Available and total civil dates of one bucket, clipped at the cutoff.

    The availability question is asked under the same rule ``available_value`` applies:
    a bucket still running at, or starting after, the cutoff has no available dates yet,
    whatever the coverage calendar says about those dates in the end. Without that, a
    coverage statement reaching past the cutoff would answer truthfully about the future
    and the ratio would carry it into the feature vector -- a leak through the
    ``*_units`` columns even while the value beside them is correctly missing. The total
    is a pure calendar fact -- how many civil dates the bucket spans -- and leaks nothing.
    """
    bucket = bucket_from_start(start, index.granularity)
    available, total = index.coverage.units(bucket)
    return (available, total) if bucket.ends_by(cutoff) else (0, total)


def anchor_start(policy: HorizonPolicy, cutoff: datetime) -> datetime:
    """Start of the last bucket that has fully elapsed at the cutoff."""
    return step(bucket_start_of(cutoff, policy.granularity), policy.granularity, -1)


def lag_features(
    index: AggregateIndex,
    entity: EntityKey,
    target_start: datetime,
    policy: HorizonPolicy,
    cutoff: datetime,
) -> dict[str, FeatureValue]:
    """Whole-bucket offsets from the target; a lag inside the horizon stays missing."""
    features: dict[str, FeatureValue] = {}
    for steps in policy.lags:
        start = step(target_start, policy.granularity, -steps)
        name = lag_name(policy, steps)
        features[name] = available_value(index, entity, start, cutoff)
        if carries_unit_ratio(policy):
            features[f"{name}_{UNITS}"] = _ratio(*unit_counts(index, start, cutoff))
    return features


@dataclass(frozen=True, slots=True)
class _Window:
    """One set of bucket starts scored as a group."""

    starts: tuple[datetime, ...]
    prefix: str
    size: int
    statistics: tuple[str, ...]


def rolling_features(
    index: AggregateIndex, entity: EntityKey, policy: HorizonPolicy, cutoff: datetime
) -> dict[str, FeatureValue]:
    """Windows end at the cutoff, so no anchor can reach into the forecast horizon."""
    anchor = anchor_start(policy, cutoff)
    features: dict[str, FeatureValue] = {}
    for window in policy.rolling_windows:
        starts = tuple(
            step(anchor, policy.granularity, -offset) for offset in range(window - 1, -1, -1)
        )
        described = _Window(starts, rolling_prefix(policy, window), window, ROLLING_STATISTICS)
        features.update(_window_features(index, entity, described, policy, cutoff))
    return features


def seasonal_features(
    index: AggregateIndex,
    entity: EntityKey,
    target_start: datetime,
    policy: HorizonPolicy,
    cutoff: datetime,
) -> dict[str, FeatureValue]:
    """The same calendar phase in earlier periods: whole seasons back, never 30 days."""
    starts = tuple(
        step(target_start, policy.granularity, -policy.season_step * period)
        for period in range(policy.season_periods, 0, -1)
    )
    described = _Window(
        starts, seasonal_prefix(policy), policy.season_periods, SEASONAL_STATISTICS
    )
    return _window_features(index, entity, described, policy, cutoff)


def _window_features(
    index: AggregateIndex,
    entity: EntityKey,
    window: _Window,
    policy: HorizonPolicy,
    cutoff: datetime,
) -> dict[str, FeatureValue]:
    values = [available_value(index, entity, start, cutoff) for start in window.starts]
    features = _statistics(window.prefix, values, window.size, window.statistics)
    if carries_unit_ratio(policy):
        counts = [unit_counts(index, start, cutoff) for start in window.starts]
        features[f"{window.prefix}_{UNITS}"] = _ratio(
            sum(available for available, _ in counts), sum(total for _, total in counts)
        )
    return features


def _ratio(available: int, total: int) -> float:
    return available / total if total else 0.0


def _statistics(
    prefix: str, values: list[FeatureValue], window: int, statistics: tuple[str, ...]
) -> dict[str, FeatureValue]:
    """Summed in chronological order so the float result is reproducible.

    Only the requested statistics are evaluated: a seasonal window asks for the mean and
    the coverage, so its maximum is never computed.
    """
    observed = [value for value in values if value is not None]
    total = _ordered_sum(observed)
    computed: Mapping[str, Callable[[], FeatureValue]] = {
        "sum": lambda: total,
        "mean": lambda: None if total is None else total / len(observed),
        "max": lambda: max(observed) if observed else None,
        "coverage": lambda: len(observed) / window,
    }
    return {f"{prefix}_{name}": computed[name]() for name in statistics}


def _ordered_sum(observed: list[float]) -> float | None:
    if not observed:
        return None
    total = 0.0
    for value in observed:
        total += value
    return total

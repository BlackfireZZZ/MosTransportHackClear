"""What the measured span and coverage can carry, and why not when they cannot.

The rule is stated once here rather than judged case by case: a horizon needs the
deepest reach its policy declares plus one forecast period of complete buckets, and
enough of the span's civil dates to have been observed at all. Every verdict is
reported next to the two numbers that produced it.

A bucket counts only when it lies wholly inside the observed span. A sample whose
first row lands at 23:00 has not observed that day, and counting it whole would turn
eight days of rows into a supported daily horizon.

This derives eligibility from the data, not from a fold plan: the surplus is an upper
bound on evaluable buckets, and scheduling folds belongs to the backtest layer.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from tramflow_ml.features.policy import POLICIES, HorizonPolicy
from tramflow_ml.features.records import Granularity
from tramflow_ml.ingestion.normalize import TARGETS, UNIT
from tramflow_ml.ingestion.records import MOSCOW
from tramflow_ml.intake.records import (
    MIN_COVERAGE_RATIO,
    OTHER_VALUE,
    TARGET_BUCKETS,
    rate,
)

SECONDS_PER_HOUR = 3600
MONTHS_PER_YEAR = 12


@dataclass(frozen=True, slots=True)
class ObservedSpan:
    """The civil dates a stream touches, and the instants that bound them."""

    first: date | None
    last: date | None
    observed_dates: int
    first_instant: datetime | None = None
    last_instant: datetime | None = None

    @property
    def span_days(self) -> int:
        if self.first is None or self.last is None:
            return 0
        return (self.last - self.first).days + 1

    @property
    def coverage_ratio(self) -> float:
        return rate(self.observed_dates, self.span_days)

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage_ratio": self.coverage_ratio,
            "first_date": self.first.isoformat() if self.first else None,
            "first_instant": self.first_instant.isoformat() if self.first_instant else None,
            "last_date": self.last.isoformat() if self.last else None,
            "last_instant": self.last_instant.isoformat() if self.last_instant else None,
            "observed_dates": self.observed_dates,
            "span_days": self.span_days,
            "unobserved_dates_in_span": self.span_days - self.observed_dates,
        }


def horizon_support(span: ObservedSpan) -> dict[str, object]:
    return {policy.horizon: _policy_support(policy, span) for policy in _sorted_policies()}


def target_support(
    target_values: Mapping[str, int], unit_values: Mapping[str, int]
) -> dict[str, object]:
    producible = unit_values.get(UNIT, 0) > 0 and unit_values.get(OTHER_VALUE, 0) == 0
    return {
        "observed": {
            name: _target_verdict(name, target_values.get(name, 0), unit_values, producible)
            for name in sorted(TARGETS)
        },
        "unrecognised_target_rows": target_values.get(OTHER_VALUE, 0),
        "units": dict(sorted(unit_values.items())),
    }


def reach_buckets(policy: HorizonPolicy) -> int:
    """The oldest bucket any feature of this policy reads."""
    return max(
        max(policy.lags),
        max(policy.rolling_windows),
        policy.season_step * policy.season_periods,
    )


def complete_buckets(granularity: Granularity, span: ObservedSpan) -> int:
    """Buckets lying wholly inside ``[first_instant, last_instant]``."""
    first, last = span.first_instant, span.last_instant
    if first is None or last is None:
        return 0
    if granularity == "hourly":
        return _whole_hours(first, last)
    if granularity == "daily":
        return _whole_days(first, last)
    return _whole_months(first, last)


def _sorted_policies() -> list[HorizonPolicy]:
    return sorted(POLICIES.values(), key=lambda policy: policy.horizon)


def _policy_support(policy: HorizonPolicy, span: ObservedSpan) -> dict[str, object]:
    granularity = policy.granularity
    history = reach_buckets(policy)
    target = TARGET_BUCKETS[policy.horizon]
    required = history + target
    observed = complete_buckets(granularity, span)
    blockers: list[str] = []
    if observed < required:
        blockers.append(
            f"insufficient_history: policy {policy.name!r} reads {history} {granularity} "
            f"buckets of history and forecasts {target} more, so it needs {required}; "
            f"the observed span of {span.span_days} days yields {observed}"
        )
    if span.coverage_ratio < MIN_COVERAGE_RATIO:
        blockers.append(
            f"sparse_coverage: {span.observed_dates} of {span.span_days} civil dates in the "
            f"span carry rows (ratio {span.coverage_ratio}); at least {MIN_COVERAGE_RATIO} "
            "is required before a history window means anything"
        )
    return {
        "blockers": blockers,
        "evaluable_buckets_upper_bound": max(observed - required, 0),
        "granularity": granularity,
        "history_buckets": history,
        "observed_complete_buckets": observed,
        "policy": policy.name,
        "required_complete_buckets": required,
        "required_coverage_ratio": MIN_COVERAGE_RATIO,
        "supportable": not blockers,
        "target_buckets": target,
    }


def _target_verdict(
    name: str, rows: int, unit_values: Mapping[str, int], producible: bool
) -> dict[str, object]:
    blockers: list[str] = []
    if rows == 0:
        blockers.append(f"absent: no row declares the target {name!r}")
    elif not producible:
        blockers.append(
            f"unit_not_producible: this layer can only produce {UNIT!r}; the sample carries "
            f"{unit_values.get(OTHER_VALUE, 0)} rows with another unit"
        )
    return {"blockers": blockers, "rows": rows, "supportable": not blockers}


def _whole_hours(first: datetime, last: datetime) -> int:
    start = first.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    if start < first.astimezone(UTC):
        start += timedelta(hours=1)
    end = last.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return max(int((end - start).total_seconds() // SECONDS_PER_HOUR), 0)


def _whole_days(first: datetime, last: datetime) -> int:
    """The date holding the last event is never complete: its remaining hours are unseen."""
    start_local, end_local = first.astimezone(MOSCOW), last.astimezone(MOSCOW)
    start = start_local.date() if _is_midnight(start_local) else start_local.date() + timedelta(1)
    end = end_local.date() - timedelta(days=1)
    return (end - start).days + 1 if end >= start else 0


def _whole_months(first: datetime, last: datetime) -> int:
    """Likewise the month holding the last event; only earlier months are wholly seen."""
    start_local, end_local = first.astimezone(MOSCOW), last.astimezone(MOSCOW)
    start = _month_index(start_local) + (0 if _is_month_start(start_local) else 1)
    end = _month_index(end_local) - 1
    return max(end - start + 1, 0)


def _is_midnight(moment: datetime) -> bool:
    return moment.timetz().replace(tzinfo=None) == time()


def _is_month_start(moment: datetime) -> bool:
    return moment.day == 1 and _is_midnight(moment)


def _month_index(moment: datetime) -> int:
    return moment.year * MONTHS_PER_YEAR + moment.month - 1

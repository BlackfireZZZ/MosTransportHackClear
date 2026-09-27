"""The fold index: which rolling origins are folds, which are not, and why.

Eligibility depends on the coverage calendar and the configuration alone, never on the
event rows. A fold set is therefore unchanged by any edit to the events, which is what
makes "the candidate and the baseline were scored on the same folds" a property of the
construction rather than a promise.
"""

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from tramflow_ml.backtest.config import BacktestConfig, FoldRules
from tramflow_ml.backtest.horizons import (
    bucket_ceiling,
    bucket_span,
    period_ceiling,
    period_step,
)
from tramflow_ml.backtest.records import BACKTEST_VERSION, Fold, Ineligible, Window
from tramflow_ml.features import (
    Bucket,
    CoverageCalendar,
    CoverageView,
    Horizon,
    bucket_from_start,
    horizon_buckets,
    step,
)
from tramflow_ml.features.periods import midnight
from tramflow_ml.features.records import digest_of

ONE_DAY = timedelta(days=1)


@dataclass(frozen=True, slots=True)
class DataSpan:
    """Half-open span of instants the source states it covers, from its coverage dates."""

    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def data_span(coverage: CoverageCalendar) -> DataSpan:
    """From the first covered civil date to Moscow midnight after the last one."""
    days = sorted(coverage.dates)
    return DataSpan(midnight(days[0]), midnight(days[-1] + ONE_DAY))


@dataclass(frozen=True, slots=True)
class _CoveredBuckets:
    """Prefix counts of covered buckets over one horizon's grid, built once per horizon.

    Train history has to be counted in buckets that carry data, not in calendar
    distance: a window spanning three years of which the source covers one day is not
    three years of history. Counting per fold would rewalk the same grid for every
    origin, so the grid is walked once and every window is answered by two bisections.
    """

    starts: tuple[datetime, ...]
    covered_before: tuple[int, ...]

    def count(self, start: datetime, end: datetime) -> int:
        """Covered buckets whose start lies in the half-open window."""
        first = bisect_left(self.starts, start)
        last = bisect_left(self.starts, end)
        return self.covered_before[last] - self.covered_before[first]


@dataclass(frozen=True, slots=True)
class _Grading:
    """What every candidate origin of one horizon is judged against."""

    rules: FoldRules
    span: DataSpan
    labels: CoverageView
    series_start: datetime
    covered: _CoveredBuckets


@dataclass(frozen=True, slots=True)
class FoldSet:
    """Every candidate origin of every configured horizon, graded exactly once."""

    config: BacktestConfig
    span: DataSpan
    folds: Mapping[Horizon, tuple[Fold, ...]]
    refusals: Mapping[Horizon, tuple[Ineligible, ...]]

    def for_horizon(self, horizon: Horizon) -> tuple[Fold, ...]:
        return self.folds.get(horizon, ())

    def refusals_for(self, horizon: Horizon) -> tuple[Ineligible, ...]:
        return self.refusals.get(horizon, ())

    def to_dict(self) -> dict[str, object]:
        return {
            "backtest_version": BACKTEST_VERSION,
            "span": self.span.to_dict(),
            "horizons": [
                {
                    "horizon": horizon,
                    "folds": [fold.to_dict() for fold in self.for_horizon(horizon)],
                    "refusals": [item.to_dict() for item in self.refusals_for(horizon)],
                }
                for horizon in self.config.horizons
            ],
        }

    @property
    def fold_hash(self) -> str:
        return digest_of(self.to_dict())


def build_fold_set(config: BacktestConfig, coverage: CoverageCalendar) -> FoldSet:
    """Compute the fold set once. Nothing downstream may compute another one."""
    span = data_span(coverage)
    labels = coverage.complete()
    folds: dict[Horizon, tuple[Fold, ...]] = {}
    refusals: dict[Horizon, tuple[Ineligible, ...]] = {}
    for rules in config.rules:
        series_start = bucket_ceiling(span.start, rules.policy.granularity)
        grading = _Grading(
            rules=rules,
            span=span,
            labels=labels,
            series_start=series_start,
            covered=_covered_buckets(rules, span, labels, series_start),
        )
        eligible, refused = _grade(grading)
        folds[rules.horizon] = eligible
        refusals[rules.horizon] = refused
    return FoldSet(config, span, MappingProxyType(folds), MappingProxyType(refusals))


def _covered_buckets(
    rules: FoldRules, span: DataSpan, labels: CoverageView, series_start: datetime
) -> _CoveredBuckets:
    granularity = rules.policy.granularity
    limit = span.end.astimezone(UTC)
    starts: list[datetime] = []
    prefix = [0]
    start = series_start
    while start.astimezone(UTC) < limit:
        available, _ = labels.units(bucket_from_start(start, granularity))
        starts.append(start)
        prefix.append(prefix[-1] + (1 if available else 0))
        start = step(start, granularity, 1)
    return _CoveredBuckets(tuple(starts), tuple(prefix))


def candidate_origins(rules: FoldRules, span: DataSpan) -> tuple[datetime, ...]:
    """Period-aligned origins tiling the span, enumerated before anything is judged."""
    limit = span.end.astimezone(UTC)
    origin = period_ceiling(span.start, rules.horizon)
    origins: list[datetime] = []
    while origin.astimezone(UTC) < limit:
        origins.append(origin)
        origin = period_step(origin, rules.horizon, rules.origin_stride_periods)
    return tuple(origins)


def fold_id(horizon: Horizon, origin: datetime) -> str:
    """Keyed on the origin, so a fold keeps its name when the rules change."""
    return f"{horizon}@{origin.isoformat()}"


def _grade(grading: _Grading) -> tuple[tuple[Fold, ...], tuple[Ineligible, ...]]:
    eligible: list[Fold] = []
    refused: list[Ineligible] = []
    for origin in candidate_origins(grading.rules, grading.span):
        outcome = _judge(grading, origin, len(eligible))
        if isinstance(outcome, Fold):
            eligible.append(outcome)
        else:
            refused.append(outcome)
    return tuple(eligible), tuple(refused)


def _judge(grading: _Grading, origin: datetime, index: int) -> Fold | Ineligible:
    rules = grading.rules
    buckets = horizon_buckets(origin, rules.horizon)
    units = tuple(grading.labels.units(bucket) for bucket in buckets)
    label_refusal = _label_refusal(grading, origin, buckets, units)
    if label_refusal is not None:
        return label_refusal
    cutoff = rules.effective_policy.cutoff(origin)
    train, validation = _windows(grading, origin, cutoff)
    train_buckets = bucket_span(train.start, train.end, rules.policy.granularity)
    covered_buckets = grading.covered.count(train.start, train.end)
    history_refusal = _history_refusal(
        grading, origin, validation, train_buckets, covered_buckets
    )
    if history_refusal is not None:
        return history_refusal
    return Fold(
        fold_id=fold_id(rules.horizon, origin),
        index=index,
        horizon=rules.horizon,
        policy_name=rules.policy.name,
        origin=origin,
        cutoff=cutoff,
        train=train,
        validation=validation,
        test=Window(origin, buckets[-1].end),
        train_buckets=train_buckets,
        train_covered_buckets=covered_buckets,
        test_buckets=len(buckets),
        label_covered_units=sum(available for available, _ in units),
        label_total_units=sum(total for _, total in units),
    )


def _label_refusal(
    grading: _Grading,
    origin: datetime,
    buckets: tuple[Bucket, ...],
    units: tuple[tuple[int, int], ...],
) -> Ineligible | None:
    """A horizon that runs past the data, or has a hole in it, is not a fold."""
    rules = grading.rules
    end = buckets[-1].end
    if end.astimezone(UTC) > grading.span.end.astimezone(UTC):
        return Ineligible(
            rules.horizon,
            origin,
            "horizon_beyond_data",
            f"horizon ends {end.isoformat()}, data ends {grading.span.end.isoformat()}",
        )
    unlabelled = sum(1 for available, _ in units if available == 0)
    if unlabelled:
        return Ineligible(
            rules.horizon,
            origin,
            "incomplete_labels",
            f"{unlabelled} of {len(buckets)} horizon buckets have no label",
        )
    covered = sum(available for available, _ in units)
    total = sum(span for _, span in units)
    ratio = covered / total
    if ratio < rules.minimum_label_unit_ratio:
        return Ineligible(
            rules.horizon,
            origin,
            "diluted_labels",
            f"{covered} of {total} horizon dates covered "
            f"({ratio:.4f} < {rules.minimum_label_unit_ratio})",
        )
    return None


def _history_refusal(
    grading: _Grading,
    origin: datetime,
    validation: Window,
    train_buckets: int,
    covered_buckets: int,
) -> Ineligible | None:
    """Calendar distance is the cheap pre-filter; covered buckets are the real rule."""
    rules = grading.rules
    required = rules.train_buckets_required
    if validation.start.astimezone(UTC) < grading.series_start.astimezone(UTC):
        return Ineligible(
            rules.horizon,
            origin,
            "insufficient_train_history",
            f"validation would start {validation.start.isoformat()}, before the data",
        )
    if train_buckets < required:
        return Ineligible(
            rules.horizon,
            origin,
            "insufficient_train_history",
            f"{train_buckets} train buckets available, {required} required",
        )
    if covered_buckets < required:
        return Ineligible(
            rules.horizon,
            origin,
            "insufficient_train_history",
            f"{covered_buckets} of {train_buckets} train buckets are covered, "
            f"{required} required",
        )
    return None


def _windows(grading: _Grading, origin: datetime, cutoff: datetime) -> tuple[Window, Window]:
    """Whole-period validation ending at or before the cutoff; train ends an embargo earlier.

    A partial validation period is dropped rather than shortened: a validation window of a
    different length would be a different task from the test one it selects models for.
    """
    rules = grading.rules
    limit = cutoff.astimezone(UTC)
    validation_end = origin
    while validation_end.astimezone(UTC) > limit:
        validation_end = period_step(validation_end, rules.horizon, -1)
    validation_start = period_step(validation_end, rules.horizon, -rules.validation_periods)
    train_end = step(validation_start, rules.policy.granularity, -rules.embargo_buckets)
    train_start = min(grading.series_start, train_end, key=lambda value: value.astimezone(UTC))
    return Window(train_start, train_end), Window(validation_start, validation_end)

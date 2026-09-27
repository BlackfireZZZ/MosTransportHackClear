"""What one slice measured, and what it refused to measure.

Error totals come from ``backtest.metrics`` unchanged, so WAPE has one definition in
this repository's offline path and zero demand yields ``None`` here for the same reason
it does there. Interval quality is the interval (Winkler) score reported beside coverage
and width: coverage alone rewards ``[0, inf)``, which is perfectly covered and says
nothing. Overload is a fraction of capacity, so it is absent -- with the reason -- unless
the target is an occupancy and the capacity is actually known. It is reported and
**not gated**: the feature layer can only produce event counts today, so no slice this
repository can build carries an overload figure at all, and a threshold over a quantity
nothing yet produces would be a number invented to look rigorous. Gating it belongs with
the task that first produces a load target.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tramflow_ml.backtest.metrics import FoldMetrics, fold_metrics
from tramflow_ml.slices.records import IntervalBounds, ScoredPoint, SliceError, SliceKey

LOAD_TARGET = "onboard_load"
LOAD_UNIT = "passengers"
RATIO_UNIT = "ratio"
COUNT_LABEL = "count"
"""Units a reader needs to not misapply a number.

``SliceMetrics.unit`` is the *target's* unit and applies to the quantities measured in
it. ``wape`` and coverage are dimensionless, and a consumer that applies the target unit
to them is simply wrong, so every metric names its own unit in ``units``.
"""

SliceStatus = Literal[
    "evaluated",
    "insufficient_history",
    "insufficient_samples",
    "insufficient_signal",
    "missing",
]
PASSING_STATUS: SliceStatus = "evaluated"
MISSING_STATUS: SliceStatus = "missing"
"""``missing`` says a required slice has no points at all, so no ``SliceMetrics`` carries
it: a measured slice has at least one sample. It is kept distinct from
``insufficient_samples`` because "this cut of the data does not exist" and "this cut is
too small to judge" call for different responses -- the first is a declaration to fix or
data to obtain, the second is more data on a cut that is already there."""


def interval_score(bounds: IntervalBounds, actual: float) -> float:
    """Winkler score: width plus ``2/alpha`` times the distance missed, lower is better.

    In the target's unit, so it is comparable with MAE and with mean demand. A wider
    interval can only lower this by the coverage it buys, which is what makes it a
    proper rule and coverage on its own not one.
    """
    alpha = 1.0 - bounds.level
    below = max(bounds.lower - actual, 0.0)
    above = max(actual - bounds.upper, 0.0)
    return bounds.width + (2.0 / alpha) * (below + above)


@dataclass(frozen=True, slots=True)
class IntervalQuality:
    """Coverage, width and score together; none of the three is reportable alone."""

    level: float
    method: str
    samples: int
    covered: int
    width_total: float
    score_total: float

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise SliceError("interval quality needs at least one bounded point")
        if not 0 <= self.covered <= self.samples:
            raise SliceError("covered points must lie within the bounded points")

    @property
    def coverage(self) -> float:
        return self.covered / self.samples

    @property
    def mean_width(self) -> float:
        return self.width_total / self.samples

    @property
    def mean_score(self) -> float:
        return self.score_total / self.samples

    def to_dict(self, target_unit: str) -> dict[str, object]:
        """No ``folds``: this is ``None`` unless every point qualifies, so the slice's
        fold count applies unchanged."""
        return {
            "level": self.level,
            "method": self.method,
            "samples": self.samples,
            "covered": self.covered,
            "coverage": self.coverage,
            "mean_width": self.mean_width,
            "mean_interval_score": self.mean_score,
            "units": {
                "level": RATIO_UNIT,
                "coverage": RATIO_UNIT,
                "covered": COUNT_LABEL,
                "mean_interval_score": target_unit,
                "mean_width": target_unit,
                "samples": COUNT_LABEL,
            },
        }


@dataclass(frozen=True, slots=True)
class OverloadQuality:
    """Only ever built where a capacity was known; there is no default capacity here."""

    samples: int
    actual_overloaded: int
    predicted_overloaded: int

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise SliceError("overload quality needs at least one point")
        for name in ("actual_overloaded", "predicted_overloaded"):
            count: int = getattr(self, name)
            if not 0 <= count <= self.samples:
                raise SliceError(f"{name} must lie within the points it counts")

    @property
    def actual_rate(self) -> float:
        return self.actual_overloaded / self.samples

    @property
    def predicted_rate(self) -> float:
        return self.predicted_overloaded / self.samples

    def to_dict(self) -> dict[str, object]:
        """No ``folds``: as with intervals, this is ``None`` unless every point qualifies."""
        return {
            "samples": self.samples,
            "actual_overloaded": self.actual_overloaded,
            "predicted_overloaded": self.predicted_overloaded,
            "actual_overload_rate": self.actual_rate,
            "predicted_overload_rate": self.predicted_rate,
            "units": {
                "actual_overload_rate": RATIO_UNIT,
                "actual_overloaded": COUNT_LABEL,
                "predicted_overload_rate": RATIO_UNIT,
                "predicted_overloaded": COUNT_LABEL,
                "samples": COUNT_LABEL,
            },
        }


def interval_quality(points: Sequence[ScoredPoint]) -> tuple[IntervalQuality | None, str]:
    bounded = [
        (point.interval, point.actual) for point in points if point.interval is not None
    ]
    if not bounded:
        return None, "no point in this slice carries an interval"
    if len(bounded) != len(points):
        return None, (
            f"only {len(bounded)} of {len(points)} points carry an interval; "
            "a partial coverage figure would describe a different slice"
        )
    levels = sorted({bounds.level for bounds, _ in bounded})
    if len(levels) > 1:
        return None, f"points declare different interval levels: {levels}"
    methods = sorted({bounds.method for bounds, _ in bounded})
    if len(methods) > 1:
        return None, f"points declare different interval methods: {methods}"
    covered = sum(bounds.covers(actual) for bounds, actual in bounded)
    width_total = sum(bounds.width for bounds, _ in bounded)
    score_total = sum(interval_score(bounds, actual) for bounds, actual in bounded)
    return (
        IntervalQuality(levels[0], methods[0], len(bounded), covered, width_total, score_total),
        "",
    )


def overload_quality(points: Sequence[ScoredPoint]) -> tuple[OverloadQuality | None, str]:
    target = points[0].target
    unit = points[0].unit
    if target != LOAD_TARGET:
        return None, (
            f"target {target!r} is not an occupancy; a fraction of capacity is defined "
            f"only for {LOAD_TARGET!r}"
        )
    if unit != LOAD_UNIT:
        return None, f"unit {unit!r} is not {LOAD_UNIT!r}, so it cannot be divided by a capacity"
    unknown = sum(1 for point in points if point.capacity is None)
    if unknown:
        return None, (
            f"capacity is unknown for {unknown} of {len(points)} points; overload is not "
            "computed against a guessed capacity"
        )
    loads = [
        (point.actual, point.predicted, point.capacity)
        for point in points
        if point.capacity is not None
    ]
    actual = sum(observed > capacity for observed, _, capacity in loads)
    predicted = sum(forecast > capacity for _, forecast, capacity in loads)
    return OverloadQuality(len(loads), actual, predicted), ""


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    """One slice's numbers, each inseparable from its unit, samples and folds."""

    key: SliceKey
    unit: str
    samples: int
    folds: int
    totals: FoldMetrics
    baseline: FoldMetrics | None
    baseline_absent_reason: str
    interval: IntervalQuality | None
    interval_absent_reason: str
    overload: OverloadQuality | None
    overload_absent_reason: str
    status: SliceStatus
    status_reason: str

    def __post_init__(self) -> None:
        if self.samples < 1 or self.folds < 1:
            raise SliceError("a slice carries at least one sample in at least one fold")
        if self.folds > self.samples:
            raise SliceError("a slice cannot span more folds than it has samples")
        if self.samples != self.totals.scored:
            raise SliceError("every sample of a slice is scored")

    @property
    def mae(self) -> float | None:
        return self.totals.mae

    @property
    def wape(self) -> float | None:
        return self.totals.wape

    @property
    def baseline_wape(self) -> float | None:
        return None if self.baseline is None else self.baseline.wape

    @property
    def mean_actual(self) -> float:
        return self.totals.actual_total / self.samples

    @property
    def relative_interval_width(self) -> float | None:
        if self.interval is None or self.totals.actual_total == 0:
            return None
        return self.interval.mean_width / self.mean_actual

    def _units(self) -> dict[str, str]:
        """``unit`` is the target's; these say which metric it actually applies to."""
        return {
            "actual_total": self.unit,
            "baseline_wape": RATIO_UNIT,
            "error_total": self.unit,
            "folds": COUNT_LABEL,
            "mae": self.unit,
            "mean_actual": self.unit,
            "relative_interval_width": RATIO_UNIT,
            "samples": COUNT_LABEL,
            "wape": RATIO_UNIT,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.key.to_dict(),
            "unit": self.unit,
            "units": self._units(),
            "samples": self.samples,
            "folds": self.folds,
            "status": self.status,
            "status_reason": self.status_reason,
            "actual_total": self.totals.actual_total,
            "error_total": self.totals.error_total,
            "mae": self.mae,
            "wape": self.wape,
            "mean_actual": self.mean_actual,
            "baseline_wape": self.baseline_wape,
            "baseline_absent_reason": self.baseline_absent_reason,
            "relative_interval_width": self.relative_interval_width,
            "interval": None if self.interval is None else self.interval.to_dict(self.unit),
            "interval_absent_reason": self.interval_absent_reason,
            "overload": None if self.overload is None else self.overload.to_dict(),
            "overload_absent_reason": self.overload_absent_reason,
        }


def error_totals(points: Sequence[ScoredPoint]) -> FoldMetrics:
    return fold_metrics(
        [point.actual for point in points], [point.predicted for point in points]
    )


def baseline_totals(points: Sequence[ScoredPoint]) -> tuple[FoldMetrics | None, str]:
    """``None`` unless every point carries a baseline, with the count that made it absent.

    One point without a baseline disables the comparison for every slice that point
    belongs to, `overall` included, so the reason has to say whether that was the whole
    run or a single row. Without it a reader cannot tell "no baselines this run" from
    "one row of five thousand".
    """
    carried = [
        (point.actual, point.baseline) for point in points if point.baseline is not None
    ]
    if not carried:
        return None, "no point in this slice carries a baseline"
    if len(carried) != len(points):
        return None, (
            f"only {len(carried)} of {len(points)} points carry a baseline; "
            "a partial comparison would judge a different slice"
        )
    return (
        fold_metrics([actual for actual, _ in carried], [value for _, value in carried]),
        "",
    )

"""The slice report and the verdict it produces.

Two rules carry the design. Every slice the data can judge is judged, so a bad slice
cannot escape by nobody having listed it; and a *required* slice must additionally exist
and be judgeable, so it cannot escape by disappearing. ``passed`` is therefore
``not failures and not unproven``: an excellent overall number buys nothing.

A slice too small or too short to judge keeps every metric and its support, is marked, and
is counted and named in the ungated summary rather than merely present in the payload.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from tramflow_ml.features.records import digest_of
from tramflow_ml.slices.metrics import (
    MISSING_STATUS,
    PASSING_STATUS,
    SliceMetrics,
    SliceStatus,
    baseline_totals,
    error_totals,
    interval_quality,
    overload_quality,
)
from tramflow_ml.slices.records import (
    OVERALL_VALUE,
    SINGLE_ORIGIN_AXES,
    SLICE_AXES,
    ScoredPoint,
    SliceError,
    SliceKey,
)
from tramflow_ml.slices.thresholds import GateThresholds
from tramflow_ml.slices.verdict import (
    SliceFailure,
    UngatedSummary,
    UnprovenSlice,
    judge,
    summarise_ungated,
)

OVERALL_KEY = SliceKey("overall", OVERALL_VALUE)


@dataclass(frozen=True, slots=True)
class SliceReport:
    target: str
    unit: str
    points: int
    folds: int
    thresholds: GateThresholds
    required: tuple[SliceKey, ...]
    metrics: tuple[SliceMetrics, ...]
    failures: tuple[SliceFailure, ...]
    unproven: tuple[UnprovenSlice, ...]
    ungated: UngatedSummary
    by_key: Mapping[SliceKey, SliceMetrics] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "by_key", MappingProxyType({item.key: item for item in self.metrics})
        )

    @property
    def passed(self) -> bool:
        return not self.failures and not self.unproven

    def verdict(self) -> str:
        """Always ends with the ungated line: a floor hides a slice from the gate, not
        from the reader."""
        if self.passed:
            head = f"PASS: {len(self.metrics)} slices judged against uncertified thresholds"
            return f"{head}\n{self.ungated.message()}"
        lines = [f"FAIL: {len(self.failures)} breached, {len(self.unproven)} unproven"]
        lines.extend(failure.message() for failure in self.failures)
        lines.extend(item.message() for item in self.unproven)
        lines.append(self.ungated.message())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "target": self.target,
            "unit": self.unit,
            "points": self.points,
            "folds": self.folds,
            "thresholds": self.thresholds.to_dict(),
            "required": [key.to_dict() for key in self.required],
            "slices": [item.to_dict() for item in self.metrics],
            "failures": [failure.to_dict() for failure in self.failures],
            "unproven": [item.to_dict() for item in self.unproven],
            "ungated": self.ungated.to_dict(),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def build_report(
    points: Iterable[ScoredPoint],
    *,
    required: Iterable[SliceKey] = (),
    thresholds: GateThresholds | None = None,
) -> SliceReport:
    limits = GateThresholds() if thresholds is None else thresholds
    demanded = _demanded(required)
    ordered = _canonical(points)
    target, unit = _single_quantity(ordered)
    groups = _group(ordered)
    metrics = tuple(_measure(key, groups[key], limits) for key in sorted(groups))
    by_key = {item.key: item for item in metrics}
    failures = tuple(
        failure
        for item in metrics
        if item.status == PASSING_STATUS
        for failure in judge(item, limits)
    )
    unproven = tuple(
        _unproven(key, by_key.get(key))
        for key in demanded
        if _is_unproven(by_key.get(key))
    )
    return SliceReport(
        target=target,
        unit=unit,
        points=len(ordered),
        folds=len({point.fold_id for point in ordered}),
        thresholds=limits,
        required=demanded,
        metrics=metrics,
        failures=failures,
        unproven=unproven,
        ungated=summarise_ungated(metrics),
    )


def _demanded(required: Iterable[SliceKey]) -> tuple[SliceKey, ...]:
    """A key no axis can emit would be permanently unproven, so it is refused here.

    Declaring a requirement against a misspelled axis would otherwise produce a verdict
    that blames the data for a defect in the declaration.
    """
    demanded = tuple(sorted({*required, OVERALL_KEY}))
    unknown = sorted({key.axis for key in demanded if key.axis not in SLICE_AXES})
    if unknown:
        raise SliceError(
            f"no axis emits {unknown}; required keys use one of {sorted(SLICE_AXES)}"
        )
    return demanded


def _canonical(points: Iterable[ScoredPoint]) -> tuple[ScoredPoint, ...]:
    """Sort into a total order, so no sum depends on the order the caller supplied.

    Two points tying on the sort key would keep their input order and could then sum
    differently, so a repeated key is refused rather than ordered arbitrarily: one
    entity's bucket is scored once in one fold, as ``forecast_v1`` also requires of a
    published artifact.
    """
    ordered = tuple(sorted(points, key=lambda point: point.sort_key))
    if not ordered:
        raise SliceError("a slice report needs at least one scored point")
    keys = {point.sort_key for point in ordered}
    if len(keys) != len(ordered):
        raise SliceError(
            f"{len(ordered) - len(keys)} scored point(s) repeat an entity, bucket and "
            "fold already scored; a bucket is scored once per fold"
        )
    return ordered


def _single_quantity(points: Sequence[ScoredPoint]) -> tuple[str, str]:
    targets = sorted({point.target for point in points})
    units = sorted({point.unit for point in points})
    if len(targets) > 1 or len(units) > 1:
        raise SliceError(
            f"a report covers one quantity; got targets {targets} and units {units}"
        )
    return targets[0], units[0]


def _group(points: Sequence[ScoredPoint]) -> dict[SliceKey, list[ScoredPoint]]:
    groups: dict[SliceKey, list[ScoredPoint]] = {}
    for point in points:
        for key in point.keys:
            groups.setdefault(key, []).append(point)
    return groups


def _measure(
    key: SliceKey, points: Sequence[ScoredPoint], limits: GateThresholds
) -> SliceMetrics:
    totals = error_totals(points)
    folds = len({point.fold_id for point in points})
    baseline, baseline_reason = baseline_totals(points)
    interval, interval_reason = interval_quality(points)
    overload, overload_reason = overload_quality(points)
    status, status_reason = _status(key, len(points), folds, totals.wape is None, limits)
    return SliceMetrics(
        key=key,
        unit=points[0].unit,
        samples=len(points),
        folds=folds,
        totals=totals,
        baseline=baseline,
        baseline_absent_reason=baseline_reason,
        interval=interval,
        interval_absent_reason=interval_reason,
        overload=overload,
        overload_absent_reason=overload_reason,
        status=status,
        status_reason=status_reason,
    )


def _status(
    key: SliceKey, samples: int, folds: int, no_demand: bool, limits: GateThresholds
) -> tuple[SliceStatus, str]:
    """Support is checked before signal: three rows carry no verdict either way."""
    if folds < limits.min_folds and key.axis not in SINGLE_ORIGIN_AXES:
        return "insufficient_history", (
            f"{folds} fold(s) is below MIN_FOLDS_FOR_GATE={limits.min_folds}; "
            "one origin is not a temporal generalisation"
        )
    if samples < limits.min_samples:
        return "insufficient_samples", (
            f"{samples} sample(s) is below MIN_SAMPLES_FOR_GATE={limits.min_samples}; "
            "reported in full and excluded from the gate"
        )
    if no_demand:
        return "insufficient_signal", (
            "the slice carries no demand, so WAPE is undefined and is reported as null "
            "rather than as a zero"
        )
    return PASSING_STATUS, ""


def _is_unproven(item: SliceMetrics | None) -> bool:
    return item is None or item.status != PASSING_STATUS


def _unproven(key: SliceKey, item: SliceMetrics | None) -> UnprovenSlice:
    if item is None:
        return UnprovenSlice(
            key=key,
            status=MISSING_STATUS,
            reason="no scored point falls in this slice",
            samples=0,
            folds=0,
        )
    return UnprovenSlice(
        key=key,
        status=item.status,
        reason=item.status_reason,
        samples=item.samples,
        folds=item.folds,
    )

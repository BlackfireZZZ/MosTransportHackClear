"""Failures, unproven slices, and the ungated summary that keeps them from hiding.

A failure carries the constant that judged it beside the number it judged, so a breach
can be told from a badly chosen threshold. An ungated slice is not silently dropped: the
report counts them by reason and names the worst WAPE among them, because a slice held
back from the gate by a floor is still a slice somebody has to look at.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tramflow_ml.slices.metrics import (
    PASSING_STATUS,
    RATIO_UNIT,
    SliceMetrics,
    SliceStatus,
)
from tramflow_ml.slices.records import SliceKey
from tramflow_ml.slices.thresholds import GateThresholds

Direction = Literal["at_most", "at_least"]


@dataclass(frozen=True, slots=True)
class SliceFailure:
    """A judged slice that breached a threshold, with everything needed to check it."""

    key: SliceKey
    metric: str
    unit: str
    observed: float
    limit: float
    direction: Direction
    threshold_name: str
    threshold_value: float
    samples: int
    folds: int

    def message(self) -> str:
        relation = "exceeds" if self.direction == "at_most" else "falls below"
        return (
            f"{self.key}: {self.metric} {self.observed:.6g} {self.unit} {relation} "
            f"{self.limit:.6g} ({self.threshold_name}={self.threshold_value:.6g}) "
            f"on {self.samples} samples across {self.folds} folds"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.key.to_dict(),
            "metric": self.metric,
            "unit": self.unit,
            "observed": self.observed,
            "limit": self.limit,
            "direction": self.direction,
            "threshold_name": self.threshold_name,
            "threshold_value": self.threshold_value,
            "samples": self.samples,
            "folds": self.folds,
            "message": self.message(),
        }


@dataclass(frozen=True, slots=True)
class UnprovenSlice:
    """A required slice the data could not judge. Absence never buys a pass."""

    key: SliceKey
    status: SliceStatus
    reason: str
    samples: int
    folds: int

    def message(self) -> str:
        return f"{self.key}: required slice is {self.status}: {self.reason}"

    def to_dict(self) -> dict[str, object]:
        return {
            **self.key.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "samples": self.samples,
            "folds": self.folds,
            "message": self.message(),
        }


@dataclass(frozen=True, slots=True)
class UngatedSummary:
    """How much of the report the gate did not judge, and the worst of it by name.

    A floor keeps a slice out of the gate; it must not keep it out of the reader's view.
    Twenty rows on one origin is not sample scarcity, and a WAPE of 1.0 there is a
    finding whatever the gate did with it.

    The same points appear on several axes, so ties on the worst WAPE are broken by the
    larger slice and then by key order: naming ``route=route-B`` rather than one of the
    narrower cuts of the same rows is the more useful half of the tie.
    """

    slices: int
    below_sample_floor: int
    below_fold_floor: int
    without_signal: int
    worst_key: SliceKey | None
    worst_wape: float | None

    def message(self) -> str:
        if not self.slices:
            return "every slice was judged"
        parts = (
            f"{self.slices} slices ungated: {self.below_sample_floor} below the sample "
            f"floor, {self.below_fold_floor} below the fold floor, "
            f"{self.without_signal} without demand"
        )
        if self.worst_key is None or self.worst_wape is None:
            return f"{parts}; no ungated slice has a defined WAPE"
        return f"{parts}; worst ungated WAPE {self.worst_wape:.6g} on {self.worst_key}"

    def to_dict(self) -> dict[str, object]:
        return {
            "slices": self.slices,
            "below_sample_floor": self.below_sample_floor,
            "below_fold_floor": self.below_fold_floor,
            "without_signal": self.without_signal,
            "worst_key": None if self.worst_key is None else self.worst_key.to_dict(),
            "worst_wape": self.worst_wape,
            "message": self.message(),
        }


def summarise_ungated(metrics: Sequence[SliceMetrics]) -> UngatedSummary:
    ungated = [item for item in metrics if item.status != PASSING_STATUS]
    scored = sorted(
        (item for item in ungated if item.wape is not None),
        key=lambda item: (-(item.wape or 0.0), -item.samples, item.key),
    )
    worst = scored[0] if scored else None
    return UngatedSummary(
        slices=len(ungated),
        below_sample_floor=sum(1 for item in ungated if item.status == "insufficient_samples"),
        below_fold_floor=sum(1 for item in ungated if item.status == "insufficient_history"),
        without_signal=sum(1 for item in ungated if item.status == "insufficient_signal"),
        worst_key=None if worst is None else worst.key,
        worst_wape=None if worst is None else worst.wape,
    )


def judge(item: SliceMetrics, limits: GateThresholds) -> list[SliceFailure]:
    failures = _judge_interval(item, limits)
    wape = item.wape
    if wape is None:
        return failures
    if wape > limits.max_wape:
        failures.append(
            _failure(item, "wape", RATIO_UNIT, wape, limits.max_wape, "MAX_WAPE", limits.max_wape)
        )
    failures.extend(_judge_baseline(item, limits, wape))
    return failures


def _judge_baseline(
    item: SliceMetrics, limits: GateThresholds, wape: float
) -> list[SliceFailure]:
    baseline_wape = item.baseline_wape
    if baseline_wape is None:
        return []
    limit = baseline_wape * limits.max_wape_ratio_to_baseline
    if wape <= limit:
        return []
    return [
        _failure(
            item,
            "wape_vs_baseline",
            RATIO_UNIT,
            wape,
            limit,
            "MAX_WAPE_RATIO_TO_BASELINE",
            limits.max_wape_ratio_to_baseline,
        )
    ]


def _judge_interval(item: SliceMetrics, limits: GateThresholds) -> list[SliceFailure]:
    interval = item.interval
    if interval is None:
        return []
    failures: list[SliceFailure] = []
    if interval.level < limits.min_interval_level:
        failures.append(
            _failure(
                item,
                "interval_level",
                RATIO_UNIT,
                interval.level,
                limits.min_interval_level,
                "MIN_INTERVAL_LEVEL",
                limits.min_interval_level,
                direction="at_least",
            )
        )
    else:
        failures.extend(_judge_coverage(item, interval.level, interval.coverage, limits))
    failures.extend(_judge_width(item, interval.mean_score, limits))
    return failures


def _judge_coverage(
    item: SliceMetrics, level: float, coverage: float, limits: GateThresholds
) -> list[SliceFailure]:
    """Only meaningful above ``min_interval_level``: a self-declared 0.02 asks for nothing.

    ``GateThresholds`` keeps ``max_coverage_shortfall`` below ``min_interval_level``, so
    the requirement computed here is always positive.
    """
    limit = level - limits.max_coverage_shortfall
    if coverage >= limit:
        return []
    return [
        _failure(
            item,
            "interval_coverage",
            RATIO_UNIT,
            coverage,
            limit,
            "MAX_COVERAGE_SHORTFALL",
            limits.max_coverage_shortfall,
            direction="at_least",
        )
    ]


def _judge_width(
    item: SliceMetrics, mean_score: float, limits: GateThresholds
) -> list[SliceFailure]:
    """Scale-relative, with an absolute allowance so low demand is not punished.

    Integer counts cannot be sharper than a few units, so a two-passenger bucket with a
    `[0, 7]` interval is behaving correctly; without the allowance the ratio alone failed
    it, and a gate that fires on correct behaviour is the one that gets switched off.
    """
    relative = item.mean_actual * limits.max_interval_score_to_mean_actual
    limit = max(relative, limits.min_interval_score_allowance)
    if mean_score <= limit:
        return []
    threshold_name, threshold_value = (
        ("MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL", limits.max_interval_score_to_mean_actual)
        if relative >= limits.min_interval_score_allowance
        else ("MIN_INTERVAL_SCORE_ALLOWANCE", limits.min_interval_score_allowance)
    )
    return [
        _failure(
            item,
            "mean_interval_score",
            item.unit,
            mean_score,
            limit,
            threshold_name,
            threshold_value,
        )
    ]


def _failure(
    item: SliceMetrics,
    metric: str,
    unit: str,
    observed: float,
    limit: float,
    threshold_name: str,
    threshold_value: float,
    *,
    direction: Direction = "at_most",
) -> SliceFailure:
    return SliceFailure(
        key=item.key,
        metric=metric,
        unit=unit,
        observed=observed,
        limit=limit,
        direction=direction,
        threshold_name=threshold_name,
        threshold_value=threshold_value,
        samples=item.samples,
        folds=item.folds,
    )

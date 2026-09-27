"""Per-fold error aggregates, defined exactly as the golden evaluation gate defines them.

``mae`` and ``wape`` are derived from three summed quantities, so a horizon aggregate is
the sum of its folds' quantities rather than an average of averages. A fold whose labels
sum to zero has no defined WAPE: the ratio is reported as ``None``, never as ``0.0`` and
never as an invented epsilon, which is the same missing-is-not-zero rule the feature
layer applies.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tramflow_ml.backtest.records import BacktestError


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    """Summed quantities plus the ratios derived from them."""

    scored: int
    actual_total: float
    error_total: float

    def __post_init__(self) -> None:
        if self.scored < 0:
            raise BacktestError("scored observations cannot be negative")
        for name in ("actual_total", "error_total"):
            value: float = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise BacktestError(f"{name} must be finite and non-negative")
        if self.scored == 0 and (self.actual_total or self.error_total):
            raise BacktestError("nothing scored cannot carry a total")

    @property
    def mae(self) -> float | None:
        return None if self.scored == 0 else self.error_total / self.scored

    @property
    def wape(self) -> float | None:
        """Undefined when a slice carries no demand at all, as in ``evaluation.py``."""
        return None if self.actual_total == 0 else self.error_total / self.actual_total

    def to_dict(self) -> dict[str, object]:
        return {
            "scored": self.scored,
            "actual_total": self.actual_total,
            "error_total": self.error_total,
            "mae": self.mae,
            "wape": self.wape,
        }


def fold_metrics(actual: Sequence[float], predicted: Sequence[float]) -> FoldMetrics:
    """Summed in the row order of the feature table, so the float result is reproducible."""
    if len(actual) != len(predicted):
        raise BacktestError("predictions must align one to one with the test rows")
    actual_total = 0.0
    error_total = 0.0
    for observed, forecast in zip(actual, predicted, strict=True):
        actual_total += observed
        error_total += abs(observed - forecast)
    return FoldMetrics(len(actual), actual_total, error_total)


def combine(parts: Iterable[FoldMetrics]) -> FoldMetrics:
    """Pool folds by their summed quantities; an average of averages would hide a slice."""
    scored = 0
    actual_total = 0.0
    error_total = 0.0
    for part in parts:
        scored += part.scored
        actual_total += part.actual_total
        error_total += part.error_total
    return FoldMetrics(scored, actual_total, error_total)

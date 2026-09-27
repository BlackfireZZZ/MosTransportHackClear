"""Provisional gate thresholds.

Not one of these numbers is certified. They were chosen so the mechanism can be
demonstrated on synthetic points; agreeing them against organizer data is TASK-051.
Every threshold is named here, carried on the report, and printed beside the observed
value it judged, so a reader can tell a breach from a badly chosen constant.
"""

import math
from dataclasses import dataclass

MIN_SAMPLES_FOR_GATE = 12
MIN_FOLDS_FOR_GATE = 2
MAX_WAPE = 0.40
MAX_WAPE_RATIO_TO_BASELINE = 1.0
MAX_COVERAGE_SHORTFALL = 0.10
MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL = 2.0
MIN_INTERVAL_LEVEL = 0.8
MIN_INTERVAL_SCORE_ALLOWANCE = 8.0

THRESHOLDS_CERTIFIED = False


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """What the gate compares against. Defaults are the provisional constants above."""

    min_samples: int = MIN_SAMPLES_FOR_GATE
    min_folds: int = MIN_FOLDS_FOR_GATE
    max_wape: float = MAX_WAPE
    max_wape_ratio_to_baseline: float = MAX_WAPE_RATIO_TO_BASELINE
    max_coverage_shortfall: float = MAX_COVERAGE_SHORTFALL
    max_interval_score_to_mean_actual: float = MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL
    min_interval_level: float = MIN_INTERVAL_LEVEL
    min_interval_score_allowance: float = MIN_INTERVAL_SCORE_ALLOWANCE

    def __post_init__(self) -> None:
        for name in ("min_samples", "min_folds"):
            count: int = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{name} must be an integer of at least 1")
        for name in (
            "max_wape",
            "max_wape_ratio_to_baseline",
            "max_coverage_shortfall",
            "max_interval_score_to_mean_actual",
            "min_interval_score_allowance",
        ):
            value: float = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.min_interval_level < 1:
            raise ValueError("min_interval_level must lie strictly between 0 and 1")
        if self.max_coverage_shortfall >= self.min_interval_level:
            raise ValueError(
                "max_coverage_shortfall must be below min_interval_level, or the coverage "
                "requirement it produces would be zero or negative and judge nothing"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "certified": THRESHOLDS_CERTIFIED,
            "MIN_SAMPLES_FOR_GATE": self.min_samples,
            "MIN_FOLDS_FOR_GATE": self.min_folds,
            "MAX_WAPE": self.max_wape,
            "MAX_WAPE_RATIO_TO_BASELINE": self.max_wape_ratio_to_baseline,
            "MAX_COVERAGE_SHORTFALL": self.max_coverage_shortfall,
            "MAX_INTERVAL_SCORE_TO_MEAN_ACTUAL": self.max_interval_score_to_mean_actual,
            "MIN_INTERVAL_LEVEL": self.min_interval_level,
            "MIN_INTERVAL_SCORE_ALLOWANCE": self.min_interval_score_allowance,
        }

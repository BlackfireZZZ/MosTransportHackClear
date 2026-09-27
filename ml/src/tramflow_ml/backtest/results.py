"""What a backtest returns, including the states in which it must not claim success.

``passed`` here is a statement about the *fold set*, not about model quality: it says
enough independent origins existed and every one of them was scored. Whether a candidate
beats its baseline is the reporting task's question, and a caller must not read this flag
as an answer to it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from tramflow_ml.backtest.folds import FoldSet
from tramflow_ml.backtest.manifest import ExperimentManifest, ModelVersion
from tramflow_ml.backtest.metrics import FoldMetrics
from tramflow_ml.backtest.records import BacktestError
from tramflow_ml.features import Horizon

BacktestStatus = Literal["evaluated", "insufficient_history", "insufficient_signal"]
PASSING_STATUS: BacktestStatus = "evaluated"


@dataclass(frozen=True, slots=True)
class FoldResult:
    """One model's predictions on one fold, reduced to a digest and error totals."""

    fold_id: str
    model: ModelVersion
    rows: int
    predictions_digest: str
    metrics: FoldMetrics

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise BacktestError(f"{self.fold_id}: a scored fold must have rows")
        if self.rows != self.metrics.scored:
            raise BacktestError(f"{self.fold_id}: every test row of an eligible fold is scored")

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "model": self.model.to_dict(),
            "rows": self.rows,
            "predictions_digest": self.predictions_digest,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HorizonOutcome:
    """One horizon's verdict, and the counts behind it.

    An ``evaluated`` outcome with too few folds is unconstructible, so the worst failure
    mode -- a run that scored nothing and reported success -- cannot be reached by any
    code path, not merely by any tested one.
    """

    horizon: Horizon
    status: BacktestStatus
    required_origins: int
    required_origins_with_demand: int
    eligible_origins: int
    origins_with_demand: int
    reason: str
    fold_ids: tuple[str, ...]
    results: Mapping[str, tuple[FoldResult, ...]]
    totals: Mapping[str, FoldMetrics]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(self, "totals", MappingProxyType(dict(self.totals)))
        if self.required_origins < 1 or self.required_origins_with_demand < 1:
            raise BacktestError("a horizon must require at least one origin")
        if len(self.fold_ids) != self.eligible_origins:
            raise BacktestError(f"{self.horizon}: fold ids must match the eligible origin count")
        if self.status != PASSING_STATUS:
            if not self.reason:
                raise BacktestError(f"{self.horizon}: a non-passing outcome must state why")
            return
        self._validate_passing()

    def _validate_passing(self) -> None:
        if self.reason:
            raise BacktestError(f"{self.horizon}: a passing outcome carries no refusal reason")
        if self.eligible_origins < self.required_origins:
            raise BacktestError(
                f"{self.horizon}: {self.eligible_origins} eligible origins cannot pass a "
                f"requirement of {self.required_origins}"
            )
        if self.origins_with_demand < self.required_origins_with_demand:
            raise BacktestError(
                f"{self.horizon}: only {self.origins_with_demand} folds carry demand, "
                f"{self.required_origins_with_demand} required"
            )
        self._validate_scored()

    def _validate_scored(self) -> None:
        """A passing horizon scored every eligible fold, for every model it compared.

        Without this the class would still permit the failure it exists to prevent, just
        through a hand-built outcome rather than through the runner: results present but
        empty, and ``passed`` True over nothing.
        """
        if not self.results:
            raise BacktestError(f"{self.horizon}: a passing outcome must carry model results")
        for name, scored in self.results.items():
            if len(scored) != self.eligible_origins:
                raise BacktestError(
                    f"{self.horizon}: {name} scored {len(scored)} folds, "
                    f"{self.eligible_origins} eligible"
                )

    @property
    def passed(self) -> bool:
        return self.status == PASSING_STATUS

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "status": self.status,
            "passed": self.passed,
            "required_origins": self.required_origins,
            "required_origins_with_demand": self.required_origins_with_demand,
            "eligible_origins": self.eligible_origins,
            "origins_with_demand": self.origins_with_demand,
            "reason": self.reason,
            "fold_ids": list(self.fold_ids),
            "models": [
                {
                    "model": name,
                    "totals": self.totals[name].to_dict(),
                    "folds": [result.to_dict() for result in self.results[name]],
                }
                for name in sorted(self.results)
            ],
        }


@dataclass(frozen=True, slots=True)
class BacktestOutcome:
    """Every horizon's outcome, the fold set they shared, and the manifest."""

    horizons: Mapping[Horizon, HorizonOutcome]
    fold_set: FoldSet
    manifest: ExperimentManifest

    def __post_init__(self) -> None:
        if not self.horizons:
            raise BacktestError("a backtest outcome must cover at least one horizon")
        object.__setattr__(self, "horizons", MappingProxyType(dict(self.horizons)))

    @property
    def passed(self) -> bool:
        return all(outcome.passed for outcome in self.horizons.values())

    @property
    def reasons(self) -> tuple[str, ...]:
        """Every refusal, in configured horizon order, for a caller that prints one line."""
        return tuple(outcome.reason for outcome in self.horizons.values() if not outcome.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "manifest": self.manifest.to_dict(),
            "horizons": [outcome.to_dict() for outcome in self.horizons.values()],
        }

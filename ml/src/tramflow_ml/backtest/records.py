"""Immutable fold records. A fold whose windows overlap cannot be constructed."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from tramflow_ml.features import Horizon
from tramflow_ml.features.records import require_aware

BACKTEST_VERSION = "backtest.v1"

IneligibleReason = Literal[
    "horizon_beyond_data",
    "incomplete_labels",
    "diluted_labels",
    "insufficient_train_history",
]


class BacktestError(ValueError):
    """Configuration or input defect that makes a backtest unsafe to run."""


@dataclass(frozen=True, slots=True)
class Window:
    """Half-open ``[start, end)`` interval of instants; an empty window is legal.

    The train window of the earliest origins is legitimately empty, so emptiness is a
    value to be counted and refused by a rule, not an error at construction.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        require_aware(self.start, "window start")
        require_aware(self.end, "window end")
        if self.end.astimezone(UTC) < self.start.astimezone(UTC):
            raise BacktestError("a window cannot end before it starts")

    @property
    def is_empty(self) -> bool:
        return self.start.astimezone(UTC) == self.end.astimezone(UTC)

    def contains(self, instant: datetime) -> bool:
        """Half-open membership by instant, so two spellings of one instant agree."""
        moment = require_aware(instant, "instant").astimezone(UTC)
        return self.start.astimezone(UTC) <= moment < self.end.astimezone(UTC)

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True)
class Fold:
    """One rolling origin with its train, validation and test windows.

    ``cutoff`` is the instant after which no information may enter a feature of this
    fold. It is produced by the feature layer's own ``HorizonPolicy.cutoff`` and is
    carried here only so the fold can be described and hashed.
    """

    fold_id: str
    index: int
    horizon: Horizon
    policy_name: str
    origin: datetime
    cutoff: datetime
    train: Window
    validation: Window
    test: Window
    train_buckets: int
    train_covered_buckets: int
    test_buckets: int
    label_covered_units: int
    label_total_units: int

    def __post_init__(self) -> None:
        self._validate_counts()
        order = (
            self.train.start,
            self.train.end,
            self.validation.start,
            self.validation.end,
            self.cutoff,
            self.origin,
        )
        moments = [moment.astimezone(UTC) for moment in order]
        if moments != sorted(moments):
            raise BacktestError(
                f"{self.fold_id}: train, validation, cutoff and origin must not overlap"
            )
        if self.test.start.astimezone(UTC) != self.origin.astimezone(UTC):
            raise BacktestError(f"{self.fold_id}: the test window must start at the origin")
        if self.test.is_empty:
            raise BacktestError(f"{self.fold_id}: the test window must not be empty")

    def _validate_counts(self) -> None:
        if self.index < 0:
            raise BacktestError("fold index must not be negative")
        if self.train_buckets < 0 or self.test_buckets < 1:
            raise BacktestError(f"{self.fold_id}: implausible bucket counts")
        if not 0 <= self.train_covered_buckets <= self.train_buckets:
            raise BacktestError(f"{self.fold_id}: covered train buckets must lie within [0, n]")
        if self.label_total_units < 1:
            raise BacktestError(f"{self.fold_id}: a horizon spans at least one civil date")
        if not 0 <= self.label_covered_units <= self.label_total_units:
            raise BacktestError(f"{self.fold_id}: label units must lie within [0, total]")

    @property
    def label_unit_ratio(self) -> float:
        """Fraction of the horizon's civil dates the source covers; 1/29 is not 29/29."""
        return self.label_covered_units / self.label_total_units

    def view(self) -> "FoldView":
        """The part of this fold a model may see.

        The label-unit counts are left behind. They describe how much of the *horizon*
        the source covers, which is a statement about the period being predicted; with
        ``minimum_label_unit_ratio`` below 1.0 it is a statement a forecaster standing at
        the cutoff could not make. Everything else here is either past or is the question
        being asked.
        """
        return FoldView(
            fold_id=self.fold_id,
            index=self.index,
            horizon=self.horizon,
            policy_name=self.policy_name,
            origin=self.origin,
            cutoff=self.cutoff,
            train=self.train,
            validation=self.validation,
            test=self.test,
            train_buckets=self.train_buckets,
            train_covered_buckets=self.train_covered_buckets,
            test_buckets=self.test_buckets,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "index": self.index,
            "horizon": self.horizon,
            "policy": self.policy_name,
            "origin": self.origin.isoformat(),
            "cutoff": self.cutoff.isoformat(),
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "train_buckets": self.train_buckets,
            "train_covered_buckets": self.train_covered_buckets,
            "test_buckets": self.test_buckets,
            "label_covered_units": self.label_covered_units,
            "label_total_units": self.label_total_units,
        }


@dataclass(frozen=True, slots=True)
class Ineligible:
    """A candidate origin that is not a fold, and why.

    A refusal that is not reported is indistinguishable from an origin nobody considered,
    so every candidate the enumeration produced is recorded either as a fold or here.
    """

    horizon: Horizon
    origin: datetime
    reason: IneligibleReason
    detail: str

    def __post_init__(self) -> None:
        require_aware(self.origin, "origin")
        if not self.detail:
            raise BacktestError("a refusal must state why")

    def to_dict(self) -> dict[str, str]:
        return {
            "horizon": self.horizon,
            "origin": self.origin.isoformat(),
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FoldView:
    """A fold as a model sees it: the windows, the cutoff, and the history behind it.

    Built by ``Fold.view()`` and never constructed directly by the runner, so a field
    added to ``Fold`` does not reach a model until someone adds it here on purpose.
    """

    fold_id: str
    index: int
    horizon: Horizon
    policy_name: str
    origin: datetime
    cutoff: datetime
    train: Window
    validation: Window
    test: Window
    train_buckets: int
    train_covered_buckets: int
    test_buckets: int

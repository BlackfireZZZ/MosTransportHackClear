"""Fold rules and the experiment configuration they hash into.

Every default here is configuration, not a certified value, and every resolved default
reaches the config hash so a manifest names the rules that actually ran rather than the
rules someone meant to run.
"""

import math
from dataclasses import dataclass, replace

from tramflow_ml.backtest.records import BacktestError
from tramflow_ml.features import FEATURE_VERSION, Horizon, HorizonPolicy, validate_count_pair
from tramflow_ml.features.records import digest_of

BACKTEST_CONFIG_VERSION = "backtest-config.v1"
DEFAULT_EMBARGO_BUCKETS = 0
DEFAULT_MINIMUM_ORIGINS = 3
DEFAULT_MINIMUM_LABEL_UNIT_RATIO = 1.0
DEFAULT_ORIGIN_STRIDE_PERIODS = 1
DEFAULT_VALIDATION_PERIODS = 1


def policy_lookback_buckets(policy: HorizonPolicy) -> int:
    """How many *covered* buckets the policy's own columns need to be non-missing.

    Derived from the policy rather than restated, so adding a longer lag automatically
    lengthens the history a fold is required to stand on. It is a count of buckets that
    carry data, not a calendar distance: a train window long enough on the calendar but
    riddled with coverage holes leaves every lag, window and season missing, and a fold
    standing on that is a false pass rather than a short one.
    """
    return max(
        max(policy.lags),
        max(policy.rolling_windows),
        policy.season_step * policy.season_periods,
    )


@dataclass(frozen=True, slots=True)
class FoldRules:
    """When a rolling origin of one horizon becomes a fold.

    ``embargo_buckets`` moves the cutoff earlier than the origin, on top of the policy's
    own ``cutoff_lead_buckets``. It defaults to 0 because in this pipeline no information
    crosses the boundary that the cutoff does not already stop: labels are per-bucket
    counts with no smoothing, a history bucket is used only once it has fully elapsed,
    and late publication is modelled by the policy lead and by the coverage calendar's
    publication instants. Raise it as soon as a target is defined over a window rather
    than a bucket, or an organizer source revises already-published dates.

    ``minimum_train_buckets`` defaults to the policy's own look-back and counts buckets
    the coverage calendar actually covers, not calendar distance.

    ``minimum_origins_with_demand`` defaults to ``minimum_origins``. It is separate
    because the two gates answer different questions -- how many independent origins
    exist, and how many of them carry any demand to score -- and raising one to get more
    folds should not silently tighten the other.
    """

    policy: HorizonPolicy
    embargo_buckets: int = DEFAULT_EMBARGO_BUCKETS
    minimum_origins: int = DEFAULT_MINIMUM_ORIGINS
    minimum_origins_with_demand: int | None = None
    minimum_train_buckets: int | None = None
    minimum_label_unit_ratio: float = DEFAULT_MINIMUM_LABEL_UNIT_RATIO
    origin_stride_periods: int = DEFAULT_ORIGIN_STRIDE_PERIODS
    validation_periods: int = DEFAULT_VALIDATION_PERIODS

    def __post_init__(self) -> None:
        self._validate_counts()
        ratio = self.minimum_label_unit_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, int | float):
            raise BacktestError("minimum_label_unit_ratio must be a number")
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise BacktestError("minimum_label_unit_ratio must be finite and within [0, 1]")
        object.__setattr__(self, "minimum_label_unit_ratio", float(ratio))

    def _validate_counts(self) -> None:
        if type(self.embargo_buckets) is not int or self.embargo_buckets < 0:
            raise BacktestError("embargo_buckets must be an integer >= 0")
        if type(self.minimum_origins) is not int or self.minimum_origins < 1:
            raise BacktestError(
                "minimum_origins must be an integer >= 1; a backtest that may evaluate "
                "zero folds and still pass is the failure this rule exists to prevent"
            )
        for name in ("origin_stride_periods", "validation_periods"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise BacktestError(f"{name} must be an integer >= 1")
        required = self.minimum_train_buckets
        if required is not None and (type(required) is not int or required < 0):
            raise BacktestError("minimum_train_buckets must be an integer >= 0 or None")
        demanded = self.minimum_origins_with_demand
        if demanded is not None and (type(demanded) is not int or demanded < 1):
            raise BacktestError("minimum_origins_with_demand must be an integer >= 1 or None")

    @property
    def horizon(self) -> Horizon:
        return self.policy.horizon

    @property
    def origins_with_demand_required(self) -> int:
        if self.minimum_origins_with_demand is not None:
            return self.minimum_origins_with_demand
        return self.minimum_origins

    @property
    def train_buckets_required(self) -> int:
        if self.minimum_train_buckets is not None:
            return self.minimum_train_buckets
        return policy_lookback_buckets(self.policy)

    @property
    def effective_policy(self) -> HorizonPolicy:
        """The policy the feature layer is handed, with the embargo folded into its lead.

        The embargo is applied this way rather than by recomputing a cutoff so that the
        one definition of the availability cutoff stays in ``HorizonPolicy.cutoff`` and
        the two cannot drift apart.
        """
        lead = self.policy.cutoff_lead_buckets + self.embargo_buckets
        return replace(self.policy, cutoff_lead_buckets=lead)

    def to_dict(self) -> dict[str, object]:
        policy = self.policy
        return {
            "policy": policy.name,
            "horizon": policy.horizon,
            "granularity": policy.granularity,
            "lags": list(policy.lags),
            "rolling_windows": list(policy.rolling_windows),
            "season_step": policy.season_step,
            "season_periods": policy.season_periods,
            "cutoff_lead_buckets": policy.cutoff_lead_buckets,
            "embargo_buckets": self.embargo_buckets,
            "minimum_origins": self.minimum_origins,
            "minimum_origins_with_demand": self.origins_with_demand_required,
            "minimum_train_buckets": self.train_buckets_required,
            "minimum_label_unit_ratio": self.minimum_label_unit_ratio,
            "origin_stride_periods": self.origin_stride_periods,
            "validation_periods": self.validation_periods,
        }


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """One experiment: the rules for each horizon and the target they all score."""

    rules: tuple[FoldRules, ...]
    target: str
    unit: str

    def __post_init__(self) -> None:
        if not self.rules:
            raise BacktestError("a backtest configuration needs at least one horizon")
        horizons = [rule.horizon for rule in self.rules]
        if len(set(horizons)) != len(horizons):
            raise BacktestError("each horizon may appear at most once in a configuration")
        validate_count_pair(self.target, self.unit)

    def rule_for(self, horizon: Horizon) -> FoldRules:
        for rule in self.rules:
            if rule.horizon == horizon:
                return rule
        raise BacktestError(f"no rules configured for the {horizon} horizon")

    @property
    def horizons(self) -> tuple[Horizon, ...]:
        return tuple(rule.horizon for rule in self.rules)

    def to_dict(self) -> dict[str, object]:
        return {
            "backtest_version": BACKTEST_CONFIG_VERSION,
            "feature_version": FEATURE_VERSION,
            "target": self.target,
            "unit": self.unit,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @property
    def config_hash(self) -> str:
        return digest_of(self.to_dict())

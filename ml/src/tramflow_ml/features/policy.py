"""The three horizon policies and the availability cutoff each one derives."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from tramflow_ml.features.periods import granularity_for, step, validate_origin
from tramflow_ml.features.records import (
    GRANULARITY_SUFFIX,
    FeatureError,
    Granularity,
    Horizon,
)


@dataclass(frozen=True, slots=True)
class HorizonPolicy:
    """Lags, windows and the cutoff for one horizon; every step is whole buckets.

    ``cutoff_lead_buckets`` moves the cutoff earlier than the forecast origin, for a
    source that publishes late. The lags, windows and season are the team proposal and
    are configuration, not certified values.
    """

    name: str
    horizon: Horizon
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    season_step: int
    season_periods: int
    cutoff_lead_buckets: int = 0

    def __post_init__(self) -> None:
        for label, steps in (("lags", self.lags), ("rolling_windows", self.rolling_windows)):
            if not steps:
                raise FeatureError(f"{self.name}: {label} must not be empty")
            if any(type(value) is not int or value < 1 for value in steps):
                raise FeatureError(f"{self.name}: {label} must be integers >= 1")
            if list(steps) != sorted(set(steps)):
                raise FeatureError(f"{self.name}: {label} must be sorted and unique")
        if self.season_step < 1 or self.season_periods < 1:
            raise FeatureError(f"{self.name}: season step and periods must be >= 1")
        if self.cutoff_lead_buckets < 0:
            raise FeatureError(f"{self.name}: cutoff lead must not be negative")

    @property
    def granularity(self) -> Granularity:
        return granularity_for(self.horizon)

    @property
    def suffix(self) -> str:
        return GRANULARITY_SUFFIX[self.granularity]

    def cutoff(self, origin: datetime) -> datetime:
        """No information after this instant may reach a feature of this horizon."""
        local = validate_origin(origin, self.horizon)
        return step(local, self.granularity, -self.cutoff_lead_buckets)


DAY_HOUR = HorizonPolicy(
    name="day/hour",
    horizon="day",
    lags=(1, 2, 24, 48, 168),
    rolling_windows=(24, 168),
    season_step=24,
    season_periods=4,
)
MONTH_DAY = HorizonPolicy(
    name="month/day",
    horizon="month",
    lags=(1, 7, 14, 364),
    rolling_windows=(7, 28),
    season_step=7,
    season_periods=4,
)
YEAR_MONTH = HorizonPolicy(
    name="year/month",
    horizon="year",
    lags=(1, 2, 3, 12),
    rolling_windows=(3, 12),
    season_step=12,
    season_periods=3,
)

POLICIES: Mapping[str, HorizonPolicy] = MappingProxyType(
    {policy.name: policy for policy in (DAY_HOUR, MONTH_DAY, YEAR_MONTH)}
)


def policy_by_name(name: str) -> HorizonPolicy:
    try:
        return POLICIES[name]
    except KeyError as error:
        raise FeatureError(f"unknown policy {name!r}; expected {sorted(POLICIES)}") from error

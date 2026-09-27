"""Observable coverage and model-capability limits at one forecast origin."""

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.decision_guard import ShiftSignal


@dataclass(frozen=True)
class LimitFlag:
    code: Literal[
        "unseen_target_month", "no_observed_route_demand", "trend_input_clipped",
        "ml_bound_below_observed_change",
    ]
    entity: str
    observed: float | None = None
    boundary: float | None = None

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


def assess_limits(
    history: pd.DataFrame, origin: date, horizon_days: int,
    signals: tuple[ShiftSignal, ...], capability: dict[str, object],
) -> tuple[LimitFlag, ...]:
    """Describe known gaps without turning a risk flag into an accuracy claim."""
    if horizon_days < 1 or history.empty or (history.date >= origin).any():
        raise ValueError("limit assessment needs pre-origin history and a positive horizon")
    flags: list[LimitFlag] = []
    seen_months = {day.month for day in history.date.unique()}
    target_months = {
        (origin + timedelta(days=offset)).month for offset in range(horizon_days)
    }
    flags.extend(
        LimitFlag("unseen_target_month", str(month))
        for month in sorted(target_months - seen_months)
    )
    route_totals = history.groupby("route", observed=True).boardings.sum()
    flags.extend(
        LimitFlag("no_observed_route_demand", str(route))
        for route, total in route_totals.items() if total == 0
    )
    clip = capability.get("trend_input_ratio_clip")
    floor: float | None = None
    ceiling: float | None = None
    if isinstance(clip, list) and len(clip) == 2:
        floor, ceiling = float(clip[0]), float(clip[1])
    ml_bound = capability.get("max_ml_relative_change")
    for signal in signals:
        if signal.kind != "demand_level":
            continue
        if floor is not None and ceiling is not None:
            limit = ceiling if signal.direction == "up" else floor
            if signal.value > ceiling or signal.value < floor:
                flags.append(LimitFlag("trend_input_clipped", signal.entity, signal.value, limit))
        if isinstance(ml_bound, (int, float)) and abs(signal.value - 1) > ml_bound:
            flags.append(LimitFlag(
                "ml_bound_below_observed_change", signal.entity,
                abs(signal.value - 1), float(ml_bound),
            ))
    return tuple(flags)

"""Cutoff-safe robust route-hour profiles for direct 61-day forecasts."""

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.competition import calendar_weekday_blend_predict
from tramflow_ml.route_models.features import OFF, WORK

RobustModel = Literal["calendar_robust_short", "calendar_robust_28", "calendar_robust_blend"]

ROBUST_CONFIG = {
    "calendar_robust_short": {
        "trend_window_days": 14,
        "trend_strength": 0.3,
        "trend_half_life_days": 42,
        "median_weight": 0.25,
    },
    "calendar_robust_28": {
        "trend_window_days": 28,
        "trend_strength": 0.3,
        "trend_half_life_days": 84,
        "median_weight": 0.5,
    },
}
HOUR_SHAPE_CONFIG = {
    "recent_days": 28,
    "previous_days": 56,
    "ratio_floor": 0.75,
    "ratio_ceiling": 1.25,
    "strength": 0.2,
}
ROUTE_SHAPE_CONFIG = {**HOUR_SHAPE_CONFIG, "group": "route,hour"}
TREND_RATIO_FLOOR = 0.75
TREND_RATIO_CEILING = 1.25


def _day_type(day: date) -> int:
    if day in WORK or (day.weekday() < 5 and day not in OFF):
        return 0
    if day.weekday() == 5 and day not in OFF:
        return 1
    return 2


def _group_median(
    history: pd.DataFrame, target: pd.DataFrame, keys: list[str]
) -> np.ndarray:
    grouped = history.groupby(keys, observed=True).boardings.median()
    return np.asarray(pd.MultiIndex.from_frame(target[keys]).map(grouped), dtype=float)


def robust_profile_predict(
    history: pd.DataFrame, future: pd.DataFrame, model: str
) -> np.ndarray:
    """Use only history before the forecast origin; future dates supply calendar keys."""
    if model == "calendar_robust_blend":
        short = robust_profile_predict(history, future, "calendar_robust_short")
        long = robust_profile_predict(history, future, "calendar_robust_28")
        return np.asarray(0.5 * short + 0.5 * long, dtype=float)
    if model not in ROBUST_CONFIG:
        raise ValueError(f"unknown robust profile {model}")
    origin = min(future.date)
    if history.empty or (history.date >= origin).any():
        raise ValueError("future labels are not allowed in robust profile history")
    config = ROBUST_CONFIG[model]
    past = history[["route", "date", "hour", "boardings"]].copy()
    past["dow"] = pd.to_datetime(past.date).dt.dayofweek
    past["season"] = pd.to_datetime(past.date).dt.month.isin((6, 7, 8)).astype(int)
    past["type"] = past.date.map(_day_type)
    target = future[["route", "date", "hour"]].copy()
    target["dow"] = pd.to_datetime(target.date).dt.dayofweek
    target["season"] = pd.to_datetime(target.date).dt.month.isin((6, 7, 8)).astype(int)
    target["type"] = target.date.map(_day_type)

    baseline = calendar_weekday_blend_predict(future)
    weekday = _group_median(past, target, ["route", "season", "dow", "hour"])
    day_type = _group_median(past, target, ["route", "season", "type", "hour"])
    exceptions = future.calendar_exception.to_numpy(dtype=bool)
    median = np.where(exceptions, day_type, 0.5 * weekday + 0.5 * day_type)
    median = np.where(np.isfinite(median), median, baseline)

    window = int(config["trend_window_days"])
    recent = past.loc[past.date >= origin - timedelta(days=window)]
    previous = past.loc[
        (past.date < origin - timedelta(days=window))
        & (past.date >= origin - timedelta(days=window * 3))
    ]
    recent_mean = recent.groupby(["route", "type"], observed=True).boardings.mean()
    previous_mean = previous.groupby(["route", "type"], observed=True).boardings.mean()
    ratio = (recent_mean / previous_mean.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(1).clip(TREND_RATIO_FLOOR, TREND_RATIO_CEILING)
    target_ratio = np.asarray(
        pd.MultiIndex.from_frame(target[["route", "type"]]).map(ratio), dtype=float
    )
    target_ratio = np.nan_to_num(target_ratio, nan=1.0)
    lead = future.lead_day.to_numpy(dtype=float)
    decay = np.exp(-np.log(2) * (lead - 1) / float(config["trend_half_life_days"]))
    adjusted = baseline * (
        1 + float(config["trend_strength"]) * decay * (target_ratio - 1)
    ) + float(config["median_weight"]) * (median - baseline)
    return np.asarray(np.maximum(0, adjusted), dtype=float)


def _hour_shares(rows: pd.DataFrame, group: Literal["route", "day_type"]) -> pd.Series:
    keys = ["route"] if group == "route" else ["route", "type"]
    hourly = rows.groupby([*keys, "hour"], observed=True).boardings.sum()
    totals = rows.groupby(keys, observed=True).boardings.sum()
    return hourly / totals.reindex(hourly.index.droplevel("hour")).to_numpy(dtype=float)


def robust_hour_shape_predict(
    history: pd.DataFrame, future: pd.DataFrame,
    base_model: RobustModel = "calendar_robust_blend",
    group: Literal["route", "day_type"] = "day_type",
) -> np.ndarray:
    """Adjust recent hourly shares while conserving every predicted route-day total."""
    baseline = robust_profile_predict(history, future, base_model)
    origin = min(future.date)
    past = history[["route", "date", "hour", "boardings"]].copy()
    past["type"] = past.date.map(_day_type)
    recent_days = int(HOUR_SHAPE_CONFIG["recent_days"])
    previous_days = int(HOUR_SHAPE_CONFIG["previous_days"])
    recent = past.loc[past.date >= origin - timedelta(days=recent_days)]
    previous = past.loc[
        (past.date < origin - timedelta(days=recent_days))
        & (past.date >= origin - timedelta(days=recent_days + previous_days))
    ]
    ratio = (
        _hour_shares(recent, group) / _hour_shares(previous, group).replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf], np.nan
    ).clip(float(HOUR_SHAPE_CONFIG["ratio_floor"]), float(HOUR_SHAPE_CONFIG["ratio_ceiling"]))
    keys = future[["route", "date", "hour"]].copy()
    shape_keys = ["route", "hour"]
    if group == "day_type":
        keys["type"] = keys.date.map(_day_type)
        shape_keys = ["route", "type", "hour"]
    mapped = np.asarray(
        pd.MultiIndex.from_frame(keys[shape_keys]).map(ratio), dtype=float
    )
    mapped = np.nan_to_num(mapped, nan=1.0)
    shifted = baseline * (1 + float(HOUR_SHAPE_CONFIG["strength"]) * (mapped - 1))
    daily_keys = [keys.route, keys.date]
    shifted_daily = pd.Series(shifted).groupby(daily_keys).transform("sum").to_numpy()
    baseline_daily = pd.Series(baseline).groupby(daily_keys).transform("sum").to_numpy()
    return np.asarray(
        np.divide(
            shifted * baseline_daily, shifted_daily,
            out=baseline.copy(), where=shifted_daily > 0,
        ), dtype=float,
    )

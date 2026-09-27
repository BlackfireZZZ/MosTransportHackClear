"""Cutoff-safe route-day CatBoost corrections around the robust hourly anchor."""

from datetime import timedelta
from typing import Literal, Protocol

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.competition import make_training_frame
from tramflow_ml.competition_profile import robust_hour_shape_predict

HybridModel = Literal["catboost_daily_bounded_50", "catboost_daily_recency_ensemble_50"]
HYBRID_CONFIG = {
    "loss_function": "MAE",
    "iterations": 400,
    "depth": 4,
    "learning_rate": 0.04,
    "l2_leaf_reg": 10,
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
    "cat_features": ["route"],
}
FIELDS = [
    "route", "day_of_week", "month", "day_of_year", "is_summer", "lead_day",
    "route_day_28", "route_day_56", "history_days", "calendar_day_type", "anchor_day",
]
CORRECTION_CAP = 0.05
RECENCY_HALF_LIFE_DAYS = 60


class DailyPredictor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


def _daily(rows: pd.DataFrame, anchor: np.ndarray, *, observed: bool) -> pd.DataFrame:
    frame = rows[[*FIELDS[:-1], "date", *(["boardings"] if observed else [])]].copy()
    frame["anchor_day"] = anchor
    reductions = {field: "first" for field in FIELDS if field not in ("route", "lead_day")}
    reductions["anchor_day"] = "sum"
    if observed:
        reductions["boardings"] = "sum"
    return frame.groupby(["route", "date", "lead_day"], sort=False, as_index=False).agg(
        reductions
    )


def _inputs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FIELDS].copy()
    result["route"] = result.route.astype(int).astype(str)
    return result


def _origin_dates(rows: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(rows.date) - pd.to_timedelta(rows.lead_day.astype(int) - 1, unit="D")
    ).dt.date


def _daily_correction(
    model: DailyPredictor, future_daily: pd.DataFrame, future: pd.DataFrame
) -> np.ndarray:
    predicted = np.asarray(model.predict(_inputs(future_daily)), dtype=float)
    anchor_day = future_daily.anchor_day.to_numpy(dtype=float)
    relative = np.divide(
        predicted * (anchor_day + 300), anchor_day,
        out=np.zeros(len(future_daily)), where=anchor_day > 0,
    )
    keyed = future_daily[["route", "date", "lead_day"]].copy()
    keyed["relative"] = relative
    mapped = future[["route", "date", "lead_day"]].merge(
        keyed, on=["route", "date", "lead_day"], how="left", validate="many_to_one",
    )
    if not np.isfinite(mapped.relative.to_numpy(dtype=float)).all():
        raise ValueError("hybrid correction is missing or nonfinite")
    return mapped.relative.to_numpy(dtype=float)


def bounded_hybrid(
    anchor: np.ndarray, uniform: np.ndarray, recency: np.ndarray | None = None
) -> np.ndarray:
    """The learned part changes a nonnegative anchor by at most 2.5%."""
    base = np.asarray(anchor, dtype=float)
    first = np.asarray(uniform, dtype=float)
    second = first if recency is None else np.asarray(recency, dtype=float)
    if base.shape != first.shape or base.shape != second.shape:
        raise ValueError("hybrid components must align")
    if not np.isfinite(base).all() or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("hybrid components must be finite")
    if (base < 0).any():
        raise ValueError("hybrid anchor must be nonnegative")
    uniform_ml = base * (1 + np.clip(first, -CORRECTION_CAP, CORRECTION_CAP))
    recency_ml = base * (1 + np.clip(second, -CORRECTION_CAP, CORRECTION_CAP))
    return np.asarray(0.5 * base + 0.25 * uniform_ml + 0.25 * recency_ml, dtype=float)


def hybrid_predict(
    history: pd.DataFrame, future: pd.DataFrame, model_name: HybridModel
) -> np.ndarray:
    """Fit simulated-origin residuals and predict from history strictly before origin."""
    if model_name not in ("catboost_daily_bounded_50", "catboost_daily_recency_ensemble_50"):
        raise ValueError(f"unknown hybrid model {model_name}")
    origin = min(future.date)
    if history.empty or (history.date >= origin).any():
        raise ValueError("future labels are not allowed in hybrid history")
    from catboost import CatBoostRegressor  # type: ignore[import-not-found,import-untyped]

    train = make_training_frame(history, origin - timedelta(days=1))
    train_origins = _origin_dates(train)
    anchor_train = np.empty(len(train), dtype=float)
    for historical_origin in sorted(train_origins.unique()):
        mask = (train_origins == historical_origin).to_numpy()
        anchor_train[mask] = robust_hour_shape_predict(
            history.loc[history.date < historical_origin], train.loc[mask],
            "calendar_robust_28", group="route",
        )
    train_daily = _daily(train, anchor_train, observed=True)
    anchor_day = train_daily.anchor_day.to_numpy(dtype=float)
    target = np.clip(
        (train_daily.boardings.to_numpy(dtype=float) - anchor_day) / (anchor_day + 300),
        -1, 2,
    )
    x_train = _inputs(train_daily)
    uniform = CatBoostRegressor(**HYBRID_CONFIG)
    uniform.fit(x_train, target)
    recency = None
    if model_name == "catboost_daily_recency_ensemble_50":
        daily_origins = _origin_dates(train_daily)
        age = np.asarray([(origin - day).days for day in daily_origins], dtype=float)
        weights = np.exp(-np.log(2) * age / RECENCY_HALF_LIFE_DAYS)
        recency = CatBoostRegressor(**HYBRID_CONFIG)
        recency.fit(x_train, target, sample_weight=weights)
    anchor = robust_hour_shape_predict(history, future, "calendar_robust_28", group="route")
    future_daily = _daily(future, anchor, observed=False)
    uniform_correction = _daily_correction(uniform, future_daily, future)
    recency_correction = (
        _daily_correction(recency, future_daily, future) if recency is not None else None
    )
    return bounded_hybrid(anchor, uniform_correction, recency_correction)

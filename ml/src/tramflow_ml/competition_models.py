"""Offline candidates for the organizer's route-hour boarding score."""

from importlib.metadata import version
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]

from tramflow_ml.competition import (
    FEATURE_COLUMNS,
    WEATHER_COLUMNS,
    calendar_profile_predict,
    calendar_weekday_blend_predict,
    profile_predict,
)
from tramflow_ml.competition_hybrid import (
    CORRECTION_CAP,
    HYBRID_CONFIG,
    RECENCY_HALF_LIFE_DAYS,
    hybrid_predict,
)
from tramflow_ml.competition_profile import (
    HOUR_SHAPE_CONFIG,
    ROBUST_CONFIG,
    ROUTE_SHAPE_CONFIG,
    TREND_RATIO_CEILING,
    TREND_RATIO_FLOOR,
    robust_hour_shape_predict,
    robust_profile_predict,
)
from tramflow_ml.competition_ts import (
    DIRECT_FOREST_CONFIG,
    FOURIER_CONFIG,
    time_series_predict,
)
from tramflow_ml.route_models.features import CALENDAR_ISSUED, CALENDAR_SOURCE

ModelName = Literal[
    "profile", "calendar_profile", "calendar_weekday_blend", "calendar_robust_short",
    "calendar_robust_28", "calendar_robust_blend", "calendar_robust_hourshape",
    "calendar_robust_28_hourshape",
    "calendar_robust_routeshape", "calendar_robust_28_routeshape",
    "hgb", "catboost", "catboost_residual",
    "catboost_relative", "catboost_weather_relative", "catboost_daily_relative",
    "catboost_daily_bounded_50", "catboost_daily_recency_ensemble_50",
    "fourier_ridge_daily", "direct_extratrees_daily",
]
HGB_CONFIG = {
    "loss": "absolute_error",
    "max_iter": 250,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "early_stopping": False,
    "random_state": 42,
}
CATBOOST_CONFIG = {
    "loss_function": "MAE",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
    "cat_features": ["route"],
}
CATBOOST_RESIDUAL_CONFIG = {
    **CATBOOST_CONFIG,
    "iterations": 350,
    "depth": 4,
    "learning_rate": 0.03,
}
CATBOOST_RELATIVE_CONFIG = {
    **CATBOOST_CONFIG,
    "iterations": 700,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
}
CATBOOST_DAILY_CONFIG = {**CATBOOST_RELATIVE_CONFIG, "depth": 5}
DAILY_FIELDS = (
    "route", "day_of_week", "month", "day_of_year", "is_summer", "lead_day",
    "route_day_28", "route_day_56", "history_days", "profile_hour",
)


def model_spec(model_name: ModelName) -> dict[str, object]:
    if model_name in ("catboost_daily_bounded_50", "catboost_daily_recency_ensemble_50"):
        return {
            "name": model_name,
            "version": "route-day-bounded-residual-v1",
            "package_version": version("catboost"),
            "config": HYBRID_CONFIG,
            "anchor": "calendar_robust_28_routeshape",
            "correction_cap": CORRECTION_CAP,
            "max_ml_relative_change": 0.5 * CORRECTION_CAP,
            "trend_input_ratio_clip": [TREND_RATIO_FLOOR, TREND_RATIO_CEILING],
            "mix": (
                {"anchor": 0.5, "catboost_uniform": 0.5}
                if model_name == "catboost_daily_bounded_50"
                else {"anchor": 0.5, "catboost_uniform": 0.25, "catboost_recency": 0.25}
            ),
            "recency_half_life_days": (
                RECENCY_HALF_LIFE_DAYS
                if model_name == "catboost_daily_recency_ensemble_50" else None
            ),
            "training_origin_stride_days": 14,
            "relative_denominator_offset": 300,
        }
    if model_name in ("calendar_robust_routeshape", "calendar_robust_28_routeshape"):
        return {
            "name": model_name,
            "version": "calendar-robust-routeshape-v1",
            "config": ROUTE_SHAPE_CONFIG,
            "trend_input_ratio_clip": [TREND_RATIO_FLOOR, TREND_RATIO_CEILING],
            "base": (
                "calendar-robust-blend-v1" if model_name == "calendar_robust_routeshape"
                else "calendar-robust-profile-v1:calendar_robust_28"
            ),
            "calendar_source": CALENDAR_SOURCE,
            "calendar_issued": CALENDAR_ISSUED.isoformat(),
        }
    if model_name in ("calendar_robust_hourshape", "calendar_robust_28_hourshape"):
        return {
            "name": model_name,
            "version": "calendar-robust-hourshape-v1",
            "config": HOUR_SHAPE_CONFIG,
            "trend_input_ratio_clip": [TREND_RATIO_FLOOR, TREND_RATIO_CEILING],
            "base": (
                "calendar-robust-blend-v1" if model_name == "calendar_robust_hourshape"
                else "calendar-robust-profile-v1:calendar_robust_28"
            ),
            "calendar_source": CALENDAR_SOURCE,
            "calendar_issued": CALENDAR_ISSUED.isoformat(),
        }
    if model_name == "calendar_robust_blend":
        return {
            "name": model_name,
            "version": "calendar-robust-blend-v1",
            "config": {"calendar_robust_short": 0.5, "calendar_robust_28": 0.5},
            "trend_input_ratio_clip": [TREND_RATIO_FLOOR, TREND_RATIO_CEILING],
            "calendar_source": CALENDAR_SOURCE,
            "calendar_issued": CALENDAR_ISSUED.isoformat(),
        }
    if model_name in ("calendar_robust_short", "calendar_robust_28"):
        return {
            "name": model_name,
            "version": "calendar-robust-profile-v1",
            "config": ROBUST_CONFIG[model_name],
            "trend_input_ratio_clip": [TREND_RATIO_FLOOR, TREND_RATIO_CEILING],
            "base": "calendar-weekday-daytype-v1",
            "calendar_source": CALENDAR_SOURCE,
            "calendar_issued": CALENDAR_ISSUED.isoformat(),
        }
    if model_name == "profile":
        return {"name": model_name, "version": "seasonal-mean-v1",
                "config": {"summer_months": [6, 7, 8], "group": "route,weekday,hour"}}
    if model_name == "calendar_profile":
        return {
            "name": model_name,
            "version": "calendar-seasonal-mean-v1",
            "config": {"base": "seasonal-mean-v1", "exception_group": "route,day_type,hour"},
            "calendar_source": CALENDAR_SOURCE,
            "calendar_issued": CALENDAR_ISSUED.isoformat(),
        }
    if model_name == "calendar_weekday_blend":
        return {
            "name": model_name, "version": "calendar-weekday-daytype-v1",
            "config": {"weekday_weight": 0.5, "day_type_weight": 0.5,
                       "exception_model": "calendar-seasonal-mean-v1"},
            "calendar_source": CALENDAR_SOURCE,
            "calendar_issued": CALENDAR_ISSUED.isoformat(),
        }
    if model_name == "hgb":
        return {"name": model_name, "package_version": version("scikit-learn"),
                "config": HGB_CONFIG}
    if model_name in ("fourier_ridge_daily", "direct_extratrees_daily"):
        return {
            "name": model_name, "package_version": version("scikit-learn"),
            "version": "daily-total-hourly-profile-v1",
            "training_source": "dense route-date-hour labels before origin",
            "config": (
                FOURIER_CONFIG if model_name == "fourier_ridge_daily" else DIRECT_FOREST_CONFIG
            ),
            "calendar_source": CALENDAR_SOURCE,
            "calendar_issued": CALENDAR_ISSUED.isoformat(),
        }
    config = {
        "catboost": CATBOOST_CONFIG,
        "catboost_residual": CATBOOST_RESIDUAL_CONFIG,
        "catboost_relative": CATBOOST_RELATIVE_CONFIG,
        "catboost_weather_relative": CATBOOST_RELATIVE_CONFIG,
        "catboost_daily_relative": CATBOOST_DAILY_CONFIG,
    }[model_name]
    result: dict[str, object] = {
        "name": model_name, "package_version": version("catboost"), "config": config,
    }
    if model_name in ("catboost_relative", "catboost_weather_relative", "catboost_daily_relative"):
        result["profile_blend_weight"] = 0.25
        result["relative_scale"] = 300 if model_name == "catboost_daily_relative" else 30
    if model_name == "catboost_weather_relative":
        result["weather_source"] = "Open-Meteo historical actuals; spatial route-zone average"
        result["weather_mode"] = "retrospective_look_ahead"
    return result


def _daily_frame(rows: pd.DataFrame, profile: np.ndarray, *, observed: bool) -> pd.DataFrame:
    columns = [
        "route", "date", "lead_day", "day_of_week", "month", "day_of_year",
        "is_summer", "route_day_28", "route_day_56", "history_days",
    ]
    result = rows[columns + (["boardings"] if observed else [])].copy()
    result["profile_hour"] = profile
    reductions = {
        column: "first"
        for column in columns
        if column not in ("route", "date", "lead_day")
    }
    reductions["profile_hour"] = "sum"
    if observed:
        reductions["boardings"] = "sum"
    return result.groupby(["route", "date", "lead_day"], sort=False, as_index=False).agg(reductions)


def _daily_relative_predict(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    # catboost is the optional `boosting` extra and is absent when `make sync-ml`
    # installs `dev` alone, so the missing case needs covering too.
    from catboost import CatBoostRegressor  # type: ignore[import-not-found,import-untyped]

    train_profile = profile_predict(train)
    future_profile = profile_predict(future)
    daily_train = _daily_frame(train, train_profile, observed=True)
    daily_future = _daily_frame(future, future_profile, observed=False)
    x_train = daily_train[list(DAILY_FIELDS)].copy()
    x_future = daily_future[list(DAILY_FIELDS)].copy()
    x_train["route"] = x_train.route.astype(int).astype(str)
    x_future["route"] = x_future.route.astype(int).astype(str)
    target = (
        daily_train.boardings.to_numpy() - daily_train.profile_hour.to_numpy()
    ) / (daily_train.profile_hour.to_numpy() + 300)
    model = CatBoostRegressor(**CATBOOST_DAILY_CONFIG)
    model.fit(x_train, target)
    daily_future["daily_prediction"] = np.maximum(
        0, daily_future.profile_hour.to_numpy()
        + (daily_future.profile_hour.to_numpy() + 300) * model.predict(x_future),
    )
    hourly = future[["route", "date", "lead_day"]].merge(
        daily_future[["route", "date", "lead_day", "profile_hour", "daily_prediction"]],
        on=["route", "date", "lead_day"], how="left", sort=False, validate="many_to_one",
    )
    denominator = hourly.profile_hour.to_numpy(dtype=float)
    daily_prediction = hourly.daily_prediction.to_numpy(dtype=float)
    adjusted = np.divide(
        future_profile * daily_prediction, denominator,
        out=np.zeros_like(future_profile), where=denominator > 0,
    )
    return np.asarray(np.maximum(0, 0.75 * future_profile + 0.25 * adjusted), dtype=float)


def predict(model_name: ModelName, train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    if model_name in ("catboost_daily_bounded_50", "catboost_daily_recency_ensemble_50"):
        return hybrid_predict(train, future, model_name)
    if model_name == "calendar_robust_routeshape":
        return robust_hour_shape_predict(train, future, group="route")
    if model_name == "calendar_robust_28_routeshape":
        return robust_hour_shape_predict(train, future, "calendar_robust_28", group="route")
    if model_name == "calendar_robust_hourshape":
        return robust_hour_shape_predict(train, future)
    if model_name == "calendar_robust_28_hourshape":
        return robust_hour_shape_predict(train, future, "calendar_robust_28")
    if model_name in ("calendar_robust_short", "calendar_robust_28", "calendar_robust_blend"):
        return robust_profile_predict(train, future, model_name)
    if model_name == "profile":
        return profile_predict(future)
    if model_name == "calendar_profile":
        return calendar_profile_predict(future)
    if model_name == "calendar_weekday_blend":
        return calendar_weekday_blend_predict(future)
    if model_name in ("fourier_ridge_daily", "direct_extratrees_daily"):
        return time_series_predict(model_name, train, future)
    if model_name == "catboost_daily_relative":
        return _daily_relative_predict(train, future)
    columns = list(FEATURE_COLUMNS)
    if model_name == "catboost_weather_relative":
        columns.extend(WEATHER_COLUMNS)
    x_train = train[columns].copy()
    x_future = future[columns].copy()
    if model_name == "hgb":
        model = HistGradientBoostingRegressor(**HGB_CONFIG)
    elif model_name in (
        "catboost", "catboost_residual", "catboost_relative", "catboost_weather_relative"
    ):
        from catboost import CatBoostRegressor

        x_train["route"] = x_train.route.astype(int).astype(str)
        x_future["route"] = x_future.route.astype(int).astype(str)
        config = {
            "catboost": CATBOOST_CONFIG,
            "catboost_residual": CATBOOST_RESIDUAL_CONFIG,
            "catboost_relative": CATBOOST_RELATIVE_CONFIG,
            "catboost_weather_relative": CATBOOST_RELATIVE_CONFIG,
        }[model_name]
        model = CatBoostRegressor(**config)
    else:
        raise ValueError(f"unknown model {model_name}")
    baseline = profile_predict(future)
    if model_name == "catboost_residual":
        model.fit(x_train, train.boardings.to_numpy() - profile_predict(train))
        return np.asarray(np.maximum(0, baseline + model.predict(x_future)), dtype=float)
    if model_name in ("catboost_relative", "catboost_weather_relative"):
        training_profile = profile_predict(train)
        target = (train.boardings.to_numpy() - training_profile) / (training_profile + 30)
        model.fit(x_train, target)
        adjusted = np.maximum(0, baseline + (baseline + 30) * model.predict(x_future))
        return np.asarray(0.75 * baseline + 0.25 * adjusted, dtype=float)
    model.fit(x_train, train.boardings.to_numpy())
    return np.asarray(np.maximum(0, model.predict(x_future)), dtype=float)

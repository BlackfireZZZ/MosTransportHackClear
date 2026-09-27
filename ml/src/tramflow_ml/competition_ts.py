"""Daily time-series candidates distributed through cutoff-safe hourly profiles."""

from datetime import date, timedelta

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.ensemble import ExtraTreesRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

from tramflow_ml.competition import (
    LABEL_ROUTES,
    calendar_weekday_blend_predict,
    hour_grid,
)
from tramflow_ml.route_models.features import OFF, WORK

FOURIER_CONFIG = {"annual_harmonics": 1, "ridge_alpha": 10.0, "blend_weight": 0.25}
DIRECT_FOREST_CONFIG = {
    "input_days": 28,
    "horizon_days": 61,
    "training_stride_days": 3,
    "n_estimators": 80,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": 1,
    "blend_weight": 0.25,
}


def _daily_matrix(history: pd.DataFrame) -> tuple[np.ndarray, list[date]]:
    if list(history.columns) != ["route", "date", "hour", "boardings"]:
        raise ValueError("time-series history must be dense route/date/hour labels")
    if not history.hour.isin(range(24)).all():
        raise ValueError("time-series history contains an invalid hour")
    first, last = min(history.date), max(history.date)
    days = list(pd.date_range(first, last).date)
    expected = pd.MultiIndex.from_product([LABEL_ROUTES, days], names=["route", "date"])
    grouped = history.groupby(["route", "date"]).boardings.agg(["sum", "count"])
    if not grouped.index.equals(expected) or (grouped["count"] != 24).any():
        raise ValueError("time-series history must cover every route/day/hour")
    if history.duplicated(["route", "date", "hour"]).any():
        raise ValueError("duplicate time-series history key")
    return grouped["sum"].to_numpy(dtype=float).reshape(len(LABEL_ROUTES), len(days)), days


def _fourier_design(days: list[date]) -> np.ndarray:
    index = pd.DatetimeIndex(days)
    day_of_year = index.dayofyear.to_numpy()
    workday = np.array(
        [int(day in WORK or (day.weekday() < 5 and day not in OFF)) for day in days]
    )
    terms = [(index.dayofweek.to_numpy() == weekday).astype(float) for weekday in range(6)]
    terms.extend(
        (
            workday.astype(float),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
        )
    )
    return np.column_stack(terms)


def _direct_features(
    daily: np.ndarray, days: list[date], route_index: int, day_index: int
) -> tuple[np.ndarray, float]:
    lags = daily[route_index, day_index - 28:day_index]
    scale = max(1.0, float(lags.mean()))
    current = days[day_index] if day_index < len(days) else days[-1] + timedelta(days=1)
    day_of_year = current.timetuple().tm_yday
    features = np.concatenate(
        (
            lags / scale,
            np.eye(len(LABEL_ROUTES))[route_index],
            np.eye(7)[current.weekday()],
            [np.sin(2 * np.pi * day_of_year / 365.25),
             np.cos(2 * np.pi * day_of_year / 365.25)],
        )
    )
    return features, scale


def _daily_totals(model_name: str, daily: np.ndarray, days: list[date], origin: date) -> np.ndarray:
    horizon = 61
    future_days = [origin + timedelta(days=lead) for lead in range(horizon)]
    if model_name == "fourier_ridge_daily":
        x_train = _fourier_design(days)
        x_future = _fourier_design(future_days)
        totals = np.zeros((len(LABEL_ROUTES), horizon))
        for route_index in range(len(LABEL_ROUTES)):
            model = Ridge(alpha=FOURIER_CONFIG["ridge_alpha"])
            model.fit(x_train, daily[route_index])
            totals[route_index] = np.maximum(0, model.predict(x_future))
        return totals
    if model_name != "direct_extratrees_daily":
        raise ValueError(f"unknown time-series model {model_name}")
    if len(days) < 28 + horizon:
        raise ValueError("insufficient complete time-series windows for direct forest")
    training_x = []
    training_y = []
    for index in range(28, len(days) - horizon + 1, 3):
        for route_index in range(len(LABEL_ROUTES)):
            features, scale = _direct_features(daily, days, route_index, index)
            training_x.append(features)
            training_y.append(daily[route_index, index:index + horizon] / scale)
    if not training_x:
        raise ValueError("no complete direct-forest training windows")
    config = {key: value for key, value in DIRECT_FOREST_CONFIG.items()
              if key in ("n_estimators", "max_depth", "min_samples_leaf",
                         "max_features", "random_state", "n_jobs")}
    forest = ExtraTreesRegressor(**config)
    forest.fit(np.asarray(training_x), np.asarray(training_y))
    future = [_direct_features(daily, days, index, len(days)) for index in range(len(LABEL_ROUTES))]
    return np.maximum(
        0,
        forest.predict(np.stack([row for row, _ in future]))
        * np.asarray([scale for _, scale in future])[:, None],
    )


def time_series_predict(
    model_name: str, history: pd.DataFrame, future: pd.DataFrame
) -> np.ndarray:
    """Forecast route-day totals, then retain the baseline's within-day hour shares."""
    origin = min(future.date)
    if max(history.date) != origin - timedelta(days=1):
        raise ValueError("time-series history must end the day before the forecast origin")
    expected = hour_grid(origin, origin + timedelta(days=60))
    keys = future[["route", "date", "hour"]].reset_index(drop=True).copy()
    keys["route"] = keys.route.astype(int)
    keys["hour"] = keys.hour.astype(int)
    if not keys.equals(expected):
        raise ValueError("time-series forecast requires the ordered 61-day grid")
    daily, days = _daily_matrix(history)
    baseline = calendar_weekday_blend_predict(future).reshape(len(LABEL_ROUTES), 61, 24)
    totals = _daily_totals(model_name, daily, days, origin)
    baseline_totals = baseline.sum(axis=2)
    candidate = np.divide(
        baseline * totals[:, :, None],
        baseline_totals[:, :, None],
        out=np.zeros_like(baseline),
        where=baseline_totals[:, :, None] > 0,
    )
    weight = (
        FOURIER_CONFIG["blend_weight"]
        if model_name == "fourier_ridge_daily"
        else DIRECT_FOREST_CONFIG["blend_weight"]
    )
    return np.asarray(((1 - weight) * baseline + weight * candidate).ravel(), dtype=float)

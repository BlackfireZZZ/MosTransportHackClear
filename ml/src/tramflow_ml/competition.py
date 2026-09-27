"""Route-hour competition data and cutoff-safe direct-horizon features.

Missing organizer label keys are filled only after an independent raw reconciliation
establishes that the positive-only label export omitted true zero-count cells.
"""

import csv
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.route_models.features import OFF, WORK

LABEL_ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
SUBMISSION_START = date(2025, 11, 1)
SUBMISSION_END = date(2025, 12, 31)
FEATURE_VERSION = "route-hour-direct-v5"
GRAPH_COLUMNS = (
    "graph_available",
    "graph_direction_count",
    "graph_stop_count_mean",
    "graph_edge_count_mean",
    "graph_length_available",
    "graph_length_m_mean",
    "graph_length_m_spread",
)
WEATHER_COLUMNS = (
    "weather_temperature_2m", "weather_relative_humidity_2m",
    "weather_precipitation", "weather_rain", "weather_snowfall",
    "weather_snow_depth", "weather_weather_code", "weather_cloud_cover",
    "weather_wind_speed_10m", "weather_wind_gusts_10m",
)
FEATURE_COLUMNS = (
    "route",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "day_of_month",
    "lead_day",
    "is_weekend",
    "is_summer",
    "route_hour_28",
    "route_hour_56",
    "route_hour_all",
    "route_dow_hour_84",
    "route_dow_hour_all",
    "route_hour_last_week",
    "route_dow_hour_same_season",
    "route_hour_same_season",
    "same_season_support",
    "route_day_28",
    "route_day_56",
    "history_days",
    *GRAPH_COLUMNS,
)


def load_labels(archive: Path) -> pd.DataFrame:
    """Read supplied aggregates without reading passenger identifiers."""
    frames = []
    with zipfile.ZipFile(archive) as source:
        for name in ("labels/labels_day_train.csv", "labels/labels_day_test.csv"):
            with source.open(name) as stream:
                frame = pd.read_csv(stream, sep=";", dtype={"route": int, "hour": int})
                if list(frame.columns) != ["route", "date", "hour", "boardings"]:
                    raise ValueError(f"unexpected label columns in {name}")
                frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d").dt.date
    return result


def hour_grid(start: date, end: date) -> pd.DataFrame:
    if start > end:
        raise ValueError("start must not exceed end")
    return pd.MultiIndex.from_product(
        [LABEL_ROUTES, pd.date_range(start, end).date, range(24)],
        names=["route", "date", "hour"],
    ).to_frame(index=False)


def complete_labels(
    source: pd.DataFrame, start: date, end: date, *, missing_as_zero: bool = False
) -> pd.DataFrame:
    """Validate a period; zero-fill requires separately verified raw-file completeness."""
    expected = hour_grid(start, end)
    rows = source.copy()
    if list(rows.columns) != ["route", "date", "hour", "boardings"]:
        raise ValueError("labels must have route,date,hour,boardings columns")
    rows["date"] = pd.to_datetime(rows["date"], format="%Y-%m-%d").dt.date
    if rows.duplicated(["route", "date", "hour"]).any():
        raise ValueError("duplicate label keys")
    values = pd.to_numeric(rows["boardings"], errors="coerce")
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("boardings must be finite nonnegative integers")
    rows["boardings"] = values.astype("int64")
    joined = expected.merge(rows, on=["route", "date", "hour"], how="left", validate="one_to_one")
    if len(rows) != joined.boardings.notna().sum():
        raise ValueError("label keys outside the expected route/date/hour grid")
    missing = int(joined.boardings.isna().sum())
    if missing and not missing_as_zero:
        raise ValueError(f"{missing} label keys missing; zero policy has not been verified")
    joined["boardings"] = joined.boardings.fillna(0).astype("int64")
    return joined


def _group_mean(
    target: pd.DataFrame, history: pd.DataFrame, keys: list[str], name: str
) -> pd.DataFrame:
    grouped = history.groupby(keys, observed=True).boardings.mean().rename(name).reset_index()
    return target.merge(grouped, on=keys, how="left", validate="many_to_one")


def _direction_graph_features(records: pd.DataFrame, origin: date, end: date) -> pd.DataFrame:
    required = [
        "route",
        "direction_id",
        "available_at",
        "valid_from",
        "valid_to",
        "stop_count",
        "edge_count",
        "length_m",
        "source_version",
    ]
    if list(records.columns) != required:
        raise ValueError("direction graph columns do not match the versioned contract")
    graph = records.copy()
    if graph[["route", "direction_id", "source_version"]].isna().any().any():
        raise ValueError("direction graph identity and version must be present")
    if not graph.route.isin(LABEL_ROUTES).all():
        raise ValueError("direction graph contains a route outside the scored set")
    for column in ("available_at", "valid_from", "valid_to"):
        graph[column] = graph[column].map(lambda value: date.fromisoformat(str(value)))
    if (graph.valid_from > graph.valid_to).any():
        raise ValueError("direction graph validity interval is reversed")
    if graph.duplicated(["route", "direction_id", "available_at"]).any():
        raise ValueError("duplicate direction graph version")
    graph = graph.loc[
        (graph.available_at < origin) & (graph.valid_from <= origin) & (graph.valid_to >= end)
    ].sort_values("available_at")
    graph = graph.drop_duplicates(["route", "direction_id"], keep="last")
    graph = graph.loc[~graph.source_version.astype(str).str.startswith("osm-inactive:")]
    if graph.empty:
        return pd.DataFrame(columns=["route", *GRAPH_COLUMNS])
    for column in ("stop_count", "edge_count"):
        values = pd.to_numeric(graph[column], errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must be finite and nonnegative")
        graph[column] = values
    raw_length = graph.length_m
    absent_length = raw_length.isna() | raw_length.astype(str).str.strip().eq("")
    lengths = pd.to_numeric(raw_length, errors="coerce")
    if ((~absent_length) & ~np.isfinite(lengths)).any() or (lengths < 0).any():
        raise ValueError("length_m must be finite and nonnegative when supplied")
    graph["length_m"] = lengths
    result = graph.groupby("route", as_index=False).agg(
        graph_direction_count=("direction_id", "nunique"),
        graph_stop_count_mean=("stop_count", "mean"),
        graph_edge_count_mean=("edge_count", "mean"),
        graph_length_available=("length_m", lambda values: values.notna().mean()),
        graph_length_m_mean=("length_m", "mean"),
        graph_length_m_min=("length_m", "min"),
        graph_length_m_max=("length_m", "max"),
    )
    result["graph_available"] = 1
    result["graph_length_m_spread"] = result.graph_length_m_max - result.graph_length_m_min
    return result[["route", *GRAPH_COLUMNS]]


def feature_frame(
    history: pd.DataFrame,
    origin: date,
    horizon_days: int = 61,
    *,
    direction_graph: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Label statistics use dates before origin; supplied weather may be retrospective."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    available = history.loc[history.date < origin].copy()
    if available.empty:
        raise ValueError("no history before forecast origin")
    first = min(available.date)
    target = hour_grid(origin, origin + timedelta(days=horizon_days - 1))
    target["day_of_week"] = pd.to_datetime(target.date).dt.dayofweek.astype("int16")
    target["month"] = pd.to_datetime(target.date).dt.month.astype("int16")
    target["day_of_year"] = pd.to_datetime(target.date).dt.dayofyear.astype("int16")
    target["day_of_month"] = pd.to_datetime(target.date).dt.day.astype("int16")
    target["lead_day"] = (pd.to_datetime(target.date) - pd.Timestamp(origin)).dt.days + 1
    target["is_weekend"] = (target.day_of_week >= 5).astype("int8")
    target["is_summer"] = target.month.isin((6, 7, 8)).astype("int8")
    available["day_of_week"] = pd.to_datetime(available.date).dt.dayofweek.astype("int16")
    available["is_summer"] = pd.to_datetime(available.date).dt.month.isin((6, 7, 8)).astype("int8")
    target["calendar_day_type"] = [
        0 if day in WORK or (day.weekday() < 5 and day not in OFF) else
        1 if day.weekday() == 5 and day not in OFF else 2
        for day in target.date
    ]
    target["calendar_exception"] = target.date.isin(OFF | WORK).astype("int8")
    available["calendar_day_type"] = [
        0 if day in WORK or (day.weekday() < 5 and day not in OFF) else
        1 if day.weekday() == 5 and day not in OFF else 2
        for day in available.date
    ]
    for days in (28, 56):
        recent = available.loc[available.date >= origin - timedelta(days=days)]
        target = _group_mean(target, recent, ["route", "hour"], f"route_hour_{days}")
        target = _group_mean(target, recent, ["route"], f"route_day_{days}")
    target = _group_mean(target, available, ["route", "hour"], "route_hour_all")
    recent84 = available.loc[available.date >= origin - timedelta(days=84)]
    target = _group_mean(target, recent84, ["route", "day_of_week", "hour"], "route_dow_hour_84")
    target = _group_mean(target, available, ["route", "day_of_week", "hour"], "route_dow_hour_all")
    seasonal_keys = ["route", "is_summer", "day_of_week", "hour"]
    target = _group_mean(target, available, seasonal_keys, "route_dow_hour_same_season")
    target = _group_mean(
        target, available, ["route", "is_summer", "hour"], "route_hour_same_season"
    )
    support = (
        available.groupby(seasonal_keys, observed=True)
        .boardings.count()
        .rename("same_season_support")
        .reset_index()
    )
    target = target.merge(support, on=seasonal_keys, how="left", validate="many_to_one")
    day_type_keys = ["route", "is_summer", "calendar_day_type", "hour"]
    target = _group_mean(target, available, day_type_keys, "route_day_type_hour_same_season")
    day_type_support = (
        available.groupby(day_type_keys, observed=True)
        .boardings.count()
        .rename("day_type_support")
        .reset_index()
    )
    target = target.merge(day_type_support, on=day_type_keys, how="left", validate="many_to_one")
    last_week = available.loc[available.date >= origin - timedelta(days=7)]
    target = _group_mean(
        target, last_week, ["route", "day_of_week", "hour"], "route_hour_last_week"
    )
    target["history_days"] = (origin - first).days
    if direction_graph is not None:
        graph = _direction_graph_features(
            direction_graph, origin, origin + timedelta(days=horizon_days - 1)
        )
        target = target.merge(graph, on="route", how="left", validate="many_to_one")
    else:
        for name in GRAPH_COLUMNS:
            target[name] = 0.0
    if weather is not None:
        from tramflow_ml.weather import VARIABLES

        keys = ["route"] if "route" in weather.columns else []
        if list(weather.columns) != [*keys, "date", "hour", *VARIABLES]:
            raise ValueError("weather columns do not match the versioned contract")
        if weather.duplicated([*keys, "date", "hour"]).any():
            raise ValueError("duplicate weather hour")
        renamed = weather.rename(columns=dict(zip(VARIABLES, WEATHER_COLUMNS, strict=True)))
        target = target.merge(
            renamed, on=[*keys, "date", "hour"], how="left", validate="many_to_one"
        )
        if target[list(WEATHER_COLUMNS)].isna().any().any():
            raise ValueError("weather has missing values in forecast horizon")
    for name in FEATURE_COLUMNS:
        target[name] = target[name].fillna(0).astype("float32")
    for name in (
        "calendar_day_type",
        "calendar_exception",
        "route_day_type_hour_same_season",
        "day_type_support",
    ):
        target[name] = target[name].fillna(0).astype("float32")
    return target


def make_training_frame(
    history: pd.DataFrame,
    last_label_day: date,
    horizon_days: int = 61,
    *,
    direction_graph: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Simulate 61-day forecasts from 14-day-spaced historical origins."""
    first = min(history.date)
    earliest = first + timedelta(days=56)
    latest = last_label_day - timedelta(days=horizon_days - 1)
    if earliest > latest:
        raise ValueError("insufficient history for complete training horizons")
    frames = []
    origin = earliest
    while origin <= latest:
        rows = feature_frame(
            history, origin, horizon_days, direction_graph=direction_graph, weather=weather
        )
        truth = history.loc[
            (history.date >= origin) & (history.date < origin + timedelta(days=horizon_days)),
            ["route", "date", "hour", "boardings"],
        ]
        merged = rows.merge(truth, on=["route", "date", "hour"], validate="one_to_one")
        if len(merged) != len(rows):
            raise ValueError(f"missing training labels after origin {origin}")
        frames.append(merged)
        origin += timedelta(days=14)
    return pd.concat(frames, ignore_index=True)


def profile_predict(features: pd.DataFrame) -> np.ndarray:
    """Historical weekday-hour mean from the target's summer/non-summer regime."""
    supported = features.same_season_support.to_numpy(dtype=float) > 0
    seasonal = features.route_dow_hour_same_season.to_numpy(dtype=float)
    fallback = features.route_dow_hour_all.to_numpy(dtype=float)
    return np.asarray(np.maximum(0, np.where(supported, seasonal, fallback)), dtype=float)


def calendar_profile_predict(features: pd.DataFrame) -> np.ndarray:
    """Apply the published 2025 workday/holiday calendar to exceptional dates."""
    baseline = profile_predict(features)
    eligible = (
        (features.calendar_exception.to_numpy(dtype=float) > 0)
        & (features.day_type_support.to_numpy(dtype=float) > 0)
    )
    corrected = features.route_day_type_hour_same_season.to_numpy(dtype=float)
    return np.maximum(0, np.where(eligible, corrected, baseline))


def calendar_weekday_blend_predict(features: pd.DataFrame) -> np.ndarray:
    """Blend weekday and published day-type means when the latter has support."""
    baseline = calendar_profile_predict(features)
    day_type = features.route_day_type_hour_same_season.to_numpy(dtype=float)
    eligible = (
        (features.calendar_exception.to_numpy(dtype=float) == 0)
        & (features.day_type_support.to_numpy(dtype=float) > 0)
    )
    return np.maximum(0, np.where(eligible, 0.5 * baseline + 0.5 * day_type, baseline))


def rounded(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    return np.asarray(np.rint(np.maximum(0, values)), dtype="int64")


def score(actual: list[int] | np.ndarray, predictions: list[float] | np.ndarray) -> float | None:
    truth = np.asarray(actual, dtype=float)
    forecast = rounded(np.asarray(predictions))
    if len(truth) != len(forecast) or not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("actual and prediction arrays must align and be nonnegative")
    denominator = float(truth.sum())
    error = float(np.abs(truth - forecast).sum())
    return None if denominator == 0 else max(0.0, 1.0 - error / denominator)


def validate_submission(rows: pd.DataFrame) -> None:
    if list(rows.columns) != ["route", "date", "hour", "prediction"]:
        raise ValueError("submission columns must be route,date,hour,prediction")
    if rows.duplicated(["route", "date", "hour"]).any():
        raise ValueError("duplicate submission keys")
    predictions = pd.to_numeric(rows.prediction, errors="coerce")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("predictions must be finite and nonnegative")
    expected = hour_grid(SUBMISSION_START, SUBMISSION_END)
    actual_keys = rows[["route", "date", "hour"]].copy()
    actual_keys["date"] = pd.to_datetime(actual_keys.date, errors="coerce").dt.date
    if len(rows) != len(expected) or not actual_keys.equals(expected):
        merged = expected.merge(actual_keys, how="outer", indicator=True)
        if (merged._merge != "both").any() or len(rows) != len(expected):
            raise ValueError("submission grid differs from required 14,640 keys")


def write_submission(rows: pd.DataFrame, output: Path) -> None:
    validate_submission(rows)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(("route", "date", "hour", "prediction"))
        for row in rows.itertuples(index=False):
            writer.writerow((row.route, row.date.isoformat(), row.hour, int(round(row.prediction))))

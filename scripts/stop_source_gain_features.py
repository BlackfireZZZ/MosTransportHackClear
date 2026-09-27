"""Offline ablation features; deliberately unavailable to online serving."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.external_factors import WEATHER_FIELDS, _dated_posts
from tramflow_ml.route_models.features import DAY_TYPE, DOW, HOLIDAY, MONTH
from tramflow_ml.stop_models import FloatArray, StopData, aggregate

SOURCE_ORDER = ("calendar", "weather", "traffic", "events")
VERSION = "independent-stop-sources.v1"


def enabled_sources(mask: int) -> list[str]:
    if not isinstance(mask, int) or not 0 <= mask < 16:
        raise ValueError("source mask must be an integer 0..15")
    return [name for bit, name in enumerate(SOURCE_ORDER) if mask & (1 << bit)]


def history_profile(history: FloatArray, origin: int, days: int,
                    window: int, calendar: bool) -> FloatArray:
    """Calendar-off grouping contains ordinary weekdays only, never holiday labels."""
    if origin < 1 or origin > len(history) or days < 1 or origin + days > 365:
        raise ValueError("profile dates outside 2025 history")
    start = max(0, origin - window)
    kinds = DAY_TYPE if calendar else DOW
    output = np.zeros((days, history.shape[1], 24))
    for kind in np.unique(kinds):
        selected = history[start:origin][kinds[start:origin] == kind]
        count = np.isfinite(selected).sum(axis=0)
        mean = np.divide(np.nansum(selected, axis=0), count,
                         out=np.zeros(count.shape), where=count > 0)
        output[kinds[origin:origin + days] == kind] = mean
    return output


def feature_frame(data: StopData, origin: int, days: int, mask: int,
                  weather: Any, posts: Any, events: FloatArray) -> Any:
    """Counts/weather/traffic end before origin; events are labelled retrospective."""
    sources = enabled_sources(mask)
    if origin < 28 or origin > len(data.train) or days < 1 or origin + days > 365:
        raise ValueError("features require history and a 2025 horizon")
    count = len(data.identities)
    d = np.repeat(np.arange(origin, origin + days), count * 24)
    s = np.tile(np.repeat(np.arange(count), 24), days)
    h = np.tile(np.arange(24), days * count)
    ids = data.identities
    x = pd.DataFrame({"route": ids.route.to_numpy()[s], "stop": s,
                      "direction": ids.direction.to_numpy()[s], "hour": h,
                      "weekday": DOW[d], "lead": d-origin+1,
                      "lat": ids.lat.fillna(0).to_numpy()[s],
                      "lon": ids.lon.fillna(0).to_numpy()[s],
                      "geometry_missing": ids.lat.isna().to_numpy()[s]})
    calendar = "calendar" in sources
    if calendar:
        x["calendar_month"] = MONTH[d]
        x["calendar_day_type"] = DAY_TYPE[d]
        x["calendar_holiday"] = HOLIDAY[d]
    for window in (28, 56, 84):
        x[f"profile_{window}"] = history_profile(data.train, origin, days,
                                                  window, calendar).ravel()
    route = aggregate(history_profile(data.train, origin, days, 56, calendar), ids, origin)
    route["day"] = (pd.to_datetime(route.date)-pd.Timestamp("2025-01-01")).dt.days
    x["route_profile"] = x[["route", "hour"]].assign(day=d).merge(
        route, on=["route", "hour", "day"], how="left", validate="many_to_one"
    ).prediction.to_numpy()
    cutoff = date(2025, 1, 1) + timedelta(days=origin)
    if "weather" in sources:
        past = weather.loc[(weather.date < cutoff)
                           & (weather.date >= cutoff-timedelta(days=28))]
        stats = past.groupby(["route", "hour"])[WEATHER_FIELDS].mean().reset_index()
        joined = x[["route", "hour"]].merge(stats, how="left", validate="many_to_one")
        for column in WEATHER_FIELDS:
            x["weather_"+column] = joined[column].fillna(-999.)
        x["weather_available"] = joined.temperature_2m.notna().astype(int)
    if "traffic" in sources:
        past = _dated_posts(posts, cutoff, 56)
        for column in ("congestion_score", "mean_speed_kmh"):
            selected = past.loc[past[column].notna()]
            stats = selected.groupby(["day_of_week", "hour"])[column].agg(
                mean="mean", support="count").reset_index().rename(
                    columns={"day_of_week": "weekday"})
            joined = x[["weekday", "hour"]].merge(stats, how="left", validate="many_to_one")
            x["traffic_"+column] = joined["mean"].fillna(-1.)
            x["traffic_"+column+"_support"] = joined.support.fillna(0.)
    if "events" in sources:
        if events.shape != (365, count, 24) or not np.isfinite(events).all() or (events < 0).any():
            raise ValueError("events require complete finite 2025 tensor")
        x["events_retrospective_recent"] = history_profile(
            events, origin, days, 84, False).ravel()
    return x



def event_tensor(path: Path, identities: Any) -> tuple[Any, dict[str, Any]]:
    """Known exact hours within 1 km; no assumed attendees or unresolved times."""
    rows = pd.read_csv(path)
    required = {"date", "hour", "lat", "lon"}
    if not required <= set(rows.columns):
        raise ValueError("event CSV requires date/hour/lat/lon")
    result = np.zeros((365, len(identities), 24))
    mapped = 0
    for row in rows.itertuples(index=False):
        day = (pd.Timestamp(row.date)-pd.Timestamp("2025-01-01")).days
        if not 0 <= day < 365 or int(row.hour) != row.hour or not 0 <= int(row.hour) < 24:
            raise ValueError("event date/hour outside 2025")
        lat, lon = float(row.lat), float(row.lon)
        if not np.isfinite([lat, lon]).all() or abs(lat) > 90 or abs(lon) > 180:
            raise ValueError("invalid event coordinates")
        distance = np.sqrt(((identities.lat.to_numpy()-lat)*111.2)**2 +
                           ((identities.lon.to_numpy()-lon)*111.2*np.cos(np.deg2rad(lat)))**2)
        nearby = np.isfinite(distance) & (distance <= 1.)
        result[day, nearby, int(row.hour)] += 1
        mapped += int(nearby.any())
    return result, {"input_hour_rows": len(rows), "rows_near_stop": mapped,
                    "radius_km": 1, "availability": "retrospective_2026_snapshot",
                    "policy": "prior84day_weekday_hour_occurrence_means",
                    "date_min": str(rows.date.min()), "date_max": str(rows.date.max()),
                    "nonzero_tensor_cells": int(np.count_nonzero(result))}


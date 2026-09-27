"""Features use only history before origin; the 2025 calendar is future-known."""

from datetime import date

import numpy as np

from tramflow_ml.route_models.data import FloatArray, History, civil_date

CALENDAR_SOURCE = "https://government.ru/docs/52895/"
CALENDAR_ISSUED = date(2024, 10, 4)
OFF = frozenset(
    [date(2025, 1, day) for day in range(1, 9)]
    + [
        date(2025, month, day)
        for month, day in (
            (2, 23),
            (3, 8),
            (5, 1),
            (5, 2),
            (5, 8),
            (5, 9),
            (6, 12),
            (6, 13),
            (11, 3),
            (11, 4),
            (12, 31),
        )
    ]
)
WORK = frozenset({date(2025, 11, 1)})
DATES = tuple(civil_date(i) for i in range(365))
DOW = np.array([day.weekday() for day in DATES])
MONTH = np.array([day.month for day in DATES])
DAY_TYPE = np.array(
    [
        0
        if day in WORK or (day.weekday() < 5 and day not in OFF)
        else (1 if day.weekday() == 5 and day not in OFF else 2)
        for day in DATES
    ]
)
HOLIDAY = np.array([day in OFF for day in DATES])
CALENDAR_NAMES = ("route", "hour", "weekday", "day_type", "july_august", "holiday")
PROFILE_NAMES = (
    "route",
    "hour",
    "weekday",
    "day_type",
    "month",
    "lead_days",
    "july_august",
    "holiday",
    "week1",
    "week2",
    "week4",
    "week8",
    "type4",
    "median4",
    "ratio1",
    "ratio2",
    "ratio8",
    "routelevel7",
    "routelevel28",
)


def calendar_features(start: int, days: int) -> FloatArray:
    if start < 0 or days < 1 or start + days > 365:
        raise ValueError("calendar supports only complete days in 2025")
    t = np.repeat(np.arange(start, start + days), 240)
    return np.column_stack(
        [
            np.tile(np.repeat(np.arange(10), 24), days),
            np.tile(np.arange(24), days * 10),
            DOW[t],
            DAY_TYPE[t],
            (MONTH[t] >= 7) & (MONTH[t] <= 8),
            HOLIDAY[t],
        ]
    ).astype(np.float64)


def profiles(history: History, origin: int, days: int) -> tuple[FloatArray, FloatArray]:
    if not 28 <= origin <= len(history.values) or not 1 <= days <= 61:
        raise ValueError("profiles need 28 past days and a horizon of 1..61 days")
    cal = calendar_features(origin, days)
    targets = np.arange(origin, origin + days)
    cols = [
        cal[:, 0],
        cal[:, 1],
        cal[:, 2],
        cal[:, 3],
        np.repeat(MONTH[targets], 240),
        np.repeat(np.arange(1, days + 1), 240),
        cal[:, 4],
        cal[:, 5],
    ]
    past = history.values[:origin]
    weeks: dict[int, FloatArray] = {}
    for window in (1, 2, 4, 8):
        begin = max(0, origin - 7 * window)
        hist = past[begin:]
        profile = np.stack([hist[DOW[begin:origin] == k].mean(axis=0) for k in range(7)])
        weeks[window] = profile[DOW[targets]].reshape(-1)
        cols.append(weeks[window])
    hist = past[-28:]
    types = np.stack([hist[DAY_TYPE[origin - 28 : origin] == k].mean(axis=0) for k in range(3)])
    cols.append(types[DAY_TYPE[targets]].reshape(-1))
    med = np.stack([np.median(hist[DOW[origin - 28 : origin] == k], axis=0) for k in range(7)])
    cols.append(med[DOW[targets]].reshape(-1))
    scale = np.maximum(weeks[4], 20.0)
    for window in (1, 2, 8):
        cols.append(weeks[window] / scale)
    for window in (7, 28):
        level = past[-window:].mean(axis=0).sum(axis=1)
        cols.append(np.broadcast_to(level[None, :, None], (days, 10, 24)).reshape(-1))
    features = np.column_stack(cols)
    if not np.isfinite(features).all():
        raise ValueError("unsupported history: a required calendar profile is absent")
    return features, scale


def training_snapshots(history: History, cutoff: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not 35 <= cutoff <= len(history.values):
        raise ValueError("training needs at least 35 historical days")
    xs, ys, scales = [], [], []
    for origin in range(28, cutoff, 7):
        days = min(61, cutoff - origin)
        features, scale = profiles(history, origin, days)
        xs.append(features)
        ys.append(history.values[origin : origin + days].reshape(-1))
        scales.append(scale)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(scales)

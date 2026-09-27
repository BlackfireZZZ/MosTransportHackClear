"""Route-level demand change observed before a forecast origin."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.decision_guard import ShiftSignal
from tramflow_ml.route_models.features import OFF, WORK


@dataclass(frozen=True)
class LevelShiftPolicy:
    recent_days: int = 14
    reference_days: int = 28
    ratio_threshold: float = 1.3
    min_reference_daily: float = 20.0
    min_reference_days: int = 20
    min_recent_week_days: int = 5


@dataclass(frozen=True)
class ShapeShiftPolicy:
    recent_days: int = 28
    reference_days: int = 56
    min_reference_days: int = 40
    min_recent_half_days: int = 10
    min_reference_daily: float = 20.0
    total_variation_threshold: float = 0.05
    min_half_alignment: float = 0.5


def detect_level_shifts(
    labels: pd.DataFrame, origin: date, policy: LevelShiftPolicy | None = None
) -> tuple[ShiftSignal, ...]:
    """Require two same-direction recent weeks and complete hourly source days."""
    policy = policy or LevelShiftPolicy()
    if policy.recent_days != 14 or policy.reference_days < 14:
        raise ValueError("detector requires two recent weeks and at least two reference weeks")
    if (policy.reference_days % 7 or not np.isfinite(policy.ratio_threshold)
            or policy.ratio_threshold <= 1 or policy.min_reference_daily < 0
            or policy.min_reference_days < 1 or policy.min_recent_week_days < 1):
        raise ValueError("invalid level-shift policy")
    required = {"route", "date", "hour", "boardings"}
    if not required.issubset(labels.columns):
        raise ValueError("route,date,hour,boardings are required")
    start = origin - timedelta(days=policy.recent_days + policy.reference_days)
    past = labels.loc[(labels.date >= start) & (labels.date < origin),
                      ["route", "date", "hour", "boardings"]].copy()
    if past.empty:
        return ()
    if past.duplicated(["route", "date", "hour"]).any():
        raise ValueError("duplicate history keys")
    if not past.hour.isin(range(24)).all():
        raise ValueError("history hour must be in 0..23")
    values = pd.to_numeric(past.boardings, errors="coerce")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("history counts must be finite and nonnegative")
    past["boardings"] = values
    daily = past.groupby(["route", "date"], observed=True).agg(
        total=("boardings", "sum"), hours=("hour", "nunique")
    ).reset_index()
    signals: list[ShiftSignal] = []
    for route, rows in daily.groupby("route", observed=True):
        by_day = rows.set_index("date")
        expected = [start + timedelta(days=offset) for offset in range(
            policy.recent_days + policy.reference_days
        )]
        if len(by_day) != len(expected) or not by_day.index.isin(expected).all():
            continue
        if (by_day.hours != 24).any():
            continue
        complete = by_day.reindex(expected)
        normal = complete.loc[[day not in OFF | WORK for day in expected]].copy()
        normal["weekday"] = [day.weekday() for day in normal.index]
        cutoff = origin - timedelta(days=policy.recent_days)
        reference_rows = normal.loc[normal.index < cutoff]
        if len(reference_rows) < policy.min_reference_days:
            continue
        by_weekday = reference_rows.groupby("weekday").total.mean()
        recent_rows = normal.loc[normal.index >= cutoff].copy()
        recent_rows["expected"] = recent_rows.weekday.map(by_weekday)
        if recent_rows.expected.isna().any():
            continue
        first_rows = recent_rows.loc[recent_rows.index < origin - timedelta(days=7)]
        second_rows = recent_rows.loc[recent_rows.index >= origin - timedelta(days=7)]
        if (len(first_rows) < policy.min_recent_week_days
                or len(second_rows) < policy.min_recent_week_days):
            continue
        reference = float(recent_rows.expected.mean())
        if reference < policy.min_reference_daily:
            continue
        recent = float(recent_rows.total.mean())
        ratios = (
            float(first_rows.total.sum() / first_rows.expected.sum()),
            float(second_rows.total.sum() / second_rows.expected.sum()),
            float(recent_rows.total.sum() / recent_rows.expected.sum()),
        )
        if all(ratio >= policy.ratio_threshold for ratio in ratios):
            direction: Literal["up", "down"] = "up"
        elif all(ratio <= 1 / policy.ratio_threshold for ratio in ratios):
            direction = "down"
        else:
            continue
        signals.append(ShiftSignal(str(route), direction, ratios[-1], recent, reference))
    return tuple(signals)


def detect_hour_shape_shifts(
    labels: pd.DataFrame, origin: date, policy: ShapeShiftPolicy | None = None
) -> tuple[ShiftSignal, ...]:
    """Compare normal-day hour shares across complete, non-overlapping weeks."""
    policy = policy or ShapeShiftPolicy()
    if (policy.recent_days < 2 or policy.recent_days % 2
            or policy.reference_days < 7 or policy.min_reference_days < 1
            or policy.min_recent_half_days < 1 or policy.min_reference_daily < 0
            or not np.isfinite(policy.total_variation_threshold)
            or not 0 < policy.total_variation_threshold <= 1
            or not np.isfinite(policy.min_half_alignment)
            or not -1 <= policy.min_half_alignment <= 1):
        raise ValueError("invalid hour-shape policy")
    required = {"route", "date", "hour", "boardings"}
    if not required.issubset(labels.columns):
        raise ValueError("route,date,hour,boardings are required")
    start = origin - timedelta(days=policy.recent_days + policy.reference_days)
    rows = labels.loc[(labels.date >= start) & (labels.date < origin),
                      ["route", "date", "hour", "boardings"]].copy()
    if rows.empty:
        return ()
    if rows.duplicated(["route", "date", "hour"]).any():
        raise ValueError("duplicate history keys")
    if not rows.hour.isin(range(24)).all():
        raise ValueError("history hour must be in 0..23")
    values = pd.to_numeric(rows.boardings, errors="coerce")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("history counts must be finite and nonnegative")
    rows["boardings"] = values
    expected = [start + timedelta(days=offset) for offset in range(
        policy.recent_days + policy.reference_days
    )]
    normal_days = [day for day in expected if day not in OFF | WORK]
    cutoff = origin - timedelta(days=policy.recent_days)
    midpoint = origin - timedelta(days=policy.recent_days // 2)
    signals: list[ShiftSignal] = []
    for route, route_rows in rows.groupby("route", observed=True):
        matrix = route_rows.pivot(index="date", columns="hour", values="boardings")
        if (len(matrix) != len(expected) or not matrix.index.isin(expected).all()
                or set(matrix.columns) != set(range(24))):
            continue
        matrix = matrix.reindex(index=normal_days, columns=range(24))
        if matrix.isna().any().any():
            continue
        reference = matrix.loc[matrix.index < cutoff]
        first = matrix.loc[(matrix.index >= cutoff) & (matrix.index < midpoint)]
        second = matrix.loc[matrix.index >= midpoint]
        if (len(reference) < policy.min_reference_days
                or len(first) < policy.min_recent_half_days
                or len(second) < policy.min_recent_half_days):
            continue
        reference_total = float(reference.to_numpy().sum())
        first_total = float(first.to_numpy().sum())
        second_total = float(second.to_numpy().sum())
        if (reference_total / len(reference) < policy.min_reference_daily
                or first_total <= 0 or second_total <= 0):
            continue
        reference_share = reference.sum().to_numpy(dtype=float) / reference_total
        first_share = first.sum().to_numpy(dtype=float) / first_total
        second_share = second.sum().to_numpy(dtype=float) / second_total
        recent_share = (first.sum() + second.sum()).to_numpy(dtype=float)
        recent_share /= first_total + second_total
        first_delta = first_share - reference_share
        second_delta = second_share - reference_share
        first_tv = float(np.abs(first_delta).sum() / 2)
        second_tv = float(np.abs(second_delta).sum() / 2)
        distance = float(np.abs(recent_share - reference_share).sum() / 2)
        denominator = float(np.linalg.norm(first_delta) * np.linalg.norm(second_delta))
        alignment = float(np.dot(first_delta, second_delta) / denominator) if denominator else 0
        if (min(first_tv, second_tv, distance) < policy.total_variation_threshold
                or alignment < policy.min_half_alignment):
            continue
        signals.append(ShiftSignal(
            str(route), "changed", distance,
            (first_total + second_total) / (len(first) + len(second)),
            reference_total / len(reference), "hour_shape", "total_variation",
        ))
    return tuple(signals)

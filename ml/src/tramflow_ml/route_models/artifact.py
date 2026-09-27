"""Experimental route-hour output; intentionally distinct from stop-based forecast.v1."""

import calendar
import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tramflow_ml.route_models.data import ROUTES

MOSCOW = ZoneInfo("Europe/Moscow")


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps require an explicit offset")
    return value.astimezone(UTC).astimezone(MOSCOW)


class RoutePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    route: int = Field(strict=True)
    bucket_start: datetime
    predicted: float = Field(ge=0)


class ProjectedPoint(BaseModel):
    timestamp: datetime
    predicted: float


class RouteForecast(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    schema_version: Literal["route-forecast.experimental.v1"] = "route-forecast.experimental.v1"
    scope: Literal["route_all_directions"] = "route_all_directions"
    target: Literal["validation_count"] = "validation_count"
    unit: Literal["event_count"] = "event_count"
    synthetic: Literal[False] = False
    status: Literal["experimental"] = "experimental"
    timezone: Literal["Europe/Moscow"] = "Europe/Moscow"
    run_id: str = Field(min_length=1, max_length=128)
    dataset_id: str = Field(min_length=1, max_length=128)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_version: str = Field(min_length=1, max_length=128)
    feature_version: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    forecast_origin: datetime
    data_cutoff: datetime
    generated_at: datetime
    points: tuple[RoutePoint, ...]

    @model_validator(mode="after")
    def validate_grid(self) -> "RouteForecast":
        origin = aware(self.forecast_origin)
        cutoff, generated = aware(self.data_cutoff), aware(self.generated_at)
        if not cutoff <= origin <= generated:
            raise ValueError("require cutoff <= origin <= generation")
        if origin.time() != datetime.min.time():
            raise ValueError("origin must be Moscow midnight")
        if len(self.points) != len(ROUTES) * 61 * 24:
            raise ValueError("require a complete ten-route 61-day hourly grid")
        keys: set[tuple[int, datetime]] = set()
        for point in self.points:
            instant = aware(point.bucket_start)
            delta = (instant - origin).total_seconds()
            key = (point.route, instant)
            if point.route not in ROUTES or not 0 <= delta < 61 * 86400 or delta % 3600:
                raise ValueError("point outside the required route/hour grid")
            if key in keys:
                raise ValueError("duplicate route/hour point")
            keys.add(key)
        return self

    def project(self, route: int, start: datetime, horizon: str) -> list[ProjectedPoint]:
        start = aware(start)
        if route not in ROUTES or start.time() != datetime.min.time():
            raise ValueError("projection needs a known route and Moscow midnight")
        if horizon not in {"day", "month"}:
            raise ValueError("only day and month are supported")
        if horizon == "month" and start.day != 1:
            raise ValueError("month starts on its first civil day")
        days = 1 if horizon == "day" else calendar.monthrange(start.year, start.month)[1]
        end = start + timedelta(days=days)
        points = sorted(
            (p for p in self.points if p.route == route and start <= aware(p.bucket_start) < end),
            key=lambda p: aware(p.bucket_start),
        )
        if len(points) != days * 24:
            raise ValueError("requested projection is not fully covered")
        if horizon == "day":
            return [ProjectedPoint(timestamp=p.bucket_start, predicted=p.predicted) for p in points]
        return [
            ProjectedPoint(
                timestamp=start + timedelta(days=d),
                predicted=math.fsum(p.predicted for p in points[d * 24 : (d + 1) * 24]),
            )
            for d in range(days)
        ]

    def write_submission(self, path: Path) -> None:
        if aware(self.forecast_origin) != datetime(2025, 11, 1, tzinfo=MOSCOW):
            raise ValueError("competition submission requires November 1 origin")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["route", "date", "hour", "prediction"])
            for p in sorted(self.points, key=lambda p: (p.route, aware(p.bucket_start))):
                stamp = aware(p.bucket_start)
                writer.writerow([p.route, stamp.date().isoformat(), stamp.hour, round(p.predicted)])

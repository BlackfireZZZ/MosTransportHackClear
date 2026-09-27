"""Publish a checksum-bound organizer route-hour forecast into the serving schema."""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import (
    ForecastActiveRunModel,
    ForecastPointModel,
    ForecastRunModel,
    RouteModel,
)
from app.infrastructure.db.session import session_factory
from app.infrastructure.scored_bundle import (
    ScoredBundle,
    load_approved_scored_bundle,
    load_scored_bundle,
)

ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
START = date(2025, 11, 1)
END = date(2025, 12, 31)
MOSCOW = ZoneInfo("Europe/Moscow")
MAX_PREDICTION = 1_000_000_000


def load_scored_csv(path: Path, expected_sha256: str) -> dict[tuple[int, date, int], int]:
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("expected SHA-256 must be 64 lowercase hex characters")
    with path.open("rb") as stream:
        payload = stream.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("forecast CSV exceeds size limit")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("forecast CSV checksum mismatch")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""), delimiter=";")
    if reader.fieldnames != ["route", "date", "hour", "prediction"]:
        raise ValueError("forecast CSV header differs from organizer contract")
    result: dict[tuple[int, date, int], int] = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("forecast CSV row has missing or extra fields")
        route_raw, day_raw, hour_raw, value_raw = (
            row["route"], row["date"], row["hour"], row["prediction"]
        )
        if not (
            re.fullmatch(r"[0-9]+", route_raw)
            and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day_raw)
            and re.fullmatch(r"[0-9]+", hour_raw)
            and re.fullmatch(r"[0-9]+", value_raw)
        ):
            raise ValueError("forecast key or prediction is not a nonnegative integer")
        route = int(route_raw)
        day = date.fromisoformat(day_raw)
        hour = int(hour_raw)
        value = int(value_raw)
        if route not in ROUTES or not START <= day <= END or not 0 <= hour < 24:
            raise ValueError("forecast key is outside the scored grid")
        if value >= MAX_PREDICTION:
            raise ValueError("forecast prediction exceeds the serving limit")
        key = (route, day, hour)
        if key in result:
            raise ValueError("duplicate forecast key")
        result[key] = value
    if len(result) != len(ROUTES) * 61 * 24:
        raise ValueError("forecast CSV must contain all 14,640 scored keys")
    return result


def _moscow_midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=MOSCOW).astimezone(UTC)


async def publish_scored_bundle(session: AsyncSession, bundle: ScoredBundle) -> list[str]:
    points = bundle.points
    if len(points) != 14640:
        raise ValueError("publication requires the complete scored grid")
    if not re.fullmatch(r"[a-f0-9]{64}", bundle.archive_sha256):
        raise ValueError("organizer archive SHA-256 is required")
    run_ids = [
        f"scored-2025-{horizon}-{bundle.manifest_sha256}" for horizon in ("day", "month")
    ]
    async with session.begin():
        await session.execute(text("SELECT pg_advisory_xact_lock(2025, 14640)"))
        existing = [await session.get(ForecastRunModel, run_id) for run_id in run_ids]
        if all(run is not None and run.state == "published" for run in existing):
            if any(
                run is None
                or run.dataset_id != bundle.archive_sha256
                or run.model_version != bundle.model_version
                or run.feature_version != bundle.feature_version
                or run.calendar_version != bundle.calendar_version
                or run.source_version != bundle.source_version
                for run in existing
            ):
                raise ValueError("existing forecast publication has different provenance")
            await _activate(session, run_ids, ROUTES)
            return run_ids
        if any(run is not None for run in existing):
            raise ValueError("partial or unpublished forecast publication exists")
        route_ids: dict[int, int] = {}
        for route in ROUTES:
            number = str(route)
            existing_route = await session.scalar(
                select(RouteModel).where(RouteModel.number == number)
            )
            if existing_route is None:
                route_id = 1000 + route
                if await session.get(RouteModel, route_id) is not None:
                    raise ValueError(f"serving route ID collision for {number}")
                session.add(RouteModel(
                    id=route_id, number=number, name=f"Трамвай №{number}", color="#d9342b"
                ))
            else:
                route_id = existing_route.id
            route_ids[route] = route_id
        await session.flush()
        generated_at = bundle.generated_at.astimezone(UTC)
        common = dict(
            state="draft",
            model_version=bundle.model_version,
            generated_at=generated_at,
            forecast_origin=bundle.forecast_origin.astimezone(UTC),
            data_cutoff=bundle.data_cutoff.astimezone(UTC),
            dataset_id=bundle.archive_sha256,
            source_version=bundle.source_version,
            feature_version=bundle.feature_version,
            entity_version="organizer-route-number-v1",
            calendar_version=bundle.calendar_version,
            graph_version=None,
            target="validation_count",
            unit="event_count",
            synthetic=False,
            interval_level=None,
            interval_method=None,
        )
        runs = [
            ForecastRunModel(id=run_id, horizon=horizon, **common)
            for run_id, horizon in zip(run_ids, ("day", "month"), strict=True)
        ]
        session.add_all(runs)
        await session.flush()
        day_rows = []
        month_rows = []
        current = START
        while current <= END:
            stamp = _moscow_midnight(current)
            for route in ROUTES:
                day_total = 0
                for hour in range(24):
                    value = points[(route, current, hour)]
                    day_total += value
                    day_rows.append(dict(
                        run_id=run_ids[0], route_id=route_ids[route], stop_id=None,
                        direction_id="legacy-unspecified", horizon="day",
                        bucket_start=stamp + timedelta(hours=hour),
                        predicted_passengers=float(value), lower_bound=None,
                        upper_bound=None, capacity=None, model_version=bundle.model_version,
                        generated_at=generated_at,
                    ))
                month_rows.append(dict(
                    run_id=run_ids[1], route_id=route_ids[route], stop_id=None,
                    direction_id="legacy-unspecified", horizon="month",
                    bucket_start=stamp, predicted_passengers=float(day_total),
                    lower_bound=None, upper_bound=None, capacity=None,
                    model_version=bundle.model_version, generated_at=generated_at,
                ))
            current += timedelta(days=1)
        await session.execute(insert(ForecastPointModel), day_rows)
        await session.execute(insert(ForecastPointModel), month_rows)
        for run in runs:
            run.state = "published"
        await session.flush()
        await _activate(session, run_ids, ROUTES)
    return run_ids


async def _activate(session: AsyncSession, run_ids: list[str], routes: tuple[int, ...]) -> None:
    for route in routes:
        route_id = await session.scalar(
            select(RouteModel.id).where(RouteModel.number == str(route))
        )
        if route_id is None:
            raise ValueError(f"serving route {route} is absent")
        for horizon, run_id in zip(("day", "month"), run_ids, strict=True):
            statement = pg_insert(ForecastActiveRunModel).values(
                route_id=route_id, horizon=horizon, run_id=run_id
            )
            await session.execute(statement.on_conflict_do_update(
                index_elements=["route_id", "horizon"], set_={"run_id": run_id}
            ))


async def _publish(path: Path, approvals: Path) -> list[str]:
    bundle = load_approved_scored_bundle(path, approvals)
    async with session_factory() as session:
        return await publish_scored_bundle(session, bundle)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tramflow-publish-scored-forecast")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approvals", type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        bundle = load_scored_bundle(args.manifest)
        print(json.dumps({
            "manifest_sha256": bundle.manifest_sha256,
            "csv_sha256": bundle.csv_sha256,
            "evaluation_sha256": bundle.evaluation_sha256,
            "producer_commit": bundle.producer_commit,
            "model_version": bundle.model_version,
        }))
        return
    if args.approvals is None:
        parser.error("--approvals is required for publication")
    run_ids = asyncio.run(_publish(args.manifest, args.approvals))
    print(json.dumps({"published_run_ids": run_ids, "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()

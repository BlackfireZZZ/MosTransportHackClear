import csv
import hashlib
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.forecast import ForecastHorizon, ForecastSelection
from app.infrastructure import forecast_publication
from app.infrastructure.db.models import (
    ForecastActiveRunModel,
    ForecastPointModel,
    ForecastRunModel,
    RouteModel,
)
from app.infrastructure.forecast_publication import (
    load_scored_csv,
    publish_scored_bundle,
)
from app.infrastructure.repositories.forecast import SqlAlchemyForecastRepository
from app.infrastructure.scored_bundle import ScoredBundle, load_scored_bundle


def scored_file(path: Path) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(("route", "date", "hour", "prediction"))
        for route in (1, 5, 7, 11, 12, 17, 25, 26, 28, 50):
            for offset in range(61):
                day = date(2025, 11, 1) + timedelta(days=offset)
                for hour in range(24):
                    writer.writerow((route, day.isoformat(), hour, 0 if route == 5 else hour + 1))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_bundle(path: Path, digest: str) -> ScoredBundle:
    return ScoredBundle(
        manifest_sha256="a" * 64,
        csv_sha256=digest,
        evaluation_sha256="b" * 64,
        archive_sha256="c" * 64,
        model_version="fixture-route-hour-v1",
        feature_version="fixture-features-v1",
        calendar_version="moscow-calendar-v1",
        source_version="organizer-archive-" + "c" * 16,
        config_sha256="d" * 64,
        producer_commit="e" * 40,
        forecast_origin=datetime(2025, 11, 1, tzinfo=UTC) - timedelta(hours=3),
        data_cutoff=datetime(2025, 10, 31, tzinfo=UTC) - timedelta(hours=3),
        generated_at=datetime(2026, 9, 27, tzinfo=UTC),
        points=load_scored_csv(path, digest),
    )


async def test_atomic_publication_and_serving(
    sql_session: AsyncSession, tmp_path: Path
) -> None:
    path = tmp_path / "submission.csv"
    digest = scored_file(path)
    bundle = fixture_bundle(path, digest)
    run_ids = await publish_scored_bundle(sql_session, bundle)
    assert len(run_ids) == 2
    assert await publish_scored_bundle(sql_session, bundle) == run_ids
    assert await sql_session.scalar(
        select(func.count()).select_from(ForecastPointModel).where(
            ForecastPointModel.run_id == run_ids[0]
        )
    ) == 14640
    assert await sql_session.scalar(
        select(func.count()).select_from(ForecastPointModel).where(
            ForecastPointModel.run_id == run_ids[1]
        )
    ) == 610
    for run_id in run_ids:
        run = await sql_session.get(ForecastRunModel, run_id)
        assert run is not None and run.state == "published"
        assert run.target == "validation_count" and run.unit == "event_count"
        assert not run.synthetic and run.interval_level is None
    route_5 = await sql_session.scalar(select(RouteModel).where(RouteModel.number == "5"))
    assert route_5 is not None and route_5.id != 5
    route_1 = await sql_session.scalar(select(RouteModel).where(RouteModel.number == "1"))
    assert route_1 is not None and route_1.id != 1
    repo = SqlAlchemyForecastRepository(sql_session)
    catalog = await repo.list_routes()
    assert [route.number for route in catalog[:10]] == [
        "1", "5", "7", "11", "12", "17", "25", "26", "28", "50"
    ]
    assert len(catalog) == 10
    assert await repo.get_snapshot(route_1.id, ForecastHorizon.YEAR) is None
    assert await repo.list_stops(route_1.id) == []
    start = datetime(2025, 11, 1, tzinfo=UTC) - timedelta(hours=3)
    hourly = await repo.get_snapshot(
        route_5.id, ForecastHorizon.DAY,
        ForecastSelection(start=start, end=start + timedelta(days=1)),
    )
    assert hourly is not None and len(hourly.points) == 24
    assert all(point.predicted_passengers == 0 for point in hourly.points)
    assert hourly.run is not None and hourly.run.run_id == run_ids[0]
    assert hourly.stops == [] and hourly.stop_points == []
    monthly = await repo.get_snapshot(
        route_5.id, ForecastHorizon.MONTH,
        ForecastSelection(start=start, end=start + timedelta(days=30)),
    )
    assert monthly is not None and len(monthly.points) == 30
    assert all(point.predicted_passengers == 0 for point in monthly.points)
    default_month = await repo.get_snapshot(route_5.id, ForecastHorizon.MONTH)
    assert default_month is not None and len(default_month.points) == 30
    route_1_day = await repo.get_snapshot(route_1.id, ForecastHorizon.DAY)
    route_1_month = await repo.get_snapshot(route_1.id, ForecastHorizon.MONTH)
    assert route_1_day is not None and [
        point.predicted_passengers for point in route_1_day.points
    ] == list(range(1, 25))
    assert route_1_month is not None and route_1_month.points[0].predicted_passengers == 300
    for operation in (
        update(ForecastPointModel)
        .where(ForecastPointModel.run_id == run_ids[0])
        .values(predicted_passengers=999),
        delete(ForecastPointModel).where(ForecastPointModel.run_id == run_ids[0]),
    ):
        with pytest.raises(DBAPIError, match="immutable"):
            async with sql_session.begin_nested():
                await sql_session.execute(operation)
    with pytest.raises(DBAPIError, match="cannot be demoted"):
        async with sql_session.begin_nested():
            await sql_session.execute(
                update(ForecastRunModel)
                .where(ForecastRunModel.id == run_ids[0])
                .values(state="failed")
            )
    source_run = await sql_session.get(ForecastRunModel, run_ids[0])
    assert source_run is not None
    values = {
        column.name: getattr(source_run, column.name)
        for column in source_run.__table__.columns
    }
    values.update(id="scored-2025-day-" + "b" * 64, state="draft")
    sql_session.add(ForecastRunModel(**values))
    await sql_session.flush()
    with pytest.raises(DBAPIError, match="incomplete"):
        async with sql_session.begin_nested():
            await sql_session.execute(
                update(ForecastRunModel)
                .where(ForecastRunModel.id == values["id"])
                .values(state="published")
            )


async def test_manifest_promotion_ignores_partial_newer_run(
    sql_session: AsyncSession, tmp_path: Path
) -> None:
    path = tmp_path / "baseline.csv"
    digest = scored_file(path)
    initial = await publish_scored_bundle(sql_session, fixture_bundle(path, digest))
    root = Path(__file__).resolve().parents[3]
    bundle = load_scored_bundle(root / "ml/competition_submissions/current.json")
    promoted = await publish_scored_bundle(sql_session, bundle)
    assert promoted != initial
    assert await publish_scored_bundle(sql_session, bundle) == promoted
    route = await sql_session.scalar(select(RouteModel).where(RouteModel.number == "1"))
    assert route is not None
    pointer = await sql_session.get(ForecastActiveRunModel, (route.id, "day"))
    assert pointer is not None and pointer.run_id == promoted[0]
    repo = SqlAlchemyForecastRepository(sql_session)
    result = await repo.get_snapshot(route.id, ForecastHorizon.DAY)
    assert result is not None and result.run is not None
    assert result.run.run_id == promoted[0]
    assert result.model_version == bundle.model_version
    assert result.generated_at == bundle.generated_at.astimezone(UTC)
    assert result.points[0].predicted_passengers == bundle.points[(1, date(2025, 11, 1), 0)]


async def test_failed_switch_rolls_back_new_run_and_keeps_active(
    sql_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[3]
    bundle = load_scored_bundle(root / "ml/competition_submissions/current.json")
    active = await publish_scored_bundle(sql_session, bundle)
    candidate = replace(bundle, manifest_sha256="f" * 64, model_version="fault-v1")

    async def interrupted(*_args: object) -> None:
        raise RuntimeError("injected interruption before active switch")

    with monkeypatch.context() as patcher:
        patcher.setattr(forecast_publication, "_activate", interrupted)
        with pytest.raises(RuntimeError, match="injected interruption"):
            await publish_scored_bundle(sql_session, candidate)

    route = await sql_session.scalar(select(RouteModel).where(RouteModel.number == "1"))
    assert route is not None
    pointer = await sql_session.get(ForecastActiveRunModel, (route.id, "day"))
    assert pointer is not None and pointer.run_id == active[0]
    assert await sql_session.get(ForecastRunModel, f"scored-2025-day-{'f' * 64}") is None


async def test_partial_published_run_cannot_take_active_pointer(sql_session: AsyncSession) -> None:
    root = Path(__file__).resolve().parents[3]
    bundle = load_scored_bundle(root / "ml/competition_submissions/current.json")
    promoted = await publish_scored_bundle(sql_session, bundle)
    route = await sql_session.scalar(select(RouteModel).where(RouteModel.number == "1"))
    assert route is not None
    repo = SqlAlchemyForecastRepository(sql_session)
    source = await sql_session.get(ForecastRunModel, promoted[0])
    assert source is not None
    values = {column.name: getattr(source, column.name) for column in source.__table__.columns}
    values.update(
        id="partial-later-day", state="draft",
        generated_at=source.generated_at + timedelta(days=1),
    )
    sql_session.add(ForecastRunModel(**values))
    await sql_session.flush()
    sql_session.add(ForecastPointModel(
        run_id="partial-later-day", route_id=route.id, stop_id=None,
        direction_id="legacy-unspecified", horizon="day",
        bucket_start=datetime(2025, 10, 31, 21, tzinfo=UTC),
        predicted_passengers=999.0, lower_bound=None, upper_bound=None,
        capacity=None, model_version=source.model_version,
        generated_at=values["generated_at"],
    ))
    await sql_session.flush()
    partial = await sql_session.get(ForecastRunModel, "partial-later-day")
    assert partial is not None
    partial.state = "published"
    await sql_session.flush()
    with pytest.raises(DBAPIError, match="complete published route run"):
        async with sql_session.begin_nested():
            await sql_session.execute(
                update(ForecastActiveRunModel)
                .where(ForecastActiveRunModel.route_id == route.id,
                       ForecastActiveRunModel.horizon == "day")
                .values(run_id="partial-later-day")
            )
    result = await repo.get_snapshot(route.id, ForecastHorizon.DAY)
    assert result is not None and result.run is not None
    assert result.run.run_id == promoted[0]
    await sql_session.execute(
        delete(ForecastActiveRunModel).where(
            ForecastActiveRunModel.route_id == route.id,
            ForecastActiveRunModel.horizon == "day",
        )
    )
    result_without_pointer = await repo.get_snapshot(route.id, ForecastHorizon.DAY)
    assert result_without_pointer is not None and result_without_pointer.run is not None
    assert result_without_pointer.run.run_id == promoted[0]
    legacy_values = {
        column.name: getattr(source, column.name) for column in source.__table__.columns
    }
    legacy_values.update(
        id="independent-boarding-day", state="draft", target="boarding_count",
        unit="passengers", generated_at=source.generated_at + timedelta(days=2),
    )
    sql_session.add(ForecastRunModel(**legacy_values))
    await sql_session.flush()
    sql_session.add(ForecastPointModel(
        run_id="independent-boarding-day", route_id=route.id, stop_id=None,
        direction_id="legacy-unspecified", horizon="day",
        bucket_start=datetime(2025, 10, 31, 21, tzinfo=UTC),
        predicted_passengers=42.0, lower_bound=None, upper_bound=None,
        capacity=None, model_version=source.model_version,
        generated_at=legacy_values["generated_at"],
    ))
    await sql_session.flush()
    legacy = await sql_session.get(ForecastRunModel, "independent-boarding-day")
    assert legacy is not None
    legacy.state = "published"
    await sql_session.flush()
    result_with_legacy = await repo.get_snapshot(route.id, ForecastHorizon.DAY)
    assert result_with_legacy is not None and result_with_legacy.run is not None
    assert result_with_legacy.run.run_id == "independent-boarding-day"

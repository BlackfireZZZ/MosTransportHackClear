from datetime import UTC, datetime, timedelta

import pytest
from conftest import isolated_session
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.domain.forecast import ForecastHorizon
from app.infrastructure.db.models import ForecastPointModel, RouteModel, RouteStopModel
from app.infrastructure.repositories.forecast import SqlAlchemyForecastRepository


@pytest.mark.parametrize("horizon,count", [("day", 24), ("month", 30), ("year", 12)])
@pytest.mark.parametrize("route_id,stop_ids", [(1, list(range(1, 7))), (2, list(range(7, 11)))])
async def test_migrated_seed_reads(
    sql_session: AsyncSession, horizon: str, count: int, route_id: int, stop_ids: list[int]
) -> None:
    repository = SqlAlchemyForecastRepository(sql_session)
    assert await repository.list_routes() == []
    snapshot = await repository.get_snapshot(route_id, ForecastHorizon(horizon))
    assert snapshot is not None
    assert snapshot.route.id == route_id
    assert snapshot.horizon.value == horizon
    assert snapshot.model_version == "graph-baseline-v1"
    assert snapshot.generated_at == datetime(2026, 9, 19, 9, tzinfo=UTC)
    assert len(snapshot.points) == count
    assert [point.timestamp for point in snapshot.points] == sorted(
        point.timestamp for point in snapshot.points
    )
    assert snapshot.points[0].predicted_passengers == pytest.approx(72 if route_id == 1 else 59)
    assert all(point.capacity == 180 for point in snapshot.points)
    assert [stop.id for stop in snapshot.stops] == stop_ids
    assert [stop.sequence for stop in snapshot.stops] == list(range(1, len(stop_ids) + 1))
    assert snapshot.stops[0].load_percent == pytest.approx(
        snapshot.stops[0].predicted_passengers / 180 * 100
    )


async def test_missing_route_and_missing_aggregate(sql_session: AsyncSession) -> None:
    repository = SqlAlchemyForecastRepository(sql_session)
    assert await repository.get_snapshot(999, ForecastHorizon.DAY) is None
    await sql_session.execute(
        delete(ForecastPointModel).where(
            ForecastPointModel.route_id == 1,
            ForecastPointModel.horizon == "day",
            ForecastPointModel.stop_id.is_(None),
        )
    )
    assert await repository.get_snapshot(1, ForecastHorizon.DAY) is None
    assert await repository.get_snapshot(1, ForecastHorizon.MONTH) is not None
    assert await repository.get_snapshot(2, ForecastHorizon.DAY) is not None


async def test_order_is_independent_of_insert_order(sql_session: AsyncSession) -> None:
    await sql_session.execute(
        delete(ForecastPointModel).where(
            ForecastPointModel.route_id == 1,
            ForecastPointModel.horizon == "day",
            ForecastPointModel.stop_id.is_(None),
        )
    )
    start = datetime(2026, 9, 20, tzinfo=UTC)
    for index in (2, 0, 1):
        sql_session.add(
            ForecastPointModel(
                id=9000 + index,
                run_id="legacy-day",
                route_id=1,
                stop_id=None,
                horizon="day",
                bucket_start=start + timedelta(hours=index),
                predicted_passengers=10 + index,
                lower_bound=0,
                upper_bound=20,
                capacity=180,
                model_version="graph-baseline-v1",
                generated_at=datetime(2026, 9, 19, 9, tzinfo=UTC),
            )
        )
    await sql_session.execute(delete(RouteStopModel).where(RouteStopModel.route_id == 1))
    await sql_session.execute(
        delete(ForecastPointModel).where(
            ForecastPointModel.route_id == 1, ForecastPointModel.stop_id.in_([4, 5, 6])
        )
    )
    for sequence in (3, 1, 2):
        sql_session.add(
            RouteStopModel(id=9000 + sequence, route_id=1, stop_id=sequence, sequence=sequence)
        )
    await sql_session.flush()
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)
    assert result is not None
    assert [point.predicted_passengers for point in result.points] == [10, 11, 12]
    assert [stop.sequence for stop in result.stops] == [1, 2, 3]


@pytest.mark.parametrize("fail", [False, True])
async def test_fixture_rolls_back_commits_even_after_failure(
    sql_engine: AsyncEngine, fail: bool
) -> None:
    class SimulatedFailure(Exception):
        pass

    try:
        async with isolated_session(sql_engine) as session:
            session.add(RouteModel(id=9000, number="fixture", name="fixture", color="#000000"))
            await session.commit()
            assert await session.get(RouteModel, 9000) is not None
            await session.rollback()
            if fail:
                raise SimulatedFailure
    except SimulatedFailure:
        pass
    async with isolated_session(sql_engine) as session:
        assert await session.get(RouteModel, 9000) is None
        assert await session.get(RouteModel, 1) is not None

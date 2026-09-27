from datetime import UTC, datetime

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.forecast import ForecastHorizon, ForecastSelection
from app.infrastructure.db.models import ForecastPointModel, ForecastRunModel
from app.infrastructure.repositories.forecast import SqlAlchemyForecastRepository


def point(**changes: object) -> ForecastPointModel:
    values = dict(
        run_id="legacy-day",
        route_id=1,
        stop_id=None,
        horizon="day",
        bucket_start=datetime(2027, 1, 1, tzinfo=UTC),
        predicted_passengers=10.0,
        lower_bound=5.0,
        upper_bound=15.0,
        capacity=20.0,
        model_version="graph-baseline-v1",
        generated_at=datetime(2026, 9, 19, 9, tzinfo=UTC),
    )
    values.update(changes)
    return ForecastPointModel(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"predicted_passengers": -1},
        {"predicted_passengers": float("nan")},
        {"predicted_passengers": float("inf")},
        {"capacity": -1},
        {"capacity": float("nan")},
        {"capacity": float("inf")},
        {"lower_bound": 11},
        {"upper_bound": 9},
        {"upper_bound": float("inf")},
        {"upper_bound": float("nan")},
        {"lower_bound": None},
        {"model_version": "wrong"},
        {"horizon": "month"},
        {"generated_at": datetime(2025, 1, 1, tzinfo=UTC)},
    ],
)
async def test_invalid_point_is_rejected(sql_session: AsyncSession, changes: dict) -> None:
    with pytest.raises(IntegrityError):
        async with sql_session.begin_nested():
            sql_session.add(point(**changes))
            await sql_session.flush()


async def test_duplicate_null_stop_is_rejected(sql_session: AsyncSession) -> None:
    sql_session.add(point())
    await sql_session.flush()
    with pytest.raises(IntegrityError):
        async with sql_session.begin_nested():
            sql_session.add(point())
            await sql_session.flush()


async def test_runs_coexist_and_draft_is_invisible(sql_session: AsyncSession) -> None:
    old = await sql_session.get(ForecastRunModel, "legacy-day")
    assert old is not None
    values = {column.name: getattr(old, column.name) for column in old.__table__.columns}
    values.update(id="new-draft", state="draft", model_version="new-model")
    sql_session.add(ForecastRunModel(**values))
    await sql_session.flush()
    sql_session.add(point(run_id="new-draft", model_version="new-model"))
    sql_session.add(point())
    await sql_session.flush()
    repository = SqlAlchemyForecastRepository(sql_session)
    result = await repository.get_snapshot(
        1,
        ForecastHorizon.DAY,
        ForecastSelection(
            start=datetime(2027, 1, 1, tzinfo=UTC),
            end=datetime(2027, 1, 2, tzinfo=UTC),
        ),
    )
    assert result is not None and result.model_version == "graph-baseline-v1"
    assert len([p for p in result.points if p.timestamp.year == 2027]) == 1


@pytest.mark.parametrize("capacity", [None, 0.0])
async def test_unavailable_capacity_does_not_divide(
    sql_session: AsyncSession, capacity: float | None
) -> None:
    await sql_session.execute(
        update(ForecastPointModel)
        .where(ForecastPointModel.run_id == "legacy-day")
        .values(capacity=capacity)
    )
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)
    assert result is not None
    assert result.peak_load_percent is None
    assert all(s.load_percent is None for s in result.stops)
    assert all(p.capacity == capacity for p in result.points)


async def test_unknown_bounds_are_preserved(sql_session: AsyncSession) -> None:
    sql_session.add(point(lower_bound=None, upper_bound=None))
    await sql_session.flush()
    result = await sql_session.scalar(
        select(ForecastPointModel).where(
            ForecastPointModel.bucket_start == datetime(2027, 1, 1, tzinfo=UTC)
        )
    )
    assert result is not None and result.lower_bound is None and result.upper_bound is None


async def test_run_metadata_is_immutable(sql_session: AsyncSession) -> None:
    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError, match="immutable"):
        async with sql_session.begin_nested():
            await sql_session.execute(
                update(ForecastRunModel)
                .where(ForecastRunModel.id == "legacy-day")
                .values(source_version="rewritten")
            )

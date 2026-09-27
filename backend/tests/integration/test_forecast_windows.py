from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.forecast import ForecastDataConflict, ForecastHorizon, ForecastSelection
from app.infrastructure.db.models import ForecastPointModel, RouteStopModel
from app.infrastructure.repositories.forecast import SqlAlchemyForecastRepository

START = datetime(2026, 9, 20, tzinfo=UTC)


def value(stop_id: int | None, hour: int, direction: str, prediction: float) -> ForecastPointModel:
    return ForecastPointModel(
        run_id="legacy-day",
        route_id=1,
        stop_id=stop_id,
        direction_id=direction,
        horizon="day",
        bucket_start=START + timedelta(hours=hour),
        predicted_passengers=prediction,
        lower_bound=prediction * 0.8,
        upper_bound=prediction * 1.2,
        capacity=100,
        model_version="graph-baseline-v1",
        generated_at=datetime(2026, 9, 19, 9, tzinfo=UTC),
    )


async def replace_values(session: AsyncSession) -> None:
    await session.execute(
        delete(ForecastPointModel).where(
            ForecastPointModel.route_id == 1, ForecastPointModel.run_id == "legacy-day"
        )
    )
    session.add_all(
        [
            value(None, 0, "outbound", 100),
            value(None, 0, "inbound", 200),
            value(None, 1, "outbound", 400),
            value(None, 1, "inbound", 500),
            value(1, 0, "outbound", 10),
            value(1, 0, "inbound", 20),
            value(1, 1, "outbound", 40),
            value(1, 1, "inbound", 50),
            value(2, 0, "outbound", 7),
        ]
    )
    await session.flush()


async def test_window_groups_directions_without_route_stop_double_count(
    sql_session: AsyncSession,
) -> None:
    await replace_values(sql_session)
    sql_session.add(RouteStopModel(id=9001, route_id=1, stop_id=1, sequence=7))
    await sql_session.flush()
    repository = SqlAlchemyForecastRepository(sql_session)
    result = await repository.get_snapshot(
        1, ForecastHorizon.DAY, ForecastSelection(start=START, end=START + timedelta(hours=1))
    )
    assert result is not None
    assert [point.predicted_passengers for point in result.points] == [300]
    assert result.points[0].lower_bound is None and result.points[0].capacity is None
    assert [(stop.id, stop.predicted_passengers) for stop in result.stops] == [(1, 30), (2, 7)]
    assert result.stop_points is not None
    assert [(point.stop_id, point.predicted_passengers) for point in result.stop_points] == [
        (1, 30),
        (2, 7),
    ]
    assert result.stop_points[0].direction_id is None
    assert result.stop_points[0].bucket_end is None
    assert result.run is not None and result.run.run_id == "legacy-day"
    assert result.run.identity_namespace == "serving-surrogate-integer"
    assert result.selection is not None and result.selection.end == START + timedelta(hours=1)
    selectors = await repository.list_stops(1)
    assert selectors is not None and [s.id for s in selectors] == list(range(1, 7))


async def test_stop_direction_selection_and_single_source_bounds(sql_session: AsyncSession) -> None:
    await replace_values(sql_session)
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(
        1,
        ForecastHorizon.DAY,
        ForecastSelection(
            stop_id=1, direction_id="outbound", start=START, end=START + timedelta(hours=2)
        ),
    )
    assert result is not None
    assert [p.predicted_passengers for p in result.points] == [10, 40]
    assert [(p.lower_bound, p.upper_bound, p.capacity) for p in result.points] == [
        (8, 12, 100),
        (32, 48, 100),
    ]
    assert len(result.stops) == 1 and result.stops[0].predicted_passengers == 50
    assert result.stops[0].load_percent is None
    assert result.stop_points is not None and all(
        p.direction_id == "outbound" for p in result.stop_points
    )


async def test_explicit_empty_window_and_unknown_stop(sql_session: AsyncSession) -> None:
    repository = SqlAlchemyForecastRepository(sql_session)
    selection = ForecastSelection(start=START + timedelta(days=2), end=START + timedelta(days=3))
    result = await repository.get_snapshot(1, ForecastHorizon.DAY, selection)
    assert result is None
    assert (
        await repository.get_snapshot(1, ForecastHorizon.DAY, ForecastSelection(stop_id=7)) is None
    )
    assert await repository.list_stops(999) is None


async def test_mixed_unspecified_direction_fails_explicitly(sql_session: AsyncSession) -> None:
    sql_session.add(value(None, 0, "outbound", 10))
    await sql_session.flush()
    with pytest.raises(ForecastDataConflict, match="directions overlap"):
        await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)


async def test_offset_window_selects_same_instant(sql_session: AsyncSession) -> None:
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(
        1,
        ForecastHorizon.DAY,
        ForecastSelection(
            start=datetime.fromisoformat("2026-09-20T03:00:00+03:00"),
            end=datetime.fromisoformat("2026-09-20T04:00:00+03:00"),
        ),
    )
    assert result is not None and len(result.points) == 1
    assert result.points[0].timestamp == START


async def test_bounded_query_plan(
    sql_session: AsyncSession, capsys: pytest.CaptureFixture[str]
) -> None:
    await sql_session.execute(
        text("""
        INSERT INTO forecast_points
          (run_id,route_id,stop_id,direction_id,horizon,bucket_start,predicted_passengers,
           lower_bound,upper_bound,capacity,model_version,generated_at)
        SELECT 'legacy-day',1,NULL,'explain-only','day',
               '2030-01-01'::timestamptz + n * interval '1 minute', 1,0,2,NULL,
               'graph-baseline-v1','2026-09-19 09:00:00+00'::timestamptz
        FROM generate_series(1,10000) n
    """)
    )
    await sql_session.execute(text("ANALYZE forecast_points"))
    rows = (
        (
            await sql_session.execute(
                text("""
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT stop_id,bucket_start,sum(predicted_passengers::text::numeric),count(*)
        FROM forecast_points WHERE run_id='legacy-day' AND route_id=1
          AND bucket_start >= '2026-09-20T00:00:00Z'
          AND bucket_start < '2026-09-21T00:00:00Z'
        GROUP BY stop_id,bucket_start ORDER BY bucket_start,stop_id LIMIT 31032
    """)
            )
        )
        .scalars()
        .all()
    )
    plan = "\n".join(rows)
    assert "Index" in plan and "bucket_start" in plan and "Execution Time" in plan
    with capsys.disabled():
        print("TASK028 EXPLAIN synthetic10000 rows:\n" + plan)


async def test_default_window_is_resolved_for_selected_direction(sql_session: AsyncSession) -> None:
    sql_session.add(
        ForecastPointModel(
            run_id="legacy-day",
            route_id=1,
            stop_id=None,
            direction_id="later",
            horizon="day",
            bucket_start=START + timedelta(days=2),
            predicted_passengers=10,
            lower_bound=None,
            upper_bound=None,
            capacity=None,
            model_version="graph-baseline-v1",
            generated_at=datetime(2026, 9, 19, 9, tzinfo=UTC),
        )
    )
    await sql_session.flush()
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(
        1, ForecastHorizon.DAY, ForecastSelection(direction_id="later")
    )
    assert result is not None and len(result.points) == 1
    assert result.points[0].timestamp == START + timedelta(days=2)


async def test_finite_source_values_cannot_overflow_aggregate(sql_session: AsyncSession) -> None:
    await replace_values(sql_session)
    from sqlalchemy import update

    await sql_session.execute(
        update(ForecastPointModel)
        .where(
            ForecastPointModel.run_id == "legacy-day",
            ForecastPointModel.route_id == 1,
            ForecastPointModel.stop_id.is_(None),
        )
        .values(predicted_passengers=1e308, lower_bound=None, upper_bound=None)
    )
    with pytest.raises(ForecastDataConflict, match="finite range"):
        await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)


async def test_forecast_stop_must_belong_to_route(sql_session: AsyncSession) -> None:
    sql_session.add(value(7, 0, "outbound", 10))
    await sql_session.flush()
    with pytest.raises(ForecastDataConflict, match="route catalog"):
        await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)


async def test_stored_default_window_overflow_fails_explicitly(sql_session: AsyncSession) -> None:
    sql_session.add(
        ForecastPointModel(
            run_id="legacy-day",
            route_id=1,
            stop_id=None,
            direction_id="far-future",
            horizon="day",
            bucket_start=datetime(9999, 12, 31, tzinfo=UTC),
            predicted_passengers=10,
            lower_bound=None,
            upper_bound=None,
            capacity=None,
            model_version="graph-baseline-v1",
            generated_at=datetime(2026, 9, 19, 9, tzinfo=UTC),
        )
    )
    await sql_session.flush()
    with pytest.raises(ForecastDataConflict, match="supported range"):
        await SqlAlchemyForecastRepository(sql_session).get_snapshot(
            1, ForecastHorizon.DAY, ForecastSelection(direction_id="far-future")
        )


async def test_filtered_http_contract_and_stop_catalog(sql_session: AsyncSession) -> None:
    from httpx import ASGITransport, AsyncClient

    from app.api.dependencies import get_forecast_service
    from app.application.services.forecast import ForecastService
    from app.main import create_app

    application = create_app()
    application.dependency_overrides[get_forecast_service] = lambda: ForecastService(
        SqlAlchemyForecastRepository(sql_session)
    )
    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/forecasts",
            params={
                "route_id": 1,
                "stop_id": 1,
                "start": "2026-09-20T03:00:00+03:00",
                "end": "2026-09-20T04:00:00+03:00",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["run"]["run_id"] == "legacy-day"
        assert body["selection"]["aggregation_key"] == "stop_bucket"
        assert (
            body["points"][0]["predicted_passengers"]
            == body["stop_points"][0]["predicted_passengers"]
        )
        assert len(body["points"]) == 1 and body["stop_points"][0]["stop_id"] == 1
        assert body["stop_points"][0]["bucket_end"] is None
        catalog = await client.get("/api/v1/routes/1/stops")
        assert catalog.status_code == 200 and len(catalog.json()) == 6
        assert (await client.get("/api/v1/routes/999/stops")).status_code == 404
        empty = await client.get(
            "/api/v1/forecasts",
            params={
                "route_id": 1,
                "start": (START + timedelta(days=2)).isoformat(),
                "end": (START + timedelta(days=3)).isoformat(),
            },
        )
        assert empty.status_code == 404
        assert empty.json()["detail"] == (
            "No forecast points for the selected route, horizon and interval"
        )


async def test_two_published_runs_never_mix_point_and_stop_values(
    sql_session: AsyncSession,
) -> None:
    from app.infrastructure.db.models import ForecastRunModel

    old = await sql_session.get(ForecastRunModel, "legacy-day")
    assert old is not None
    metadata = {column.name: getattr(old, column.name) for column in old.__table__.columns}
    metadata.update(id="new-published", model_version="new-model")
    sql_session.add(ForecastRunModel(**metadata))
    await sql_session.flush()
    for stop_id in (None, 1):
        point = value(stop_id, 0, "outbound", 777)
        point.run_id = "new-published"
        point.model_version = "new-model"
        sql_session.add(point)
    await sql_session.flush()
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)
    assert result is not None and result.model_version == "new-model"
    assert result.run is not None and result.run.run_id == "new-published"
    assert [p.predicted_passengers for p in result.points] == [777]
    assert [s.predicted_passengers for s in result.stops] == [777]
    assert result.stop_points is not None and [
        p.predicted_passengers for p in result.stop_points
    ] == [777]


async def test_unrepresentable_load_fails_before_http_serialization(
    sql_session: AsyncSession,
) -> None:
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import update

    from app.api.dependencies import get_forecast_service
    from app.application.services.forecast import ForecastService
    from app.main import create_app

    await sql_session.execute(
        update(ForecastPointModel)
        .where(
            ForecastPointModel.run_id == "legacy-day",
            ForecastPointModel.route_id == 1,
        )
        .values(predicted_passengers=1e308, lower_bound=None, upper_bound=None, capacity=1e-308)
    )
    application = create_app()
    application.dependency_overrides[get_forecast_service] = lambda: ForecastService(
        SqlAlchemyForecastRepository(sql_session)
    )
    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/forecasts", params={"route_id": 1})
        assert response.status_code == 409
        assert response.json()["detail"] == "load percentage exceeds finite range"


async def test_single_source_preserves_float_precision(sql_session: AsyncSession) -> None:
    from sqlalchemy import update

    expected = 1.2345678901234567
    await sql_session.execute(
        update(ForecastPointModel)
        .where(
            ForecastPointModel.run_id == "legacy-day",
            ForecastPointModel.route_id == 1,
            ForecastPointModel.stop_id.is_(None),
        )
        .values(predicted_passengers=expected, lower_bound=None, upper_bound=None)
    )
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)
    assert result is not None and all(
        point.predicted_passengers == expected for point in result.points
    )


async def test_fractional_direction_sum_preserves_round_trip_precision(
    sql_session: AsyncSession,
) -> None:
    from sqlalchemy import update

    await replace_values(sql_session)
    first, second = 1.2345678901234567, 2.3456789012345678
    for direction, prediction in [("outbound", first), ("inbound", second)]:
        await sql_session.execute(
            update(ForecastPointModel)
            .where(
                ForecastPointModel.run_id == "legacy-day",
                ForecastPointModel.route_id == 1,
                ForecastPointModel.stop_id.is_(None),
                ForecastPointModel.direction_id == direction,
            )
            .values(predicted_passengers=prediction, lower_bound=0, upper_bound=4)
        )
    result = await SqlAlchemyForecastRepository(sql_session).get_snapshot(1, ForecastHorizon.DAY)
    assert result is not None and result.points[0].predicted_passengers == float(
        Decimal(str(first)) + Decimal(str(second))
    )


@pytest.mark.parametrize("mode", ["directions", "time", "single"])
async def test_occupancy_rejects_unsupported_sums_and_preserves_single_bucket(
    sql_session: AsyncSession,
    mode: str,
) -> None:
    from httpx import ASGITransport, AsyncClient

    from app.api.dependencies import get_forecast_service
    from app.application.services.forecast import ForecastService
    from app.infrastructure.db.models import ForecastRunModel
    from app.main import create_app

    old = await sql_session.get(ForecastRunModel, "legacy-day")
    assert old is not None
    metadata = {column.name: getattr(old, column.name) for column in old.__table__.columns}
    metadata.update(id="occupancy-test", target="onboard_load", unit="passengers", synthetic=False)
    sql_session.add(ForecastRunModel(**metadata))
    await sql_session.flush()
    specs = [(0, "outbound", 10)]
    if mode == "directions":
        specs.append((0, "inbound", 20))
    if mode == "time":
        specs.append((1, "outbound", 20))
    for hour, direction, prediction in specs:
        point = value(1, hour, direction, prediction)
        point.run_id = "occupancy-test"
        sql_session.add(point)
    await sql_session.flush()
    application = create_app()
    application.dependency_overrides[get_forecast_service] = lambda: ForecastService(
        SqlAlchemyForecastRepository(sql_session)
    )
    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/forecasts",
            params={
                "route_id": 1,
                "stop_id": 1,
                "start": START.isoformat(),
                "end": (START + timedelta(hours=2)).isoformat(),
            },
        )
        if mode == "single":
            assert response.status_code == 200, response.text
            assert response.json()["points"][0]["predicted_passengers"] == 10
            assert response.json()["run"]["target"] == "onboard_load"
        else:
            assert response.status_code == 409, response.text
            assert (
                response.json()["detail"]
                == "forecast target cannot be summed across sources or time buckets"
            )

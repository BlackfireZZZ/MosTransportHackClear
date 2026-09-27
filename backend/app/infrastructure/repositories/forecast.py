import math
from calendar import monthrange
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import Numeric, Text, cast, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.forecast import (
    BUCKET_LIMITS,
    WINDOW_LIMITS,
    ForecastDataConflict,
    ForecastHorizon,
    ForecastPoint,
    ForecastQueryError,
    ForecastRunMetadata,
    ForecastSelection,
    ForecastSnapshot,
    ResolvedForecastSelection,
    RouteStopSummary,
    RouteSummary,
    StopForecastPoint,
    StopLoad,
    finite_load_percent,
    require_additive_target,
)
from app.infrastructure.db.models import (
    ForecastActiveRunModel,
    ForecastPointModel,
    ForecastRunModel,
    RouteModel,
    RouteStopModel,
    StopModel,
)

MAX_STOPS = 1000
MAX_GROUPS = MAX_STOPS * 31 + 31
MOSCOW = ZoneInfo("Europe/Moscow")
SCORED_ROUTE_NUMBERS = frozenset({"1", "5", "7", "11", "12", "17", "25", "26", "28", "50"})


class SqlAlchemyForecastRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_routes(self) -> list[RouteSummary]:
        result = await self._session.execute(
            select(RouteModel)
            .join(ForecastActiveRunModel, ForecastActiveRunModel.route_id == RouteModel.id)
            .where(ForecastActiveRunModel.horizon == ForecastHorizon.DAY.value)
            .order_by(RouteModel.number)
        )
        rows = list(result.scalars())
        rows.sort(
            key=lambda row: (0, int(row.number)) if row.number.isdecimal() else (1, row.number)
        )
        return [RouteSummary(row.id, row.number, row.name, row.color) for row in rows]

    async def list_stops(self, route_id: int) -> list[RouteStopSummary] | None:
        if await self._session.get(RouteModel, route_id) is None:
            return None
        result = await self._session.execute(
            select(StopModel, func.min(RouteStopModel.sequence).label("sequence"))
            .join(RouteStopModel, RouteStopModel.stop_id == StopModel.id)
            .where(RouteStopModel.route_id == route_id)
            .group_by(StopModel.id)
            .order_by(func.min(RouteStopModel.sequence), StopModel.id)
            .limit(MAX_STOPS + 1)
        )
        rows = result.all()
        if len(rows) > MAX_STOPS:
            raise ForecastQueryError("route exceeds the stop response limit")
        return [
            RouteStopSummary(s.id, s.name, s.latitude, s.longitude, sequence)
            for s, sequence in rows
        ]

    async def get_snapshot(
        self, route_id: int, horizon: ForecastHorizon, selection: ForecastSelection | None = None
    ) -> ForecastSnapshot | None:
        chosen = selection or ForecastSelection()
        chosen.validate(horizon)
        route = await self._session.get(RouteModel, route_id)
        if route is None:
            return None
        if route.number in SCORED_ROUTE_NUMBERS and horizon == ForecastHorizon.YEAR:
            return None
        stops = await self.list_stops(route_id)
        assert stops is not None
        if chosen.stop_id is not None and chosen.stop_id not in {stop.id for stop in stops}:
            return None
        active_id = await self._session.scalar(
            select(ForecastActiveRunModel.run_id).where(
                ForecastActiveRunModel.route_id == route_id,
                ForecastActiveRunModel.horizon == horizon.value,
            )
        )
        if active_id is None:
            run_statement = (
                select(ForecastRunModel)
                .where(
                    ForecastRunModel.horizon == horizon.value,
                    ForecastRunModel.state == "published",
                    exists().where(
                        ForecastPointModel.run_id == ForecastRunModel.id,
                        ForecastPointModel.route_id == route_id,
                    ),
                )
                .order_by(ForecastRunModel.generated_at.desc(), ForecastRunModel.id.desc())
                .limit(1)
            )
            if route.number in SCORED_ROUTE_NUMBERS and horizon in (
                ForecastHorizon.DAY, ForecastHorizon.MONTH
            ):
                run_statement = run_statement.where(
                    or_(
                        ForecastRunModel.target != "validation_count",
                        func.complete_scored_route_run(ForecastRunModel.id),
                    )
                )
            run = await self._session.scalar(run_statement)
        else:
            run = await self._session.get(ForecastRunModel, active_id)
        if run is None:
            return None
        point = ForecastPointModel
        entity = (
            point.stop_id.is_(None) if chosen.stop_id is None else point.stop_id == chosen.stop_id
        )
        start, end = chosen.start, chosen.end
        if start is None:
            first_bucket = (
                select(point.bucket_start)
                .where(point.run_id == run.id, point.route_id == route_id, entity)
                .order_by(point.bucket_start)
                .limit(1)
            )
            if chosen.direction_id is not None:
                first_bucket = first_bucket.where(point.direction_id == chosen.direction_id)
            start = await self._session.scalar(first_bucket)
            if start is None:
                return None
            try:
                if horizon == ForecastHorizon.MONTH:
                    local = start.astimezone(MOSCOW)
                    days = monthrange(local.year, local.month)[1]
                    next_month = local.date().replace(day=1) + timedelta(days=days)
                    end = datetime.combine(next_month, datetime.min.time(), MOSCOW).astimezone(UTC)
                else:
                    end = start + WINDOW_LIMITS[horizon]
            except OverflowError as error:
                raise ForecastDataConflict(
                    "stored forecast window exceeds supported range"
                ) from error
        assert end is not None
        # Direct float8-to-numeric conversion rounds significant digits; round-trip text
        # preserves serialized values while numeric SUM avoids float accumulator overflow.
        statement = (
            select(
                point.stop_id,
                point.bucket_start,
                func.sum(cast(cast(point.predicted_passengers, Text), Numeric)).label("predicted"),
                func.min(point.predicted_passengers).label("single_prediction"),
                func.min(point.lower_bound).label("lower"),
                func.max(point.upper_bound).label("upper"),
                func.min(point.capacity).label("capacity"),
                func.count().label("sources"),
                func.min(point.direction_id).label("direction"),
                func.count()
                .filter(point.direction_id == "legacy-unspecified")
                .label("unspecified"),
            )
            .where(
                point.run_id == run.id,
                point.route_id == route_id,
                point.bucket_start >= start,
                point.bucket_start < end,
            )
            .group_by(point.stop_id, point.bucket_start)
            .order_by(point.bucket_start, point.stop_id)
            .limit(MAX_GROUPS + 1)
        )
        if chosen.stop_id is not None:
            statement = statement.where(point.stop_id == chosen.stop_id)
        if chosen.direction_id is not None:
            statement = statement.where(point.direction_id == chosen.direction_id)
        rows = (await self._session.execute(statement)).all()
        if len(rows) > MAX_GROUPS:
            raise ForecastQueryError("selection exceeds the grouped response limit")
        points: list[ForecastPoint] = []
        stop_points: list[StopForecastPoint] = []
        by_stop: dict[int, list[ForecastPoint]] = defaultdict(list)
        stop_ids = {stop.id for stop in stops}
        for row in rows:
            require_additive_target(run.target, run.unit, row.sources)
            if row.stop_id is not None and row.stop_id not in stop_ids:
                raise ForecastDataConflict("forecast stop is not in the selected route catalog")
            predicted = row.single_prediction if row.sources == 1 else float(row.predicted)
            if not math.isfinite(predicted):
                raise ForecastDataConflict("aggregate passenger count exceeds finite range")
            if row.unspecified and row.sources > 1:
                raise ForecastDataConflict(
                    "unspecified and explicit directions overlap in one bucket"
                )
            value = ForecastPoint(
                timestamp=row.bucket_start,
                predicted_passengers=predicted,
                lower_bound=row.lower if row.sources == 1 else None,
                upper_bound=row.upper if row.sources == 1 else None,
                capacity=row.capacity if row.sources == 1 else None,
            )
            finite_load_percent(value.predicted_passengers, value.capacity)
            if row.stop_id == chosen.stop_id:
                points.append(value)
            if row.stop_id is not None:
                by_stop[row.stop_id].append(value)
                stop_points.append(
                    StopForecastPoint(
                        stop_id=row.stop_id,
                        timestamp=row.bucket_start,
                        bucket_end=None,
                        direction_id=row.direction
                        if row.sources == 1 and not row.unspecified
                        else None,
                        predicted_passengers=value.predicted_passengers,
                        lower_bound=value.lower_bound,
                        upper_bound=value.upper_bound,
                        capacity=value.capacity,
                        aggregation_scope="stop_bucket_direction"
                        if chosen.direction_id
                        else "stop_bucket_all_directions",
                    )
                )
        if len(points) > BUCKET_LIMITS[horizon] or any(
            len(values) > BUCKET_LIMITS[horizon] for values in by_stop.values()
        ):
            raise ForecastQueryError("selection exceeds the horizon bucket limit")
        if not points:
            return None
        loads = []
        for stop in stops:
            values = by_stop.get(stop.id, [])
            if not values:
                continue
            require_additive_target(run.target, run.unit, len(values))
            predicted = sum(value.predicted_passengers for value in values)
            if not math.isfinite(predicted):
                raise ForecastDataConflict("aggregate passenger count exceeds finite range")
            capacity = values[0].capacity if len(values) == 1 else None
            loads.append(
                StopLoad(
                    stop.id,
                    stop.name,
                    stop.latitude,
                    stop.longitude,
                    predicted,
                    finite_load_percent(predicted, capacity),
                    stop.sequence,
                )
            )
        return ForecastSnapshot(
            route=RouteSummary(route.id, route.number, route.name, route.color),
            horizon=horizon,
            generated_at=run.generated_at,
            model_version=run.model_version,
            points=points,
            stops=loads,
            stop_points=stop_points,
            run=ForecastRunMetadata(
                run_id=run.id,
                dataset_id=run.dataset_id,
                source_version=run.source_version,
                feature_version=run.feature_version,
                entity_version=run.entity_version,
                calendar_version=run.calendar_version,
                graph_version=run.graph_version,
                target=run.target,
                unit=run.unit,
                synthetic=run.synthetic,
                forecast_origin=run.forecast_origin,
                data_cutoff=run.data_cutoff,
                interval_level=run.interval_level,
                interval_method=run.interval_method,
            ),
            selection=ResolvedForecastSelection(
                start=start,
                end=end,
                stop_id=chosen.stop_id,
                direction_id=chosen.direction_id,
                aggregation_key="route_bucket" if chosen.stop_id is None else "stop_bucket",
            ),
        )

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.services.forecast import ForecastService
from app.application.services.overpass import OverpassService
from app.application.services.planning import PlanningService
from app.application.services.tram_graph import TramNetworkService
from app.core.config import settings
from app.infrastructure.db.session import get_session
from app.infrastructure.forecast_geometry import FileForecastMapProvider
from app.infrastructure.overpass import HttpOverpassGateway
from app.infrastructure.planning import FilePlanningRepository
from app.infrastructure.repositories.forecast import SqlAlchemyForecastRepository
from app.infrastructure.repositories.tram_graph import FileTramGraphRepository

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@lru_cache
def get_forecast_map_provider() -> FileForecastMapProvider:
    return FileForecastMapProvider(
        settings.forecast_geometry_mapping,
        settings.tram_graph_json.parent,
    )


def get_forecast_service(session: SessionDep) -> ForecastService:
    return ForecastService(SqlAlchemyForecastRepository(session), get_forecast_map_provider())


ForecastServiceDep = Annotated[ForecastService, Depends(get_forecast_service)]


@lru_cache
def get_tram_graph_repository() -> FileTramGraphRepository:
    """Cached so the ~1 MB of committed JSON is parsed once per process, not per request."""
    return FileTramGraphRepository(settings.tram_graph_json, settings.tram_graph_geojson)


TramGraphRepositoryDep = Annotated[FileTramGraphRepository, Depends(get_tram_graph_repository)]


def get_tram_network_service(repository: TramGraphRepositoryDep) -> TramNetworkService:
    # Injected rather than called: calling it directly puts the repository outside the
    # dependency graph, where dependency_overrides cannot reach it and a test has to
    # reach through the cache instead of replacing it.
    return TramNetworkService(repository)


TramNetworkServiceDep = Annotated[TramNetworkService, Depends(get_tram_network_service)]


def get_overpass_service() -> OverpassService:
    return OverpassService(
        HttpOverpassGateway(
            interpreter_url=settings.overpass_interpreter_url,
            status_url=settings.overpass_status_url,
            query_timeout=settings.overpass_timeout,
            health_timeout=settings.overpass_health_timeout,
        ),
        max_query_chars=settings.overpass_max_query_chars,
    )


OverpassServiceDep = Annotated[OverpassService, Depends(get_overpass_service)]


def get_overpass_query_enabled() -> bool:
    return settings.overpass_query_enabled


OverpassQueryEnabledDep = Annotated[bool, Depends(get_overpass_query_enabled)]


@lru_cache
def get_planning_service() -> PlanningService:
    return PlanningService(FilePlanningRepository(settings.tram_graph_json.parent / "planning"))


PlanningServiceDep = Annotated[PlanningService, Depends(get_planning_service)]

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.observability import ObservabilityMiddleware
from app.api.router import api_router
from app.core.config import settings
from app.infrastructure.db.session import engine
from app.infrastructure.observability import configure_http_logger


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    configure_http_logger(settings.log_level)
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Прогноз пассажиропотока московского наземного транспорта",
        lifespan=lifespan,
    )
    # Order matters: add_middleware prepends, so the LAST one added is outermost.
    # CORS must be outermost, otherwise the JSON error ObservabilityMiddleware
    # produces on a crash never gets its Access-Control-Allow-Origin and a browser
    # discards it as a CORS failure instead of showing the error.
    application.add_middleware(ObservabilityMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).rstrip("/") for origin in settings.backend_cors_origins],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()

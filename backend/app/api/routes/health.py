from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import SessionDep, TramNetworkServiceDep
from app.domain.tram_graph import TramGraphDataError

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready")
async def readiness(session: SessionDep, tram_network: TramNetworkServiceDep) -> dict[str, str]:
    """Ready means every dependency needed to answer is in place, not just the database.

    The graph is loaded here rather than on the first request that needs it, so a
    deployment shipped without it reports unready instead of looking healthy and then
    failing whichever request happens to arrive first.
    """
    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not ready",
        ) from exc
    try:
        await tram_network.ensure_available()
    except TramGraphDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Tram graph is not ready: {exc}",
        ) from exc
    return {"status": "ready"}

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import OverpassQueryEnabledDep, OverpassServiceDep
from app.domain.overpass import OverpassError, OverpassQueryTooLongError
from app.schemas.overpass import (
    OverpassQueryRequest,
    OverpassQueryResponse,
    OverpassStatusResponse,
)

router = APIRouter(prefix="/overpass", tags=["overpass"])


@router.get("/status", response_model=OverpassStatusResponse)
async def get_status(service: OverpassServiceDep) -> OverpassStatusResponse:
    """Probe the upstream instance.

    Always 200: an unreachable Overpass is a fact for the dispatcher UI to display,
    not a failure of this API. Readiness of the service itself is `/health/ready`.
    """
    return OverpassStatusResponse.model_validate(await service.check_status())


@router.post("/query", response_model=OverpassQueryResponse)
async def run_query(
    request: OverpassQueryRequest, service: OverpassServiceDep, enabled: OverpassQueryEnabledDep
) -> OverpassQueryResponse:
    if not enabled:
        raise HTTPException(status_code=403, detail="Overpass query proxy is disabled")
    try:
        result = await service.run_query(request.query)
    except OverpassQueryTooLongError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except OverpassError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error
    return OverpassQueryResponse(result=result.payload, remark=result.remark)

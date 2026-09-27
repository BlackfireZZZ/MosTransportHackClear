from fastapi import APIRouter, HTTPException

from app.api.dependencies import PlanningServiceDep
from app.application.services.planning import PlanningUnavailable
from app.schemas.planning import PlanningRequest, PlanningResponse

router = APIRouter(tags=["planning"])


@router.post("/planning/forecast", response_model=PlanningResponse)
def planning_forecast(request: PlanningRequest, service: PlanningServiceDep) -> PlanningResponse:
    try:
        result = service.forecast(request.model_dump(mode="json"))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except PlanningUnavailable as error:
        raise HTTPException(status_code=503, detail="Planning artifact unavailable") from error
    return PlanningResponse.model_validate(result)

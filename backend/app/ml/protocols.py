from typing import Protocol

from app.domain.forecast import ForecastHorizon, ForecastSnapshot


class ForecastModel(Protocol):
    """Stable seam between feature pipelines and forecast publication."""

    @property
    def version(self) -> str: ...

    async def predict(self, route_id: int, horizon: ForecastHorizon) -> ForecastSnapshot: ...

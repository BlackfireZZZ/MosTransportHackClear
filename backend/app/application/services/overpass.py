"""Use case for the Overpass passthrough.

The only decision the application makes is whether a query is small enough to be
worth a slot on the shared instance; everything about HTTP, error pages and JSON
lives behind the gateway port.
"""

from typing import Protocol

from app.domain.overpass import OverpassQueryTooLongError, OverpassResult, OverpassStatus


class OverpassGateway(Protocol):
    async def run_query(self, query: str) -> OverpassResult: ...

    async def check_status(self) -> OverpassStatus: ...


class OverpassService:
    def __init__(self, gateway: OverpassGateway, max_query_chars: int) -> None:
        self._gateway = gateway
        self._max_query_chars = max_query_chars

    async def run_query(self, query: str) -> OverpassResult:
        if len(query) > self._max_query_chars:
            raise OverpassQueryTooLongError(
                f"query is {len(query)} characters, limit is {self._max_query_chars}"
            )
        return await self._gateway.run_query(query)

    async def check_status(self) -> OverpassStatus:
        """A dead upstream is a fact to report, not a failure of this endpoint."""
        return await self._gateway.check_status()

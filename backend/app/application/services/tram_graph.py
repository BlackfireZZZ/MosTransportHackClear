"""Read-only use cases over the committed tram graph.

The service depends on a port, not on a file or a table: the graph happens to be a
static asset today (see `docs/decisions/0003-tram-graph-file-repository.md`), and
moving it into PostgreSQL later must not reach past this module.
"""

from typing import Protocol

from app.domain.tram_graph import (
    NetworkGeometry,
    NetworkStats,
    RouteDetail,
    StopDetail,
    TramEdge,
    TramNetwork,
    TramRoute,
    TramStop,
)
from app.domain.tram_pathfinding import TramPath, find_path


class TramGraphRepository(Protocol):
    async def get_network(self) -> TramNetwork: ...


class TramNetworkService:
    def __init__(self, repository: TramGraphRepository) -> None:
        self._repository = repository

    async def ensure_available(self) -> None:
        """Load the graph, so a readiness probe fails on a broken one."""
        await self._repository.get_network()

    async def stats(self) -> NetworkStats:
        network = await self._repository.get_network()
        return network.stats()

    async def list_routes(self) -> tuple[TramRoute, ...]:
        network = await self._repository.get_network()
        return tuple(network.routes.values())

    async def get_route(self, ref: str) -> RouteDetail | None:
        network = await self._repository.get_network()
        return network.route_detail(ref)

    async def search_stops(self, query: str | None, limit: int) -> tuple[TramStop, ...]:
        network = await self._repository.get_network()
        return network.search_stops(query, limit)

    async def get_stop(self, stop_id: int) -> StopDetail | None:
        network = await self._repository.get_network()
        return network.stop_detail(stop_id)

    async def list_edges(self, ref: str | None) -> tuple[TramEdge, ...] | None:
        """Every edge, or a route's edges. None means the ref is unknown."""
        network = await self._repository.get_network()
        if ref is None:
            return network.edges
        route = network.routes.get(ref)
        return None if route is None else route.edges

    async def geometry(self, ref: str | None) -> NetworkGeometry | None:
        """Track geometry, optionally cut to one route. None means the ref is unknown."""
        network = await self._repository.get_network()
        if ref is not None and ref not in network.routes:
            return None
        return network.geometry(ref)

    async def find_path(self, source: int, target: int) -> TramPath:
        """Cheapest ride by track distance.

        Never raises for an unroutable pair: the network has two disconnected
        components and directed edges, so "there is no such ride" is an answer.
        """
        network = await self._repository.get_network()
        return find_path(network, source, target)

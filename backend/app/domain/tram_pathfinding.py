"""Routing over the tram network.

Separated from `tram_graph`, which holds the graph and its indexes. This module
is the one place with an algorithm, and the multimodal graph in the product
concept will only grow it. The dependency runs one way -- routing knows the
network, the network does not know routing -- so there is no import cycle.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import StrEnum

from app.domain.tram_graph import (
    Coordinate,
    GeometryQuality,
    TramEdge,
    TramNetwork,
    TramStop,
    sorted_refs,
)


class PathAbsence(StrEnum):
    """Why no path was returned, in a form a client can branch on."""

    UNKNOWN_STOP = "unknown_stop"
    DIFFERENT_COMPONENTS = "different_components"
    WRONG_DIRECTION = "wrong_direction"


@dataclass(frozen=True, slots=True)
class TramPath:
    """The result of a routing request.

    `found=False` with a populated `reason` is a successful answer describing the
    absence of a route, not a failure: the network genuinely has unreachable pairs.

    `reason` is English prose for a human reading the API directly. A client that
    renders in another language should branch on `reason_code` and write its own text.
    """

    found: bool
    reason: str | None
    reason_code: PathAbsence | None
    stops: tuple[TramStop, ...]
    total_length_m: float
    geometry: tuple[Coordinate, ...]
    routes: tuple[str, ...]
    geometry_quality: GeometryQuality | None = None
    missing_geometry_edges: int = 0


def absent_path(code: PathAbsence, reason: str) -> TramPath:
    return TramPath(
        found=False,
        reason=reason,
        reason_code=code,
        stops=(),
        total_length_m=0.0,
        geometry=(),
        routes=(),
    )


def find_path(network: TramNetwork, source: int, target: int) -> TramPath:
    for label, stop_id in (("from", source), ("to", target)):
        if stop_id not in network.stops:
            return absent_path(
                PathAbsence.UNKNOWN_STOP, f"unknown stop id {stop_id} in '{label}'"
            )

    if network.component_of[source] != network.component_of[target]:
        return absent_path(
            PathAbsence.DIFFERENT_COMPONENTS,
            f"{network.stops[source].name} and {network.stops[target].name} are in different "
            "parts of the tram network; there is no track connecting them",
        )

    chain = _cheapest_chain(network, source, target)
    if chain is None:
        return absent_path(
            PathAbsence.WRONG_DIRECTION,
            f"no route from {network.stops[source].name} to {network.stops[target].name} in the "
            "direction of travel; the track exists but only the other way round",
        )

    stops = [network.stops[source]] + [network.stops[edge.target] for edge in chain]
    total = 0.0
    routes: set[str] = set()
    geometry: list[Coordinate] = []
    for edge in chain:
        total += edge.length_m
        routes.update(edge.routes)
        coordinates = list(network.track_coordinates(edge))
        if geometry and coordinates and geometry[-1] == coordinates[0]:
            coordinates = coordinates[1:]
        geometry.extend(coordinates)
    if not geometry:
        geometry = [(stops[0].longitude, stops[0].latitude)]

    return TramPath(
        found=True,
        reason=None,
        reason_code=None,
        stops=tuple(stops),
        total_length_m=round(total, 1),
        geometry=tuple(geometry),
        routes=sorted_refs(routes),
        missing_geometry_edges=sum(
            network.geometry_quality(edge) == GeometryQuality.INFERRED for edge in chain
        ),
        geometry_quality=(
            GeometryQuality.INFERRED
            if any(network.geometry_quality(edge) == GeometryQuality.INFERRED for edge in chain)
            else GeometryQuality.SYNTHETIC if network.metadata.synthetic
            else GeometryQuality.PROVIDED
        ),
    )

def _cheapest_chain(network: TramNetwork, source: int, target: int) -> list[TramEdge] | None:
    """Dijkstra over `length_m`, returning the edges travelled, or None."""
    if source == target:
        return []
    distances: dict[int, float] = {source: 0.0}
    arrival: dict[int, TramEdge] = {}
    visited: set[int] = set()
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == target:
            break
        for edge in network.outgoing.get(current, ()):
            candidate = distance + edge.length_m
            if candidate < distances.get(edge.target, float("inf")):
                distances[edge.target] = candidate
                arrival[edge.target] = edge
                heapq.heappush(queue, (candidate, edge.target))

    if target not in arrival:
        return None
    chain: list[TramEdge] = []
    cursor = target
    while cursor != source:
        edge = arrival[cursor]
        chain.append(edge)
        cursor = edge.source
    chain.reverse()
    return chain

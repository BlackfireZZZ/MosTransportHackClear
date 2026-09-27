"""Great-circle distance on a mean-radius sphere; adequate at stop-tolerance scale."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

EARTH_RADIUS_METRES = 6_371_008.8


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("latitude/longitude outside the valid range")


def haversine_metres(origin: GeoPoint, target: GeoPoint) -> float:
    lat1, lon1 = math.radians(origin.latitude), math.radians(origin.longitude)
    lat2, lon2 = math.radians(target.latitude), math.radians(target.longitude)
    chord = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METRES * math.asin(math.sqrt(chord))


def stops_within(
    position: GeoPoint, candidates: Mapping[str, GeoPoint], tolerance_metres: float
) -> tuple[tuple[float, str], ...]:
    """Candidates within the closed tolerance, nearest first, ties broken by stop id."""
    ranked = sorted(
        (haversine_metres(position, point), stop_id) for stop_id, point in candidates.items()
    )
    return tuple(hit for hit in ranked if hit[0] <= tolerance_metres)

"""Combine dated ordered patterns and explicitly transferred published schedules."""

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from .sequence import CycleTemplate


@dataclass(frozen=True)
class RouteEvidence:
    template: CycleTemplate
    visits: tuple[dict[str, Any], ...]
    departures: tuple[tuple[int, ...], ...]
    schedule_kinds: tuple[str, ...]
    source_urls: tuple[str, ...]
    warnings: tuple[str, ...]
    applicable_pattern: bool


def metres(a: dict[str, Any], b: dict[str, Any]) -> float:
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    dlat, dlon = lat2 - lat1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(min(1, h)))


def monotone_stop_match(
    visits: list[dict[str, Any]], stops: list[dict[str, Any]], tolerance: float = 150
) -> tuple[dict[int, int], float]:
    """One-to-one, order-preserving geographic join; opposite platforms cannot flip order."""
    n, m = len(visits), len(stops)
    costs = np.full((n + 1, m + 1), np.inf)
    costs[:, 0] = np.arange(n + 1) * tolerance
    costs[0, :] = np.arange(m + 1) * (tolerance / 3)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance = metres(visits[i - 1], stops[j - 1])
            options = [
                costs[i - 1, j] + tolerance,
                costs[i, j - 1] + tolerance / 3,
                costs[i - 1, j - 1] + distance if distance <= tolerance else math.inf,
            ]
            action = int(np.argmin(options))
            costs[i, j] = options[action]
            back[i, j] = action
    matched: dict[int, int] = {}
    i, j = n, m
    while i and j:
        action = back[i, j]
        if action == 2:
            matched[i - 1] = j - 1
            i -= 1
            j -= 1
        elif action == 0:
            i -= 1
        else:
            j -= 1
    return matched, float(costs[n, m])


def _shorten(
    pattern: dict[str, Any], endpoints: tuple[dict[str, Any], dict[str, Any]]
) -> dict[str, Any]:
    visits = pattern["ordered_visits"]
    a = min(range(len(visits)), key=lambda i: metres(visits[i], endpoints[0]))
    b = min(range(len(visits)), key=lambda i: metres(visits[i], endpoints[1]))
    lo, hi = sorted((a, b))
    if hi - lo < 2 or min(metres(visits[a], endpoints[0]), metres(visits[b], endpoints[0])) > 300:
        raise ValueError("short-route endpoint has no compatible ordered pattern")
    if min(metres(visits[a], endpoints[1]), metres(visits[b], endpoints[1])) > 300:
        raise ValueError("short-route endpoint has no compatible ordered pattern")
    selected = [dict(v) for v in visits[lo : hi + 1]]
    selected[0]["boardable"] = True
    selected[-1]["boardable"] = False
    return {
        **pattern,
        "ordered_visits": selected,
        "edge_lengths_m": pattern["edge_lengths_m"][lo:hi],
        "distance_source": pattern["distance_source"][lo:hi],
        "variant": "official_2025_summer_short_route",
    }


def known_service_warnings(route: str, day: date) -> list[str]:
    warnings = []
    if route == "50" and date(2025, 7, 10) <= day < date(2025, 8, 11):
        warnings.append("known_13_plus_50_diversion_unmodeled")
    if route == "17" and day.weekday() >= 5 and day >= date(2025, 4, 5):
        warnings.append("weekend_medvedkovo_closure_end_unknown")
    if route == "12" and day.weekday() >= 5 and day >= date(2025, 6, 7):
        warnings.append("weekend_entuziastov_diversion_end_unknown")
    if route == "7" and day.weekday() >= 5 and day >= date(2025, 8, 16):
        warnings.append("weekend_sokolniki_short_route_end_unknown")
    return warnings


def compose_route(
    patterns: list[dict[str, Any]], timetables: dict[str, Any], route: str, day: date
) -> RouteEvidence:
    pair = sorted((p for p in patterns if p["route"] == route), key=lambda p: p["relation_id"])
    if len(pair) != 2:
        raise ValueError("exactly two directed patterns required")
    sources = [
        s for s in timetables["sources"] if s["stops"] and s["stops"][0]["route_ref"] == route
    ]
    historical = [s for s in sources if s["status"] == "historical_snapshot"]
    warnings = known_service_warnings(route, day)
    if route == "7" and date(2025, 7, 10) <= day < date(2025, 8, 11):
        if not historical:
            raise ValueError("historical short-route sequence required")
        endpoints = historical[0]["stops"][0], historical[0]["stops"][-1]
        pair = [_shorten(p, endpoints) for p in pair]
        warnings.append("short_turn_geometry_from_ordered_base_and_official_endpoints")
    elif route == "7":
        sources = [s for s in sources if s["status"] != "historical_snapshot"]
    visits = []
    lengths = []
    terminals = []
    pattern_ids = []
    directions = []
    departures = []
    kinds = []
    urls = set()
    for direction, pattern in enumerate(pair):
        ordered = pattern["ordered_visits"]
        candidates = []
        for source in sources:
            schedule_day = date.fromisoformat(source["schedule_date"])
            if schedule_day != day and (schedule_day.weekday() >= 5) != (day.weekday() >= 5):
                continue
            match, cost = monotone_stop_match(ordered, source["stops"])
            if len(match) < max(3, len(ordered) // 2):
                continue
            exact = source["status"] == "historical_snapshot" and schedule_day == day
            rank = (not exact, -len(match), cost, source["source_url"])
            candidates.append((rank, source, match))
        chosen = min(candidates, key=lambda c: c[0]) if candidates else None
        distance_sources = pattern["distance_source"]
        for i, visit in enumerate(ordered):
            visits.append(
                {
                    **visit,
                    "pattern_id": f"osm:{pattern['relation_id']}:v{pattern['version']}",
                    "direction": str(direction),
                    "pattern_sequence": i,
                    "period_basis": pattern["period_basis"],
                    "distance_source": distance_sources[i]
                    if i < len(distance_sources)
                    else "inferred_terminal_connector",
                }
            )
            pattern_ids.append(visits[-1]["pattern_id"])
            directions.append(str(direction))
            terminals.append(i == len(ordered) - 1)
            lengths.append(pattern["edge_lengths_m"][i] if i < len(ordered) - 1 else 0.0)
            if chosen and i in chosen[2]:
                source = chosen[1]
                stop = source["stops"][chosen[2][i]]
                departures.append(tuple(stop["departures_seconds"]))
                exact = source["status"] == "historical_snapshot" and source[
                    "schedule_date"
                ] == str(day)
                kinds.append("historical_exact_day" if exact else source["status"] + "_transfer")
                urls.add(source["source_url"])
            else:
                departures.append(())
                kinds.append("absent")
        urls.add(pattern["source_url"])
    for i in range(2):
        end = pair[i]["ordered_visits"][-1]
        start = pair[(i + 1) % 2]["ordered_visits"][0]
        if metres(end, start) > 750:
            raise ValueError("opposite pattern terminals are not spatially compatible")
    if any(v["boardable"] and not d for v, d in zip(visits, departures, strict=True)):
        warnings.append("incomplete_timetable_coverage_clock_evidence_disabled")
    blocked = any("unmodeled" in w or "end_unknown" in w for w in warnings)
    template = CycleTemplate(
        route,
        tuple(pattern_ids),
        tuple(str(v["stop_id"]) for v in visits),
        tuple(directions),
        tuple(lengths),
        tuple(terminals),
        "dated_osm_membership;current_rail_geometry;inferred_terminals",
    )
    return RouteEvidence(
        template,
        tuple(visits),
        tuple(departures),
        tuple(kinds),
        tuple(sorted(urls)),
        tuple(warnings),
        not blocked,
    )


def timetable_emissions(
    times: NDArray[np.float64],
    evidence: RouteEvidence,
    *,
    use_schedule: bool = True,
    payment_delay: float = 15,
) -> NDArray[np.float64]:
    """Weak clock evidence, with symmetric tails for delayed/early service; never AVL."""
    result = np.zeros((len(times), len(evidence.visits)), dtype=np.float64)
    use_schedule = use_schedule and all(
        not v["boardable"] or bool(d)
        for v, d in zip(evidence.visits, evidence.departures, strict=True)
    )
    clocks = (times + 3 * 3600 - payment_delay) % 86400
    for state, (visit, departures, kind) in enumerate(
        zip(evidence.visits, evidence.departures, evidence.schedule_kinds, strict=True)
    ):
        if not visit["boardable"]:
            result[:, state] = -np.inf
        elif use_schedule and departures:
            stamps = np.array(
                sorted({d + shift for d in departures for shift in (-86400, 0, 86400)})
            )
            insertion = np.searchsorted(stamps, clocks)
            insertion = np.clip(insertion, 1, len(stamps) - 1)
            delta = np.minimum(abs(clocks - stamps[insertion]), abs(clocks - stamps[insertion - 1]))
            exact = kind == "historical_exact_day"
            sigma = 90 if exact else 180
            weight = 0.7 if exact else 0.2
            result[:, state] = weight * np.log(0.15 + 0.85 * np.exp(-0.5 * (delta / sigma) ** 2))
    return result


def as_of(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=ZoneInfo("Europe/Moscow")).astimezone(UTC)

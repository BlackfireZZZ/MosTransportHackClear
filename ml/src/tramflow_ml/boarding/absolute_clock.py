"""Offline clock alignment of inferred profiles, never confirmed historical trips.

Input clocks are Europe/Moscow seconds from service-day midnight. Observations must
already be unwrapped across midnight. Costs are uncalibrated seconds, not probabilities.
"""

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from functools import lru_cache
from typing import Any


def _finite(values: Sequence[float], *, increasing: bool = False) -> list[float]:
    result = [float(x) for x in values]
    if any(not math.isfinite(x) or x < 0 for x in result):
        raise ValueError("nonnegative finite service-day seconds required")
    if increasing and any(a >= b for a, b in zip(result, result[1:], strict=False)):
        raise ValueError("strictly increasing extended service-day seconds required")
    return result


def _schedule(values: Sequence[float]) -> list[float]:
    result = _finite(values)
    if any(a > b for a, b in zip(result, result[1:], strict=False)):
        raise ValueError("nondecreasing schedule clocks required")
    return result


def reconstruct_profiles(
    departures: Sequence[Sequence[float]],
    edge_seconds: Sequence[float],
    *,
    tolerance: float = 120.0,
    beam_width: int = 2,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    """Seed terminal departures; retain bounded inferred clock chains per seed.

    Missing stop clocks yield no complete profile unless allow_missing is set;
    that opt-in propagates edge durations and flags estimated stop times.
    No trip IDs, calendar validity, or cross-stop identity exist
    in this input. Raw clocks below 24h may be lifted one day for midnight traversal;
    explicit >=24h times retain their supplied service-day identity.
    """
    edges = _finite(edge_seconds)
    if not departures or len(edges) != len(departures) - 1 or any(x <= 0 for x in edges):
        raise ValueError("one positive edge per consecutive stop required")
    if not math.isfinite(tolerance) or tolerance < 0 or beam_width < 1:
        raise ValueError("nonnegative tolerance and positive beam width required")
    clocks = [sorted(set(_finite(values))) for values in departures]
    if not clocks[0] or (not allow_missing and any(not values for values in clocks)):
        return []
    estimated = [i for i, values in enumerate(clocks) if not values]
    profiles: list[dict[str, Any]] = []
    for seed in clocks[0]:
        beam: list[tuple[float, list[float]]] = [(0.0, [seed])]
        for values, edge in zip(clocks[1:], edges, strict=True):
            candidates = []
            available = sorted(set(values) | {x + 86400 for x in values if x < 86400})
            for cost, chain in beam:
                target = chain[-1] + edge
                if not values:
                    candidates.append((cost, [*chain, target]))
                    continue
                lo, hi = (
                    bisect_left(available, target - tolerance),
                    bisect_right(available, target + tolerance),
                )
                for value in available[lo:hi]:
                    if value > chain[-1] and abs(value - target) <= tolerance:
                        candidates.append((cost + abs(value - target), [*chain, value]))
            beam = sorted(candidates, key=lambda x: (x[0], x[1]))[:beam_width]
        for cost, chain in beam:
            profiles.append(
                {
                    "profile_id": len(profiles),
                    "times": chain,
                    "reconstruction_cost": cost,
                    "inferred": True,
                    "estimated_stops": estimated,
                }
            )
    return profiles


@lru_cache(maxsize=8192)
def _paths(
    times: tuple[float, ...],
    schedule: tuple[float, ...],
    start: int,
    offset: float,
    max_step: int,
    skip_penalty: float,
    residual_weight: float,
) -> list[tuple[float, list[int]]]:
    states = {start: (0.0, [start])}
    for previous_time, current_time in zip(times, times[1:], strict=False):
        next_states: dict[int, tuple[float, list[int]]] = {}
        for previous, (cost, path) in states.items():
            for current in range(previous + 1, min(len(schedule), previous + max_step + 1)):
                local = current_time - previous_time - (schedule[current] - schedule[previous])
                clock = current_time - schedule[current] - offset
                new_cost = cost + abs(local) + residual_weight * abs(clock)
                new_cost += skip_penalty * (current - previous - 1)
                candidate = (new_cost, [*path, current])
                if current not in next_states or candidate < next_states[current]:
                    next_states[current] = candidate
        states = next_states
    return sorted(states.values())


def _result(
    profile: dict[str, Any],
    times: list[float],
    path: list[int],
    offset: float,
    score: float,
) -> dict[str, Any]:
    schedule = [float(profile["times"][index]) for index in path]
    expected = [b - a for a, b in zip(schedule, schedule[1:], strict=False)]
    local = [b - a - e for a, b, e in zip(times, times[1:], expected, strict=False)]
    return {
        "profile_id": profile["profile_id"],
        "score": score,
        "stop_indices": list(path),
        "start_is_terminal": path[0] == 0,
        "initial_offset": offset,
        "schedule_times": schedule,
        "predicted_times": [x + offset for x in schedule],
        "local_residuals": local,
        "clock_residuals": [t - s - offset for t, s in zip(times, schedule, strict=True)],
        "expected_intervals": expected,
        "steps": [b - a for a, b in zip(path, path[1:], strict=False)],
        "observed_times": times,
        "inferred_profile": bool(profile.get("inferred", True)),
    }


def fit_profiles(
    times: Sequence[float],
    profiles: Sequence[dict[str, Any]],
    *,
    clock_weight: float = 1.0,
    start_prior: float = 60.0,
    residual_weight: float = 0.25,
    max_offset: float = 900.0,
    max_step: int = 6,
    skip_penalty: float = 15.0,
    alternatives: int = 5,
    start_mode: str = "any",
) -> dict[str, Any]:
    """Compare starts under a fixed initial offset, local timing and soft terminal prior.

    Equal adjacent schedule clocks are allowed for minute-resolution feeds.
    Offset includes operational deviation and unknown payment delay. It is never
    independently reset at later stops. Setting clock_weight, start_prior and
    residual_weight to zero gives the interval-only baseline within identical
    max_offset bounds. Alternatives retain one best path per profile/start, not all
    possible paths; candidate counts and score gap therefore are not confidence.
    """
    observed = _finite(times, increasing=True)
    if not observed:
        raise ValueError("at least one observation required")
    controls = [clock_weight, start_prior, residual_weight, max_offset, skip_penalty]
    if any(not math.isfinite(x) or x < 0 for x in controls):
        raise ValueError("finite nonnegative weights and bounds required")
    if start_mode not in {"any", "terminal"}:
        raise ValueError("start_mode must be any or terminal")
    if max_step < 1 or alternatives < 0:
        raise ValueError("positive max_step and nonnegative alternative count required")
    winners = []
    count = 0
    ids = set()
    for profile in profiles:
        if profile["profile_id"] in ids:
            raise ValueError("unique profile IDs required")
        ids.add(profile["profile_id"])
        schedule = _schedule(profile["times"])
        for start, scheduled in enumerate(schedule):
            if start_mode == "terminal" and start != 0:
                continue
            offset = observed[0] - scheduled
            if abs(offset) > max_offset:
                continue
            count += 1
            paths = _paths(
                tuple(observed),
                tuple(schedule),
                start,
                offset,
                max_step,
                skip_penalty,
                residual_weight,
            )
            if paths:
                cost, path = paths[0]
                score = cost + clock_weight * abs(offset) + start_prior * (start != 0)
                winners.append(_result(profile, observed, path, offset, score))
    winners.sort(key=lambda x: (x["score"], str(x["profile_id"]), x["stop_indices"]))
    return {
        "best": winners[0] if winners else None,
        "alternatives": winners[1 : alternatives + 1],
        "gap": winners[1]["score"] - winners[0]["score"] if len(winners) > 1 else None,
        "candidate_count": count,
        "feasible_count": len(winners),
        "alternative_unit": "best_path_per_profile_and_start",
    }


def continue_profile(
    times: Sequence[float],
    profile: dict[str, Any],
    prefix: dict[str, Any],
    *,
    max_step: int = 6,
    skip_penalty: float = 15.0,
    residual_weight: float = 0.25,
) -> dict[str, Any] | None:
    """Fit future observations with prefix profile, offset and last stop locked.

    Output includes the last prefix event as its first point. No new start search,
    initial-offset estimation, profile selection or terminal prior is performed.
    """
    observed = _finite([prefix["observed_times"][-1], *times], increasing=True)
    schedule = _schedule(profile["times"])
    if profile["profile_id"] != prefix["profile_id"]:
        raise ValueError("prefix profile must remain locked")
    if max_step < 1 or any(not math.isfinite(x) or x < 0 for x in [skip_penalty, residual_weight]):
        raise ValueError("positive max_step and nonnegative finite weights required")
    start = prefix["stop_indices"][-1]
    if start < 0 or start >= len(schedule):
        raise ValueError("prefix stop outside profile")
    if schedule[start] != prefix["schedule_times"][-1]:
        raise ValueError("prefix schedule must remain locked")
    offset = float(prefix["initial_offset"])
    if not math.isfinite(offset):
        raise ValueError("finite prefix offset required")
    paths = _paths(
        tuple(observed), tuple(schedule), start, offset, max_step, skip_penalty, residual_weight
    )
    if not paths:
        return None
    cost, path = paths[0]
    return _result(profile, observed, path, offset, cost)

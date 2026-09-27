"""Frozen scalar duty-clock calibration; timetable compatibility is not location truth."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

BOUND = 1800
GRID = tuple(range(-BOUND, BOUND + 1, 30))
CAP = 300.0


def causal_onsets(times: list[float], gap: float = 60) -> list[float]:
    """First event after observed silence; first event has no known preceding gap."""
    if not math.isfinite(gap) or gap <= 0 or any(not math.isfinite(t) for t in times):
        raise ValueError("finite clocks and positive finite gap required")
    if any(a > b for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("ordered clocks required")
    return [b for a, b in zip(times, times[1:], strict=False) if b - a >= gap]


def split_profiles(profiles: list[dict[str, Any]], cutoff: float) -> dict[str, Any]:
    """Use whole trips separated at cutoff for every shift within +/-1800 seconds."""
    if not math.isfinite(cutoff):
        raise ValueError("finite cutoff required")
    early, late, crossing = [], [], []
    for p in profiles:
        times = p["times"]
        if (
            not times
            or any(not math.isfinite(t) for t in times)
            or any(a > b for a, b in zip(times, times[1:], strict=False))
        ):
            raise ValueError("ordered nonempty finite trip required")
        if str(p["direction"]) not in ("0", "1"):
            raise ValueError("binary direction required")
        if times[-1] + BOUND < cutoff:
            early.append(p)
        elif times[0] - BOUND >= cutoff:
            late.append(p)
        else:
            crossing.append(p)
    return {"early": early, "late": late, "crossing": len(crossing)}


def clocks(profiles: list[dict[str, Any]]) -> list[float]:
    return sorted({float(t) for p in profiles for t in p["times"]})


def nearest_errors(
    observed: list[float], schedule: list[float], shift: float = 0
) -> NDArray[np.float64]:
    """Return absolute nearest-clock errors; missing schedule is infinite error."""
    if not math.isfinite(shift) or any(not math.isfinite(t) for t in [*observed, *schedule]):
        raise ValueError("finite times and shift required")
    if any(a > b for a, b in zip(schedule, schedule[1:], strict=False)):
        raise ValueError("sorted schedule required")
    x = np.asarray(observed, dtype=float)
    if not schedule:
        return np.full(len(x), np.inf)
    s = np.asarray(schedule, dtype=float) + shift
    i = np.searchsorted(s, x)
    return np.minimum(
        abs(x - s[np.clip(i, 0, len(s) - 1)]), abs(x - s[np.clip(i - 1, 0, len(s) - 1)])
    )


def measure(observed: list[float], schedule: list[float], shift: float) -> dict[str, Any]:
    errors = nearest_errors(observed, schedule, shift)
    return {
        "events": len(observed),
        "capped_mae": float(np.minimum(errors, CAP).mean()) if len(errors) else None,
        "within60": int((errors <= 60).sum()),
        "within300": int((errors <= CAP).sum()),
        "schedule_clocks": len(schedule),
    }


def calibrate(early: list[float], schedule: list[float]) -> dict[str, Any]:
    """Fit one scalar from early events only; retain all shifts within 5 seconds MAE."""
    if not early or not schedule:
        return {"shift": None, "loss": None, "near_shifts": [], "at_bound": False}
    losses = [float(np.minimum(nearest_errors(early, schedule, s), CAP).mean()) for s in GRID]
    i = min(range(len(GRID)), key=lambda j: (losses[j], abs(GRID[j]), GRID[j]))
    return {
        "shift": GRID[i],
        "loss": losses[i],
        "near_shifts": [s for s, v in zip(GRID, losses, strict=True) if v <= losses[i] + 5],
        "at_bound": abs(GRID[i]) == BOUND,
    }


def directions(observed: list[float], profiles: list[dict[str, Any]], shift: float) -> list[int]:
    """Resolve only unique active trip and nearest clock within 60 seconds.

    Overlapping trips, nearest-clock ties across directions and uncovered clocks
    stay unknown even if one lexical trip could have supplied a label.
    """
    per_direction = [
        nearest_errors(observed, clocks([p for p in profiles if str(p["direction"]) == d]), shift)
        for d in ("0", "1")
    ]
    active = np.zeros(len(observed), dtype=int)
    trip_direction = np.full(len(observed), -1, dtype=int)
    x = np.asarray(observed, dtype=float)
    for p in profiles:
        mask = (x >= p["times"][0] + shift) & (x <= p["times"][-1] + shift)
        active += mask
        trip_direction[mask] = int(p["direction"])
    selected = np.where(per_direction[0] < per_direction[1], 0, 1)
    distance = np.minimum(per_direction[0], per_direction[1])
    unique = (
        (active == 1)
        & (distance <= 60)
        & ~np.isclose(per_direction[0], per_direction[1], rtol=0, atol=1e-9)
        & (selected == trip_direction)
    )
    return np.where(unique, selected, -1).tolist()


def direction_summary(
    observed: list[float],
    profiles: list[dict[str, Any]],
    fit: dict[str, Any],
) -> dict[str, Any]:
    if fit["shift"] is None:
        return {"assigned": 0, "stable": 0, "unknown": len(observed), "changed_from_zero": 0}
    chosen = np.asarray(directions(observed, profiles, fit["shift"]))
    zero = np.asarray(directions(observed, profiles, 0))
    stable = chosen >= 0
    for shift in fit["near_shifts"]:
        stable &= np.asarray(directions(observed, profiles, shift)) == chosen
    return {
        "assigned": int((chosen >= 0).sum()),
        "stable": int(stable.sum()),
        "unknown": int((chosen < 0).sum()),
        "direction0": int((chosen == 0).sum()),
        "direction1": int((chosen == 1).sum()),
        "changed_from_zero": int(((zero >= 0) & (chosen >= 0) & (zero != chosen)).sum()),
        "comparable_to_zero": int(((zero >= 0) & (chosen >= 0)).sum()),
    }


def guarded_calibration(
    early: list[float],
    schedule: list[float],
    midpoint: float,
) -> dict[str, Any]:
    """Apply the full early fit only after disjoint early-half consistency checks.

    The second half validates the first-half offset; no later-day observations
    enter this gate. A rejection retains zero, never a fabricated missing loss.
    """
    if not math.isfinite(midpoint):
        raise ValueError("finite early midpoint required")
    first = [t for t in early if t < midpoint]
    second = [t for t in early if t >= midpoint]
    full = calibrate(early, schedule)
    a, b = calibrate(first, schedule), calibrate(second, schedule)
    gain = (
        measure(second, schedule, 0)["capped_mae"]
        - measure(second, schedule, a["shift"])["capped_mae"]
        if second and a["shift"] is not None
        else None
    )
    if len(first) < 10 or len(second) < 10 or a["shift"] is None or b["shift"] is None:
        reason = "insufficient_early_halves"
    elif any(max(x["near_shifts"]) - min(x["near_shifts"]) > 120 for x in (a, b)):
        reason = "ambiguous_early_shift"
    elif abs(a["shift"] - b["shift"]) > 60 or abs(full["shift"] - a["shift"]) > 60:
        reason = "unstable_early_shift"
    elif gain is None or gain < 5:
        reason = "no_early_validation_gain"
    else:
        reason = "accepted"
    accepted = reason == "accepted"
    return {
        "shift": full["shift"] if accepted else 0,
        "accepted": accepted,
        "reason": reason,
        "early_validation_gain": gain,
        "first_fit": a,
        "second_fit": b,
        "full_fit": full,
        "near_shifts": full["near_shifts"] if accepted else [0],
        "first_events": len(first),
        "second_events": len(second),
    }

"""Bilateral offline evidence; inferred GTFS duty codes are never observed trip IDs."""

import math
from typing import Any

import numpy as np

from .absolute_clock import continue_profile, fit_profiles


def duty_component(trip_id: str) -> str | None:
    """Return this feed's candidate duty suffix, without certifying its semantics."""
    parts = trip_id.split("_")
    if len(parts) != 4 or any(not p.isdigit() for p in parts):
        return None
    return parts[3]


def shuffle_gaps(times: list[float], seed: int) -> list[float]:
    """Preserve first clock, count, total duration and interval multiset."""
    values = np.asarray(times, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("at least two strictly increasing finite clocks required")
    gaps = np.random.default_rng(seed).permutation(np.diff(values))
    return [float(values[0]), *map(float, values[0] + np.cumsum(gaps))]


def compare_directions(
    times: list[float],
    profiles: list[dict[str, Any]],
    *,
    prefix_size: int = 4,
    start_prior: float = 0,
    failure_cost: float = 900,
) -> dict[str, Any]:
    """Select by prefix only; keep opposite-direction continuation and every refusal.

    Continuation still aligns observed suffix onsets to later stops; its residual
    is a compatibility check, not a next-arrival prediction or geographic accuracy.
    The capped loss assigns failure_cost to each unavailable suffix interval.
    """
    if not 2 <= prefix_size < len(times):
        raise ValueError("prefix and nonempty suffix required")
    if not math.isfinite(failure_cost) or failure_cost <= 0:
        raise ValueError("positive finite failure cost required")
    directions = sorted({str(p["direction"]) for p in profiles})
    if set(directions) - {"0", "1"}:
        raise ValueError("binary reference directions required")
    fits: dict[str, Any] = {}
    for direction in ("0", "1"):
        candidates = [p for p in profiles if str(p["direction"]) == direction]
        fit = fit_profiles(
            times[:prefix_size], candidates, clock_weight=0.2, start_prior=start_prior,
            residual_weight=0.25, alternatives=0,
        )
        best = fit["best"]
        suffix = None
        selected = None
        if best is not None:
            selected = next(p for p in candidates if p["profile_id"] == best["profile_id"])
            suffix = continue_profile(times[prefix_size:], selected, best, residual_weight=0.25)
        errors = [abs(v) for v in suffix["local_residuals"]] if suffix else []
        path = best["stop_indices"] + suffix["stop_indices"][1:] if best and suffix else []
        fits[direction] = {
            "profiles": len(candidates), "candidate_starts": fit["candidate_count"],
            "prefix_score": best["score"] if best else None,
            "prefix": best, "continuation": suffix,
            "suffix_errors": errors,
            "suffix_mae": float(np.mean(errors)) if errors else None,
            "capped_loss": float(np.mean(np.minimum(errors, failure_cost))) if errors
            else failure_cost,
            "stop_names": [selected["stop_names"][i] for i in path] if selected else [],
            "stop_ids": [selected["stop_ids"][i] for i in path] if selected else [],
            "destination": selected["stop_names"][-1] if selected else None,
        }
    scores = {d: f["prefix_score"] for d, f in fits.items() if f["prefix_score"] is not None}
    winner = min(scores, key=lambda d: (scores[d], d)) if scores else None
    margin = abs(scores["0"] - scores["1"]) if len(scores) == 2 else None
    if margin is not None and math.isclose(margin, 0, abs_tol=1e-9):
        winner = None
    opposite = str(1 - int(winner)) if winner is not None else None
    advantage = None
    if opposite is not None and winner is not None:
        advantage = fits[opposite]["capped_loss"] - fits[winner]["capped_loss"]
    return {
        "winner": winner, "margin": margin, "fits": fits,
        "prefix_bilateral": len(scores) == 2,
        "suffix_bilateral": all(f["continuation"] is not None for f in fits.values()),
        "chosen_suffix_advantage": advantage,
        "direction_is_observed": False, "failure_cost": failure_cost,
    }

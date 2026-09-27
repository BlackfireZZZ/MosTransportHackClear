"""Clock-independent paired matching conditional on an independently selected direction."""

import hashlib
import json
import math
from typing import Any

import numpy as np

from .absolute_clock import continue_profile, fit_profiles


def scheduled_direction(first: float, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Use first absolute onset only; inclusive endpoint ties remain ambiguous."""
    candidates = [p for p in profiles if p["times"][0] <= first <= p["times"][-1]]
    selected = candidates[0] if len(candidates) == 1 else None
    return {
        "direction": str(selected["direction"]) if selected else None,
        "candidate_count": len(candidates),
        "profile_id": selected["profile_id"] if selected else None,
        "scheduled_end": selected["times"][-1] if selected else None,
    }


def relative_profiles(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate equal intervals and stop visits, preserving all source profile IDs."""
    unique: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        clocks = [float(t) for t in profile["times"]]
        if not clocks or any(not math.isfinite(t) for t in clocks):
            raise ValueError("finite nonempty schedule required")
        relative = [t - clocks[0] for t in clocks]
        if any(a > b for a, b in zip(relative, relative[1:], strict=False)):
            raise ValueError("ordered schedule required")
        if len(profile["stop_ids"]) != len(relative):
            raise ValueError("one stop visit per schedule clock required")
        key = json.dumps([str(profile["direction"]), relative, profile["stop_ids"]])
        if key not in unique:
            shape = json.dumps([relative, profile["stop_ids"]])
            unique[key] = {
                "profile_id": (
                    str(profile["direction"]) + ":" + hashlib.sha256(shape.encode()).hexdigest()
                ),
                "direction": str(profile["direction"]), "times": relative,
                "stop_ids": list(profile["stop_ids"]), "source_profile_ids": [],
            }
        unique[key]["source_profile_ids"].append(profile["profile_id"])
    for p in unique.values():
        p["source_profile_ids"] = sorted(p["source_profile_ids"])
    return sorted(unique.values(), key=lambda p: p["profile_id"])


def _measure(result: dict[str, Any] | None, failure_cost: float) -> dict[str, Any]:
    errors = [abs(v) for v in result["local_residuals"]] if result else []
    return {
        "feasible": result is not None,
        "mae": float(np.mean(errors)) if errors else None,
        "capped_loss": float(np.mean(np.minimum(errors, failure_cost))) if errors else failure_cost,
        "cost_per_interval": result["score"] / len(errors) if errors and result else None,
        "skipped_stops": sum(step - 1 for step in result["steps"]) if result else None,
        "errors": errors,
    }


def compare_relative(
    times: list[float],
    profiles: list[dict[str, Any]],
    *,
    prefix_size: int = 4,
    max_step: int = 6,
    failure_cost: float = 900,
) -> dict[str, Any]:
    """Apply identical interval-only search to both directions, independent of clock.

    Suffix alignment sees suffix observations; errors measure compatibility, not
    predicted arrival times. An unavailable full continuation costs failure_cost.
    """
    if not 2 <= prefix_size < len(times) or max_step < 1:
        raise ValueError("nonempty prefix/suffix and positive max_step required")
    if not math.isfinite(failure_cost) or failure_cost <= 0:
        raise ValueError("positive finite failure cost required")
    if any(not math.isfinite(t) for t in times) or any(
        a >= b for a, b in zip(times, times[1:], strict=False)
    ):
        raise ValueError("finite strictly increasing observations required")
    observed = [float(t - times[0]) for t in times]
    relative = relative_profiles(profiles)
    if {p["direction"] for p in relative} - {"0", "1"}:
        raise ValueError("binary direction required")
    offset_bound = max([1.0, *(p["times"][-1] for p in relative)])
    output = {}
    for direction in ("0", "1"):
        candidates = [p for p in relative if p["direction"] == direction]
        controls = dict(clock_weight=0, start_prior=0, residual_weight=0,
                        max_offset=offset_bound, max_step=max_step, skip_penalty=15)
        prefix = fit_profiles(observed[:prefix_size], candidates, alternatives=1, **controls)
        full = fit_profiles(observed, candidates, alternatives=0, **controls)
        best = prefix["best"]
        continuation = None
        if best is not None:
            profile = next(p for p in candidates if p["profile_id"] == best["profile_id"])
            continuation = continue_profile(observed[prefix_size:], profile, best,
                                            max_step=max_step, skip_penalty=15, residual_weight=0)
        output[direction] = {
            "unique_profiles": len(candidates), "candidate_starts": prefix["candidate_count"],
            "prefix": best,
            "prefix_tied": (
                prefix["gap"] is not None and math.isclose(prefix["gap"], 0, abs_tol=1e-9)
            ),
            "full": _measure(full["best"], failure_cost),
            "suffix": _measure(continuation, failure_cost),
        }
    return output

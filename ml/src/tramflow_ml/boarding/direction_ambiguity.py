"""Exact prefix endpoint ambiguity, without interpreting costs as probabilities."""

import math
from typing import Any

from .absolute_clock import continue_profile
from .direction_comparison import relative_profiles


def prefix_states(
    times: list[float],
    profiles: list[dict[str, Any]],
    *,
    max_step: int = 6,
) -> dict[str, Any]:
    """Retain every globally optimal (relative profile, endpoint) per direction.

    With zero clock/residual costs, the endpoint is sufficient for continuation;
    start and intermediate tied paths do not change its future cost. Uniform
    weighting is over distinct states, not over the number of ways to reach them.
    """
    if len(times) < 2 or max_step < 1 or any(not math.isfinite(t) for t in times):
        raise ValueError("finite prefix of at least two events and positive max_step required")
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    if any(g <= 0 for g in gaps):
        raise ValueError("strictly increasing prefix required")
    relative = relative_profiles(profiles)
    if {p["direction"] for p in relative} - {"0", "1"}:
        raise ValueError("binary direction required")
    output: dict[str, Any] = {}
    for direction in ("0", "1"):
        candidates: list[dict[str, Any]] = []
        for p in relative:
            if p["direction"] != direction:
                continue
            schedule = p["times"]
            costs = {i: 0.0 for i in range(len(schedule))}
            for gap in gaps:
                following: dict[int, float] = {}
                for previous, cost in costs.items():
                    for current in range(previous + 1, min(len(schedule), previous + max_step + 1)):
                        value = cost + abs(gap - (schedule[current] - schedule[previous]))
                        value += 15 * (current - previous - 1)
                        following[current] = min(following.get(current, math.inf), value)
                costs = following
            candidates.extend(
                {"profile_id": p["profile_id"], "endpoint": i, "cost": cost}
                for i, cost in costs.items()
            )
        minimum = min((s["cost"] for s in candidates), default=None)
        states = [
            s
            for s in candidates
            if minimum is not None and math.isclose(s["cost"], minimum, rel_tol=0, abs_tol=1e-9)
        ]
        output[direction] = {"cost": minimum, "states": states}
    return output


def compare_ambiguity(
    times: list[float],
    profiles: list[dict[str, Any]],
    *,
    prefix_size: int = 4,
    max_step: int = 6,
    failure_cost: float = 900,
) -> dict[str, Any]:
    """Freeze prefix states; average capped suffix compatibility over all states.

    No suffix state selection or renormalization after failure is allowed.
    The min/max envelope is sensitivity, never an accuracy/confidence interval.
    """
    if not 2 <= prefix_size < len(times):
        raise ValueError("prefix and nonempty suffix required")
    if not math.isfinite(failure_cost) or failure_cost <= 0:
        raise ValueError("positive finite failure cost required")
    if any(not math.isfinite(t) for t in times) or any(
        a >= b for a, b in zip(times, times[1:], strict=False)
    ):
        raise ValueError("finite strictly increasing observations required")
    observed = [float(t - times[0]) for t in times]
    selected = prefix_states(observed[:prefix_size], profiles, max_step=max_step)
    relative = {p["profile_id"]: p for p in relative_profiles(profiles)}
    output = {}
    for direction in ("0", "1"):
        rows = []
        for state in selected[direction]["states"]:
            p = relative[state["profile_id"]]
            endpoint = state["endpoint"]
            prefix = {
                "profile_id": p["profile_id"],
                "stop_indices": [endpoint],
                "schedule_times": [p["times"][endpoint]],
                "observed_times": [observed[prefix_size - 1]],
                "initial_offset": observed[prefix_size - 1] - p["times"][endpoint],
            }
            suffix = continue_profile(
                observed[prefix_size:],
                p,
                prefix,
                max_step=max_step,
                skip_penalty=15,
                residual_weight=0,
            )
            errors = [abs(v) for v in suffix["local_residuals"]] if suffix else []
            loss = (
                sum(min(v, failure_cost) for v in errors) / len(errors) if errors else failure_cost
            )
            rows.append(
                {
                    **state,
                    "feasible": suffix is not None,
                    "capped_loss": loss,
                    "mae": sum(errors) / len(errors) if errors else None,
                }
            )
        losses = [r["capped_loss"] for r in rows] or [failure_cost]
        output[direction] = {
            "prefix_cost": selected[direction]["cost"],
            "state_count": len(rows),
            "states": rows,
            "feasible_states": sum(r["feasible"] for r in rows),
            "all_feasible": bool(rows) and all(r["feasible"] for r in rows),
            "any_feasible": any(r["feasible"] for r in rows),
            "mean_capped_loss": sum(losses) / len(losses),
            "min_capped_loss": min(losses),
            "max_capped_loss": max(losses),
        }
    return output

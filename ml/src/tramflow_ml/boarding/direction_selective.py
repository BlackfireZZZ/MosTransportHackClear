"""Prefix-only direction selection with explicit abstention and uncalibrated margins."""

import math
from typing import Any

from .direction_ambiguity import prefix_states


def select_direction(
    prefix_times: list[float], profiles: list[dict[str, Any]], *,
    minimum_margin_per_interval: float = 15,
) -> dict[str, Any]:
    """Require bilateral finite costs and the same sufficiently separated winner.

    Only supplied prefix events enter selection. Margins are seconds of alignment
    cost per interval, not calibrated probabilities or geographic accuracy.
    """
    if not math.isfinite(minimum_margin_per_interval) or minimum_margin_per_interval < 0:
        raise ValueError("finite nonnegative minimum margin required")
    per_step: dict[str, Any] = {}
    for step in (3, 6):
        states = prefix_states(prefix_times, profiles, max_step=step)
        costs = {direction: states[direction]['cost'] for direction in ('0', '1')}
        bilateral = all(cost is not None and math.isfinite(cost) for cost in costs.values())
        winner = None
        margin = None
        if bilateral:
            difference = costs['1'] - costs['0']
            margin = abs(difference) / (len(prefix_times) - 1)
            if not math.isclose(difference, 0, rel_tol=0, abs_tol=1e-9):
                winner = '0' if difference > 0 else '1'
        per_step[str(step)] = {
            'costs': costs, 'bilateral_finite': bilateral,
            'direction': winner, 'margin_per_interval': margin,
        }
    checks = list(per_step.values())
    direction = None
    if any(not check['bilateral_finite'] for check in checks):
        reason = 'missing_or_nonfinite_direction_cost'
    elif any(check['direction'] is None for check in checks):
        reason = 'tied_costs'
    elif checks[0]['direction'] != checks[1]['direction']:
        reason = 'step_disagreement'
    elif any(check['margin_per_interval'] < minimum_margin_per_interval for check in checks):
        reason = 'insufficient_margin'
    else:
        direction = checks[0]['direction']
        reason = 'selected'
    return {
        'direction': direction, 'reason': reason, 'per_step': per_step,
        'minimum_margin_per_interval': minimum_margin_per_interval,
        'calibrated_probability': False, 'direction_is_observed': False,
    }

"""Conservative same-wave candidate coalescing, without a fixed stop-spacing ban."""

from typing import Any

import numpy as np

from .multiscale import SCALES, peaks, representation


def adaptive_supports(times: Any, devices: Any, threshold: float) -> list[list[float]]:
    r = representation(times, devices)
    best = np.argmax(r["scan"], axis=1)
    widths = SCALES[best]
    score = r["scan"][np.arange(len(best)), best]
    chosen = peaks(r["times"], score, widths, threshold)
    return [[float(r["times"][i]), float(r["times"][i] + widths[i])] for i in chosen]


def coalesce(
    times: Any,
    candidates: Any,
    supports: list[list[float]],
    *,
    quiet_score: float = 0.01,
    first_observation_anchor: bool = False,
) -> dict[str, Any]:
    """Only coalesce within one adaptive support; an unusual quiet gap vetoes merging.

    Full event gaps overlapping the candidate interval provide quiet evidence.
    Quiet scores use occupied-second rate and are heuristic, not calibrated p-values.
    Raw members remain auditable; no stop identity or door telemetry is inferred.
    An optional mandatory first-payment anchor is an observation-boundary hypothesis,
    not evidence of a trip start or route stop ordinal. It obeys the same merge veto.
    """
    t, c = np.asarray(times, dtype=float), np.asarray(candidates, dtype=float)
    if (
        not len(t)
        or not np.isfinite(t).all()
        or not np.isfinite(c).all()
        or np.any(np.diff(t) < 0)
        or np.any(np.diff(c) <= 0)
        or np.any(c < t[0])
        or np.any(c > t[-1])
        or not 0 < quiet_score < 1
    ):
        raise ValueError("finite sorted events and strictly ordered in-range candidates required")
    previous_end = -np.inf
    for start, end in supports:
        if not np.isfinite([start, end]).all() or start >= end or start < previous_end:
            raise ValueError("ordered nonoverlapping positive supports required")
        previous_end = end
    input_candidates = len(c)
    anchor_added = first_observation_anchor and (not len(c) or c[0] != t[0])
    if anchor_added:
        c = np.insert(c, 0, t[0])
    members: list[dict[str, Any]] = []
    decisions = []
    for onset in c:
        support = next((i for i, (a, b) in enumerate(supports) if a <= onset < b), None)
        merge = False
        quiet = None
        if members and support is not None and support == members[-1]["support"]:
            start, end = supports[support]
            events = t[(t >= start) & (t < end)]
            distinct = np.unique(events)
            gaps = np.diff(distinct)
            between = gaps[(distinct[:-1] < onset) & (distinct[1:] > members[-1]["members"][-1])]
            rate = len(events) / max(1, len(np.unique(np.floor(events))))
            quiet = float(min(1, max(1, len(gaps)) * np.exp(-rate * max(between, default=0))))
            merge = quiet >= quiet_score
            decisions.append(
                {
                    "left": members[-1]["members"][-1],
                    "right": float(onset),
                    "support": support,
                    "quiet_score": quiet,
                    "decision": "merge_same_support" if merge else "keep_quiet_gap",
                }
            )
        if merge:
            members[-1]["members"].append(float(onset))
        else:
            members.append({"time": float(onset), "members": [float(onset)], "support": support})
    return {
        "times": [v["time"] for v in members],
        "groups": members,
        "decisions": decisions,
        "input_candidates": input_candidates,
        "anchor": {
            "enabled": first_observation_anchor,
            "added": bool(anchor_added),
            "time": float(t[0]) if first_observation_anchor else None,
            "basis": "first_observed_payment" if first_observation_anchor else None,
            "trip_start_confirmed": False,
            "route_stop_index": None,
        },
        "merged_candidates": len(c) - len(members),
        "quiet_threshold": quiet_score,
    }

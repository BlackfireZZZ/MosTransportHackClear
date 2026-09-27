"""Conservative edge durations from unpaired clocks, never reconstructed trip IDs.

Acceptance establishes internal clock consistency, not historical service validity.
Thresholds are declared heuristics; clocks transferred from another date stay proxy
evidence even when accepted. GTFS stop_times with trip_id would permit direct joins;
the current snapshots have no equivalent key.
"""

import math
from collections import Counter
from typing import Any

import numpy as np

from .composition import RouteEvidence

DAY = 86400
TOLERANCE = 30.0
MIN_MATCHES = 8
MIN_COVERAGE = 0.65
ALIAS_RATIO = 0.9
MAX_DEPARTURES = 2048
SUPPORTED_KINDS = {"historical_exact_day", "historical_snapshot_transfer", "current_proxy_transfer"}


def _clocks(values: tuple[int, ...]) -> list[float]:
    if len(values) > MAX_DEPARTURES:
        raise ValueError("departure count exceeds bounded clock alignment budget")
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or not 0 <= v < DAY
        for v in values
    ):
        raise ValueError("departure clocks must be finite seconds in [0, 86400)")
    if len(set(values)) != len(values):
        raise ValueError("duplicate departure clocks lack distinct trip identities")
    return sorted(float(v) for v in values)


def _pairs(a: list[float], b: list[float], lag: float) -> list[tuple[float, float]]:
    extended = sorted((clock + shift, i) for shift in (-DAY, 0, DAY) for i, clock in enumerate(b))
    used: set[int] = set()
    pairs = []
    cursor = 0
    for clock in a:
        target = clock + lag
        while cursor < len(extended) and extended[cursor][0] < target - TOLERANCE:
            cursor += 1
        candidate = cursor
        choices = []
        while candidate < len(extended) and extended[candidate][0] <= target + TOLERANCE:
            value, index = extended[candidate]
            if index not in used:
                choices.append((abs(value - target), candidate))
            candidate += 1
        if choices:
            _, selected = min(choices)
            value, index = extended[selected]
            used.add(index)
            pairs.append((clock, value - clock))
            cursor = selected + 1
    return pairs


def _estimate(a: list[float], b: list[float], length: float) -> dict[str, Any]:
    lower = max(15.0, length / 18.0)
    upper = min(1800.0, max(180.0, length / 1.2 + 120.0))
    result: dict[str, Any] = {
        "physical_bounds_seconds": [lower, upper],
        "source_departures": len(a),
        "target_departures": len(b),
        "matched_departures": 0,
        "coverage": 0.0,
        "candidate_aliases": [],
    }
    if min(len(a), len(b)) < MIN_MATCHES:
        return {**result, "reason": "insufficient_departures"}
    candidates = sorted({lower, upper, *np.arange(math.ceil(lower / 15) * 15, upper, 15)})
    scores = []
    for candidate in candidates:
        if not lower <= candidate <= upper:
            continue
        pairs = [p for p in _pairs(a, b, candidate) if lower <= p[1] <= upper]
        if not pairs:
            continue
        median = float(np.median([p[1] for p in pairs]))
        mad = float(np.median([abs(p[1] - median) for p in pairs]))
        scores.append((len(pairs), -mad, median, pairs))
    scores.sort(key=lambda s: (-s[0], -s[1], s[2]))
    peaks: list[tuple[int, float, float, list[tuple[float, float]]]] = []
    for score in scores:
        if all(abs(score[2] - previous[2]) > 2 * TOLERANCE for previous in peaks):
            peaks.append(score)
    if not peaks:
        return {**result, "reason": "no_physically_feasible_lag"}
    count, negative_mad, median, pairs = peaks[0]
    result.update(
        matched_departures=count,
        coverage=count / max(len(a), len(b)),
        proposed_seconds=median,
        mad_seconds=-negative_mad,
        uncertainty_seconds=max(TOLERANCE, -negative_mad * 1.4826),
        candidate_aliases=[
            {"seconds": s[2], "matched_departures": s[0], "coverage": s[0] / max(len(a), len(b))}
            for s in peaks[:10]
        ],
    )
    if count < MIN_MATCHES or result["coverage"] < MIN_COVERAGE:
        return {**result, "reason": "insufficient_matching_coverage"}
    aliases = [s for s in peaks[1:] if s[0] >= count * ALIAS_RATIO]
    if aliases:
        return {**result, "reason": "periodic_or_competing_lag_alias"}
    gaps = [(a[(i + 1) % len(a)] - v) % DAY for i, v in enumerate(a)]
    start = a[(int(np.argmax(gaps)) + 1) % len(a)]
    ordered = sorted(a, key=lambda v: (v - start) % DAY)
    first = set(ordered[: len(ordered) // 2])
    halves = [[p[1] for p in pairs if (p[0] in first) == is_first] for is_first in (True, False)]
    half_counts = [len(first), len(a) - len(first)]
    medians = [float(np.median(h)) if h else None for h in halves]
    coverage = [len(h) / n for h, n in zip(halves, half_counts, strict=True)]
    result.update(temporal_half_medians_seconds=medians, temporal_half_coverage=coverage)
    if any(len(h) < 3 for h in halves) or min(coverage) < MIN_COVERAGE:
        return {**result, "reason": "inconsistent_temporal_half_coverage"}
    assert medians[0] is not None and medians[1] is not None
    if abs(medians[0] - medians[1]) > TOLERANCE:
        return {**result, "reason": "inconsistent_temporal_half_duration"}
    return {
        **result,
        "reason": "accepted_unpaired_clock_shift",
        "seconds": median,
        "accepted": True,
    }


def estimate_schedule_edges(evidence: RouteEvidence) -> dict[str, Any]:
    """Return one outgoing-edge record per visit, with nullable accepted duration.

    A rejected estimate never supplies seconds. Matching is monotone and one-to-one
    within each lag hypothesis; hypotheses are statistical alignments, not trip IDs.
    Input arrays must have identical lengths; invalid individual clocks reject their
    edge explicitly. Clock values refer to a recurring civil day, with circular
    midnight matching; this is not GTFS service-date reconstruction.
    """
    n = len(evidence.visits)
    arrays = (
        evidence.departures,
        evidence.schedule_kinds,
        evidence.template.edge_lengths_m,
        evidence.template.terminal_after,
        evidence.template.directions,
        evidence.template.pattern_ids,
    )
    if not n or any(len(values) != n for values in arrays):
        raise ValueError("route evidence arrays must have equal nonzero length")
    edges = []
    for state in range(n):
        target = (state + 1) % n
        kind = evidence.schedule_kinds[state]
        edge: dict[str, Any] = {
            "state": state,
            "target_state": target,
            "seconds": None,
            "accepted": False,
            "reason": "unavailable",
            "source_kind": kind,
            "historical_exact_day": kind == "historical_exact_day",
            "trip_identity_verified": False,
        }
        length = evidence.template.edge_lengths_m[state]
        if evidence.template.terminal_after[state]:
            edge["reason"] = "terminal_connector"
        elif not all(evidence.visits[s].get("boardable", False) for s in (state, target)):
            edge["reason"] = "nonboardable_boundary"
        elif (
            evidence.template.directions[state] != evidence.template.directions[target]
            or evidence.template.pattern_ids[state] != evidence.template.pattern_ids[target]
        ):
            edge["reason"] = "pattern_boundary"
        elif kind == "absent" or not all(evidence.departures[s] for s in (state, target)):
            edge["reason"] = "absent_schedule"
        elif kind != evidence.schedule_kinds[target]:
            edge["reason"] = "incompatible_schedule_kinds"
        elif kind not in SUPPORTED_KINDS:
            edge["reason"] = "unsupported_schedule_kind"
        elif not math.isfinite(length) or length <= 0:
            edge["reason"] = "invalid_edge_length"
        else:
            try:
                a, b = (_clocks(evidence.departures[s]) for s in (state, target))
            except ValueError as exc:
                edge.update(reason="invalid_departure_clocks", error=str(exc))
            else:
                edge.update(_estimate(a, b, length))
        edges.append(edge)
    accepted = sum(e["accepted"] for e in edges)
    return {
        "schema_version": "boarding-schedule-intervals.v1",
        "route": evidence.template.route,
        "method": "unpaired_clock_shift_monotone_matching",
        "trip_identity_verified": False,
        "historical_validity_verified": False,
        "thresholds": {
            "clock_tolerance_seconds": TOLERANCE,
            "minimum_matched_departures": MIN_MATCHES,
            "minimum_coverage": MIN_COVERAGE,
            "competing_alias_support_ratio": ALIAS_RATIO,
            "maximum_speed_mps": 18.0,
            "minimum_speed_mps_for_upper_bound": 1.2,
            "upper_bound_allowance_seconds": 120,
            "maximum_departures_per_stop": MAX_DEPARTURES,
        },
        "edges": edges,
        "edge_count": n,
        "accepted_edge_count": accepted,
        "accepted_edge_fraction": accepted / n,
        "reason_counts": dict(sorted(Counter(e["reason"] for e in edges).items())),
    }

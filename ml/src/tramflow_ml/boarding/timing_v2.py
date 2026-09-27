"""First-validation reconstruction with audited local edge timing, never measured stops."""

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .composition import RouteEvidence
from .dense_profiles import lookup_profile
from .real import burst_groups
from .sequence import SequenceConfig, infer_sequence


@dataclass(frozen=True)
class TimingConfig:
    timestamp: str = "first"
    gap_seconds: float = 30
    span_seconds: float = 90
    clock_sigma: float = 180
    clock_weight: float = 0.2
    validation_delay: float = 15

    def __post_init__(self) -> None:
        if self.timestamp not in {"first", "mean"}:
            raise ValueError("timestamp must be first or mean")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.gap_seconds, self.span_seconds, self.clock_sigma)
        ):
            raise ValueError("positive finite timing thresholds required")
        if not 0 <= self.clock_weight <= 1 or not math.isfinite(self.validation_delay):
            raise ValueError("invalid clock weight or validation delay")


DEFAULT_TIMING = TimingConfig()


def group_times(times: NDArray[np.float64], config: TimingConfig) -> tuple[Any, Any, Any]:
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("nonempty sorted finite event times required")
    groups = burst_groups(times, config.gap_seconds, config.span_seconds)
    starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
    counts = np.bincount(groups)
    observed = (
        times[starts]
        if config.timestamp == "first"
        else (np.bincount(groups, weights=times) / counts)
    )
    return groups, observed, counts


def clock_evidence(times: Any, evidence: RouteEvidence, config: TimingConfig) -> Any:
    """Common precision across states avoids the v1 mixed-source precision preference."""
    result = np.zeros((len(times), len(evidence.visits)))
    complete = all(
        not v["boardable"] or bool(d)
        for v, d in zip(evidence.visits, evidence.departures, strict=True)
    )
    clocks = (times + 10800 - config.validation_delay) % 86400
    for state, (visit, departures) in enumerate(
        zip(evidence.visits, evidence.departures, strict=True)
    ):
        if not visit["boardable"]:
            result[:, state] = -np.inf
        elif complete and config.clock_weight and departures:
            stamps = np.array(
                sorted({d + shift for d in departures for shift in (-86400, 0, 86400)})
            )
            idx = np.clip(np.searchsorted(stamps, clocks), 1, len(stamps) - 1)
            delta = np.minimum(abs(clocks - stamps[idx]), abs(clocks - stamps[idx - 1]))
            result[:, state] = config.clock_weight * np.log(
                0.15 + 0.85 * np.exp(-0.5 * (delta / config.clock_sigma) ** 2)
            )
    return result


def edge_durations(evidence: RouteEvidence, estimates: dict[str, Any]) -> tuple[Any, list[str]]:
    values = (
        np.array(evidence.template.edge_lengths_m) / 5
        + 20
        + np.array(evidence.template.terminal_after) * 120
    )
    basis = ["geometry_assumption"] * len(values)
    for edge in estimates["edges"]:
        if edge["accepted"]:
            state, seconds = edge["state"], edge["seconds"]
            if not 0 <= state < len(values) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid accepted schedule duration")
            values[state] = seconds
            basis[state] = "inferred_schedule_interval"
    return values, basis


def reconstruct_session(
    frame: Any,
    evidence: RouteEvidence,
    edge_seconds: Any,
    config: TimingConfig = DEFAULT_TIMING,
    profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = frame.sort_values(["second", "event_key"]).reset_index(drop=True)
    groups, times, counts = group_times(frame.second.to_numpy(dtype=float), config)
    n, states = len(times), len(evidence.visits)
    if len(frame.event_at.str[:10].unique()) != 1 or frame.event_key.duplicated().any():
        raise ValueError("unique events from one civil date required")
    day = str(frame.event_at.iloc[0])[:10]
    session_id = hashlib.sha256((str(frame.group.iloc[0]) + ":" + day).encode()).hexdigest()[:24]
    pattern_key = "|".join(
        [
            evidence.template.route,
            *evidence.template.pattern_ids,
            *evidence.template.stop_ids,
            *evidence.template.directions,
        ]
    )
    pattern_key = hashlib.sha256(pattern_key.encode()).hexdigest()[:24]
    applied_profiles = 0
    edge_seconds = np.array(edge_seconds, dtype=float, copy=True)
    if profiles is not None:
        for state in range(states):
            value = lookup_profile(profiles, day, pattern_key, state)
            if value is not None:
                edge_seconds[state] = value
                applied_profiles += 1
    deltas = np.diff(times)
    bounded = deltas[deltas <= 1800]
    limit = min(128, max(12, math.ceil(float(bounded.max()) / 30) + 3 if len(bounded) else 12))
    emission = clock_evidence(times, evidence, config)
    no_clock = np.where(np.isfinite(emission), 0.0, -np.inf)
    scenarios = [(1.0, emission), (1.25, emission), (5 / 6, emission), (1.0, no_clock)]
    outputs = [
        infer_sequence(
            times,
            evidence.template,
            SequenceConfig(max_steps=limit),
            log_emissions=e,
            edge_seconds=np.asarray(edge_seconds) * scale,
        )
        for scale, e in scenarios
    ]
    primary = outputs[0]
    path = primary.path_states
    probability = primary.posterior[np.arange(n), path]
    agreement = np.mean([o.path_states == path for o in outputs], axis=0)
    unique = primary.top_probabilities[:, 0] - primary.top_probabilities[:, 1] > 1e-9
    support = np.zeros(n, dtype=int)
    bounds = sorted({0, n, *primary.reset_indices})
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        support[a:b] = b - a
    stationary = np.zeros(n, dtype=bool)
    start = 0
    for i in range(1, n + 1):
        if i == n or path[i] != path[start] or primary.path_steps[i] != 0:
            if times[i - 1] - times[start] > 300:
                stationary[start:i] = True
            start = i
    eligible = (
        (probability >= 0.5)
        & (agreement >= 0.75)
        & unique
        & ~stationary
        & (support >= 4)
        & evidence.applicable_pattern
    )
    if any("step_bound_sensitive" in o.flags for o in outputs):
        eligible[:] = False
    visits = evidence.visits
    selected = [
        {
            "stop_id": visits[int(s)]["stop_id"] if unique[i] else None,
            "direction": visits[int(s)]["direction"] if unique[i] else None,
            "conditional_posterior": float(probability[i]),
            "scenario_agreement": float(agreement[i]),
            "raw_stable": bool(eligible[i]),
            "state": int(s),
        }
        for i, s in enumerate(path)
    ]
    rows = []
    for index, event in enumerate(frame.itertuples(index=False)):
        burst = int(groups[index])
        rows.append(
            {
                "event_key": event.event_key,
                "event_at": event.event_at,
                "route": event.route,
                "session_id": session_id,
                "burst": burst,
                **selected[burst],
                "raw_stable": bool(eligible[burst]) and not event.identity_conflict,
                "training_eligible": False,
            }
        )
    transitions = []
    for i in range(1, n):
        old, state, step = int(path[i - 1]), int(path[i]), int(primary.path_steps[i])
        transitions.append(
            {
                "date": day,
                "route": evidence.template.route,
                "session_id": session_id,
                "vehicle_key": str(frame.vehicle_key.iloc[0]),
                "pattern_key": pattern_key,
                "from_stop": str(visits[old]["stop_id"]),
                "to_stop": str(visits[state]["stop_id"]),
                "direction": str(visits[old]["direction"]),
                "from_state": old,
                "to_state": state,
                "steps": step,
                "observed_seconds": float(times[i] - times[i - 1]),
                "expected_seconds": float(primary.expected_seconds[i]),
                "residual_seconds": float(primary.residual_seconds[i]),
                "posterior": float(min(probability[i], probability[i - 1])),
                "agreement": float(min(agreement[i], agreement[i - 1])),
                "from_events": int(counts[i - 1]),
                "to_events": int(counts[i]),
                "identity_conflict": bool(frame.identity_conflict.any()),
                "anchor_eligible": bool(eligible[i] and eligible[i - 1]),
                "applicable_pattern": evidence.applicable_pattern,
                "skipped_stops": [
                    str(visits[(old + k) % states]["stop_id"]) for k in range(1, max(1, step))
                ],
            }
        )
    soft = []
    hours = frame.event_at.str[:13].to_numpy()
    for hour in sorted(set(hours)):
        mass = np.bincount(groups[hours == hour], minlength=n) @ primary.posterior
        if not np.isclose(mass.sum(), int((hours == hour).sum()), rtol=0, atol=1e-6):
            raise ValueError("original-hour posterior conservation failed")
        for state, value in enumerate(mass):
            if value > 0:
                soft.append(
                    {
                        "route": evidence.template.route,
                        "event_hour": hour,
                        "stop_id": str(visits[state]["stop_id"]),
                        "direction": str(visits[state]["direction"]),
                        "expected_count": float(value),
                    }
                )
    anchors = []
    for start in bounds[:-1]:
        order = np.argsort(-primary.posterior[start], kind="stable")[:8]
        for state in order:
            departures = evidence.departures[state]
            clock = float((times[start] + 10800 - config.validation_delay) % 86400)
            closest = (
                min(
                    (d + shift for d in departures for shift in (-86400, 0, 86400)),
                    key=lambda d: (abs(d - clock), d),
                )
                if departures
                else None
            )
            anchors.append(
                {
                    "session_id": session_id,
                    "burst": start,
                    "state": int(state),
                    "stop_id": str(visits[state]["stop_id"]),
                    "direction": str(visits[state]["direction"]),
                    "nearest_departure_seconds": closest,
                    "posterior": float(primary.posterior[start, state]),
                    "trip_id": None,
                    "kind": "inferred_phase_departure_candidate",
                }
            )
    return {
        "events": rows,
        "transitions": transitions,
        "soft": soft,
        "anchors": anchors,
        "pattern_key": pattern_key,
        "profile_edges_applied": applied_profiles,
        "bursts": n,
        "resets": len(primary.reset_indices),
        "flags": sorted(set(f for o in outputs for f in o.flags)),
        "candidate_visits": sum(1 for k in primary.path_steps if k != 0)
        + sum(max(int(k) - 1, 0) for k in primary.path_steps),
        "raw_stable": sum(r["raw_stable"] for r in rows),
    }

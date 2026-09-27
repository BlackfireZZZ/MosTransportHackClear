"""Bounded offline alignment; scores are costs, never calibrated probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    """Ordered visits, including repeated stop IDs; durations are explicit scenario priors."""

    pattern_id: str
    direction: str
    stop_ids: tuple[str, ...]
    travel_seconds: tuple[float, ...]
    dwell_seconds: float = 0.0
    historical_verified: bool = False

    def __post_init__(self) -> None:
        if not self.pattern_id or not self.direction or not self.stop_ids:
            raise ValueError("pattern identity, direction and visits are required")
        if any(not stop for stop in self.stop_ids):
            raise ValueError("empty stop ID")
        if len(self.travel_seconds) != len(self.stop_ids) - 1:
            raise ValueError("one travel duration per consecutive visit pair required")
        for duration in (*self.travel_seconds, self.dwell_seconds):
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("durations must be finite and nonnegative")

    @property
    def offsets(self) -> tuple[float, ...]:
        offsets = [0.0]
        for travel in self.travel_seconds:
            offsets.append(offsets[-1] + travel + self.dwell_seconds)
        return tuple(offsets)


@dataclass(frozen=True)
class DelayScenario:
    name: str = "zero_delay"
    payment_delay_min: float = 0.0
    payment_delay_max: float = 0.0
    clock_offset: float = 0.0

    def __post_init__(self) -> None:
        values = (self.payment_delay_min, self.payment_delay_max, self.clock_offset)
        if not self.name or not all(math.isfinite(value) for value in values):
            raise ValueError("named finite scenario required")
        if not 0 <= self.payment_delay_min <= self.payment_delay_max:
            raise ValueError("invalid payment delay interval")


@dataclass(frozen=True)
class Anchor:
    """External event-to-visit constraint; inferred pauses are not independent anchors."""

    pattern_id: str
    event_index: int
    visit_index: int
    source: str
    independent: bool = False


@dataclass(frozen=True)
class DecodeConfig:
    top_k: int = 5
    max_events: int = 200
    max_states: int = 100_000
    max_transitions: int = 1_000_000
    skip_cost: float = 1.0
    outlier_cost: float = 5.0
    ambiguity_margin: float = 0.0

    def __post_init__(self) -> None:
        for budget in (self.top_k, self.max_events, self.max_states, self.max_transitions):
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                raise ValueError("positive integer budget required")
        for cost in (self.skip_cost, self.outlier_cost, self.ambiguity_margin):
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("costs must be finite and nonnegative")


_DEFAULT_CONFIG = DecodeConfig()
_DEFAULT_SCENARIO = DelayScenario()


@dataclass(frozen=True)
class Candidate:
    pattern_id: str
    direction: str
    scenario: str
    visits: tuple[int | None, ...]
    score: float
    start_interval: tuple[float, float] | None
    latent_visits: tuple[int, ...]
    open_left: bool = True
    open_right: bool = True


@dataclass(frozen=True)
class AlignmentResult:
    method: str
    status: str
    candidates: tuple[Candidate, ...]
    stop_ids: tuple[str | None, ...]
    reasons: tuple[str, ...]
    transitions: int = 0
    feasible_paths: int = 0
    alternatives_omitted: int = 0
    approximation: bool = False
    calibrated: bool = False


@dataclass(frozen=True)
class _State:
    visits: tuple[int | None, ...] = ()
    last: int = -1
    lower: float = -math.inf
    upper: float = math.inf
    score: float = 0.0


def no_stop_baseline(event_count: int) -> AlignmentResult:
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
        raise ValueError("nonnegative event count required")
    return AlignmentResult("B0", "abstained", (), (None,) * event_count, ("no_stop_baseline",))


def _validate(
    timestamps: Sequence[float], patterns: Sequence[Pattern], anchors: Sequence[Anchor]
) -> None:
    if any(not math.isfinite(t) for t in timestamps):
        raise ValueError("timestamps must be finite")
    if any(a > b for a, b in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("timestamps must be sorted; ties are legal")
    by_id = {p.pattern_id: p for p in patterns}
    if len(by_id) != len(patterns):
        raise ValueError("pattern IDs must be unique")
    for anchor in anchors:
        if anchor.pattern_id not in by_id or not anchor.source:
            raise ValueError("anchor requires known pattern and provenance")
        if not 0 <= anchor.event_index < len(timestamps):
            raise ValueError("anchor event outside input")
        if not 0 <= anchor.visit_index < len(by_id[anchor.pattern_id].stop_ids):
            raise ValueError("anchor visit outside pattern")


def _candidate(state: _State, pattern: Pattern, scenario: DelayScenario) -> Candidate:
    observed = {visit for visit in state.visits if visit is not None}
    latent = (
        tuple(visit for visit in range(min(observed), max(observed) + 1) if visit not in observed)
        if observed
        else ()
    )
    interval = (state.lower, state.upper) if observed else None
    return Candidate(
        pattern.pattern_id,
        pattern.direction,
        scenario.name,
        state.visits,
        state.score,
        interval,
        latent,
    )


def decode(
    timestamps: Sequence[float],
    patterns: Sequence[Pattern],
    *,
    scenarios: Sequence[DelayScenario] = (DelayScenario(),),
    anchors: Sequence[Anchor] = (),
    config: DecodeConfig = _DEFAULT_CONFIG,
) -> AlignmentResult:
    """Exact bounded frontier expansion with interval feasibility, no beam pruning.

    Starts are unrestricted. Events may skip visits, share a visit or be outliers.
    Budget failure discards all partial results, including completed patterns.
    Absolute output requires unique best assignments across all scenarios, an
    independent anchor and a historically verified pattern. Such output remains
    weak labels: uniqueness under a prior is not measured stop accuracy.
    """
    _validate(timestamps, patterns, anchors)
    if not scenarios or len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("nonempty uniquely named scenarios required")
    unknown = (None,) * len(timestamps)
    if not timestamps:
        return AlignmentResult("exact_monotone", "empty", (), (), ("empty_input",))
    transitions = 0

    def exhausted() -> AlignmentResult:
        return AlignmentResult(
            "exact_monotone",
            "undecoded",
            (),
            unknown,
            ("budget_exhausted",),
            transitions,
        )

    if len(timestamps) > config.max_events:
        return exhausted()
    candidates: list[Candidate] = []
    for pattern in patterns:
        if any(anchor.pattern_id != pattern.pattern_id for anchor in anchors):
            continue
        offsets = pattern.offsets
        for scenario in scenarios:
            states = [_State()]
            for event_index, timestamp in enumerate(timestamps):
                required = {a.visit_index for a in anchors if a.event_index == event_index}
                if len(required) > 1:
                    states = []
                    break
                next_states: list[_State] = []
                for state in states:
                    choices: list[int | None] = list(range(max(0, state.last), len(offsets)))
                    if not required:
                        choices.append(None)
                    for visit in choices:
                        transitions += 1
                        if transitions > config.max_transitions:
                            return exhausted()
                        if required and visit not in required:
                            continue
                        if visit is None:
                            new_state = _State(
                                (*state.visits, None),
                                state.last,
                                state.lower,
                                state.upper,
                                state.score + config.outlier_cost,
                            )
                        else:
                            lower = max(
                                state.lower,
                                timestamp
                                - scenario.clock_offset
                                - scenario.payment_delay_max
                                - offsets[visit]
                                - pattern.dwell_seconds,
                            )
                            upper = min(
                                state.upper,
                                timestamp
                                - scenario.clock_offset
                                - scenario.payment_delay_min
                                - offsets[visit],
                            )
                            if lower > upper:
                                continue
                            skipped = max(0, visit - state.last - 1) if state.last >= 0 else 0
                            new_state = _State(
                                (*state.visits, visit),
                                visit,
                                lower,
                                upper,
                                state.score + skipped * config.skip_cost,
                            )
                        next_states.append(new_state)
                        if len(next_states) + len(candidates) > config.max_states:
                            return exhausted()
                states = next_states
            candidates.extend(_candidate(state, pattern, scenario) for state in states)
    if not candidates:
        return AlignmentResult(
            "exact_monotone", "abstained", (), unknown, ("pattern_mismatch",), transitions
        )
    candidates.sort(
        key=lambda c: (
            c.score,
            c.pattern_id,
            c.scenario,
            tuple(-1 if v is None else v for v in c.visits),
        )
    )
    best = candidates[0]
    contenders = [c for c in candidates if c.score <= best.score + config.ambiguity_margin]
    assignments = {(c.pattern_id, c.visits) for c in contenders}
    pattern = next(p for p in patterns if p.pattern_id == best.pattern_id)
    reasons: list[str] = []
    if len(assignments) > 1:
        reasons.append("ambiguous_alignment")
    if not pattern.historical_verified:
        reasons.append("pattern_version_unverified")
    if not any(a.independent and a.pattern_id == best.pattern_id for a in anchors):
        reasons.append("no_independent_anchor")
    stop_ids = (
        unknown
        if reasons
        else tuple(None if visit is None else pattern.stop_ids[visit] for visit in best.visits)
    )
    return AlignmentResult(
        "exact_monotone",
        "relative_only" if reasons else "anchored_weak_labels",
        tuple(candidates[: config.top_k]),
        stop_ids,
        tuple(reasons),
        transitions,
        len(candidates),
        max(0, len(candidates) - config.top_k),
    )


def greedy_baseline(
    timestamps: Sequence[float],
    pattern: Pattern,
    *,
    anchor: Anchor | None = None,
    scenario: DelayScenario = _DEFAULT_SCENARIO,
    tolerance_seconds: float = 0.0,
) -> AlignmentResult:
    """B2 nearest expected visit with monotone greedy assignment after an explicit anchor.

    Fixes the start scenario using minimum delay at the anchor event; uncertainty
    and variable payment delay belong in the exact decoder.
    """
    _validate(timestamps, (pattern,), () if anchor is None else (anchor,))
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("finite nonnegative tolerance required")
    if anchor is None:
        return AlignmentResult("B2", "abstained", (), (None,) * len(timestamps), ("no_anchor",))
    offsets = pattern.offsets
    start = (
        timestamps[anchor.event_index]
        - scenario.clock_offset
        - scenario.payment_delay_min
        - offsets[anchor.visit_index]
    )
    visits: list[int | None] = []
    last = 0
    for event_index, timestamp in enumerate(timestamps):
        upper_visit = anchor.visit_index if event_index < anchor.event_index else len(offsets) - 1
        residuals = [
            (
                abs(
                    timestamp
                    - scenario.clock_offset
                    - scenario.payment_delay_min
                    - start
                    - offsets[visit]
                ),
                visit,
            )
            for visit in range(last, upper_visit + 1)
        ]
        residual, visit = min(residuals)
        tied = sum(abs(r - residual) < 1e-9 for r, _ in residuals) > 1
        if event_index == anchor.event_index:
            visit, residual, tied = anchor.visit_index, 0.0, False
        if residual > tolerance_seconds or tied:
            visits.append(None)
        else:
            visits.append(visit)
            last = visit
    candidate = _candidate(_State(tuple(visits), last, start, start), pattern, scenario)
    reasons = (
        ()
        if pattern.historical_verified and anchor.independent
        else ("unverified_pattern_or_anchor",)
    )
    return AlignmentResult(
        "B2",
        "anchored_weak_labels" if not reasons else "relative_only",
        (candidate,),
        tuple(None if reasons or visit is None else pattern.stop_ids[visit] for visit in visits),
        reasons,
    )

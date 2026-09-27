"""Cutoff-safe, model-agnostic review of a proposed forecast switch."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Literal


@dataclass(frozen=True)
class ShiftSignal:
    entity: str
    direction: Literal["up", "down", "changed"]
    value: float
    recent: float
    reference: float
    kind: Literal["demand_level", "hour_shape"] = "demand_level"
    metric: Literal["ratio", "total_variation"] = "ratio"


@dataclass(frozen=True)
class Evaluation:
    score: float
    entity_scores: dict[str, float | None]


@dataclass(frozen=True)
class GuardPolicy:
    horizon_days: int = 61
    min_long_window_days: int = 28
    min_comparable_origins: int = 2
    min_score_gain: float = 0.002


@dataclass(frozen=True)
class CandidateReview:
    model: str
    gains: tuple[float, ...]
    shifted_entity_gains: tuple[float, ...]
    eligible: bool


@dataclass(frozen=True)
class GuardReport:
    origin: str
    incumbent: str
    candidates: tuple[str, ...]
    signals: tuple[ShiftSignal, ...]
    diagnostic_origins: tuple[str, ...]
    comparable_origins: tuple[str, ...]
    reviews: tuple[CandidateReview, ...]
    status: Literal[
        "not_long_window", "no_shift", "insufficient_evidence",
        "candidate_rejected", "switch_supported",
    ]
    selected_model: str
    incumbent_window_days: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def review_switch(
    *,
    origin: date,
    first_review_origin: date,
    incumbent: str,
    candidates: tuple[str, ...],
    incumbent_window_days: int,
    detect: Callable[[date], tuple[ShiftSignal, ...]],
    evaluate: Callable[[str, date], Evaluation],
    policy: GuardPolicy | None = None,
) -> GuardReport:
    """Require gains on disjoint, completed horizons with a matching prior shift."""
    policy = policy or GuardPolicy()
    if (policy.horizon_days < 1 or policy.min_comparable_origins < 1
            or policy.min_score_gain < 0):
        raise ValueError("invalid guard policy")
    if incumbent_window_days < 1 or not candidates:
        raise ValueError("positive incumbent window and candidates are required")
    if len(set((incumbent, *candidates))) != len(candidates) + 1:
        raise ValueError("models must be distinct")
    if incumbent_window_days < policy.min_long_window_days:
        return GuardReport(origin.isoformat(), incumbent, candidates, (), (), (), (),
                           "not_long_window", incumbent, incumbent_window_days)
    signals = detect(origin)
    if not signals:
        return GuardReport(origin.isoformat(), incumbent, candidates, (), (), (), (),
                           "no_shift", incumbent, incumbent_window_days)
    active = {(item.kind, item.entity, item.direction) for item in signals}
    prior: list[tuple[date, tuple[str, ...]]] = []
    cursor = origin - timedelta(days=policy.horizon_days)
    while cursor >= first_review_origin:
        matching = tuple(
            item.entity for item in detect(cursor)
            if (item.kind, item.entity, item.direction) in active
        )
        prior.append((cursor, matching))
        cursor -= timedelta(days=policy.horizon_days)
    diagnostic_origins = tuple(day.isoformat() for day, _ in prior)
    comparable_origins = tuple(day.isoformat() for day, entities in prior if entities)
    if not prior:
        return GuardReport(
            origin.isoformat(), incumbent, candidates, signals, diagnostic_origins,
            comparable_origins, (),
            "insufficient_evidence", incumbent, incumbent_window_days,
        )
    reviews: list[CandidateReview] = []
    reference = {day: evaluate(incumbent, day) for day, _ in prior}
    for candidate in candidates:
        gains: list[float] = []
        entity_gains: list[float] = []
        matching_gains: list[float] = []
        complete_entity_comparisons = True
        for day, entities in prior:
            baseline = reference[day]
            alternative = evaluate(candidate, day)
            gains.append(alternative.score - baseline.score)
            if entities:
                matching_gains.append(gains[-1])
            for entity in entities:
                base_score = baseline.entity_scores.get(entity)
                alt_score = alternative.entity_scores.get(entity)
                if base_score is not None and alt_score is not None:
                    entity_gains.append(alt_score - base_score)
                else:
                    complete_entity_comparisons = False
        eligible = (
            len(comparable_origins) >= policy.min_comparable_origins
            and all(gain >= 0 for gain in gains)
            and all(gain >= policy.min_score_gain for gain in matching_gains)
            and complete_entity_comparisons
            and len(entity_gains) >= len(comparable_origins)
            and all(gain > 0 for gain in entity_gains)
        )
        reviews.append(CandidateReview(candidate, tuple(gains), tuple(entity_gains), eligible))
    winners = [item for item in reviews if item.eligible]
    selected = max(winners, key=lambda item: min(item.gains)).model if winners else incumbent
    status: Literal["switch_supported", "candidate_rejected", "insufficient_evidence"]
    if winners:
        status = "switch_supported"
    elif len(comparable_origins) >= policy.min_comparable_origins:
        status = "candidate_rejected"
    else:
        status = "insufficient_evidence"
    return GuardReport(
        origin.isoformat(), incumbent, candidates, signals, diagnostic_origins,
        comparable_origins, tuple(reviews),
        status, selected,
        incumbent_window_days,
    )

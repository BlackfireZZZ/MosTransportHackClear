"""Synthetic evidence only; generated labels are never real boarding-stop ground truth."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from tramflow_ml.boarding.alignment import (
    AlignmentResult,
    Anchor,
    DecodeConfig,
    DelayScenario,
    Pattern,
    decode,
    greedy_baseline,
    no_stop_baseline,
)


@dataclass(frozen=True)
class SyntheticObservation:
    """Decoder inputs exclude evaluation truth and generator parameters."""

    timestamps: tuple[float, ...]
    patterns: tuple[Pattern, ...]
    anchors: tuple[Anchor, ...]
    scenarios: tuple[DelayScenario, ...]


@dataclass(frozen=True)
class SyntheticTruth:
    stop_ids: tuple[str, ...]
    visits: tuple[int, ...]
    pattern_id: str


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    family: str
    observations: SyntheticObservation
    truth: SyntheticTruth
    expected_abstention: bool
    missing_consecutive_visits: int
    challenge: str


_CASES = (
    "dense",
    "missing_1",
    "missing_3",
    "missing_5",
    "open_ends",
    "sparse_directions",
    "payment_delays",
    "clock_offset",
    "wrong_pattern_unverified",
    "corrupted_anchor",
    "repeated_stop",
    "no_anchor",
)
_STOPS = ("s0", "s1", "s2", "s0", "s4", "s5", "s6", "s7")
_DEFAULT_BUDGET = DecodeConfig(max_events=16, max_states=20_000, max_transitions=100_000)


def _case(family: str, name: str, rng: random.Random) -> SyntheticCase:
    arrivals = [100.0]
    correlated = 0.0
    for _ in range(7):
        if family == "renewal":
            gap = rng.uniform(25.0, 35.0)
        else:
            correlated = max(-8.0, min(8.0, correlated + rng.uniform(-4.0, 4.0)))
            gap = 30.0 + correlated
        arrivals.append(arrivals[-1] + gap)
    missing = int(name.removeprefix("missing_")) if name.startswith("missing_") else 0
    keep = [visit for visit in range(8) if not 1 <= visit <= missing]
    if name == "open_ends":
        keep = [2, 3, 4, 5]
    if name == "sparse_directions":
        keep = [3]
    events: list[tuple[float, int]] = []
    clock = 17.0 if name == "clock_offset" else 0.0
    for visit in keep:
        delay = rng.uniform(0.0, 8.0)
        if name == "payment_delays" and visit % 3 == 1:
            delay += 42.0
        timestamp = arrivals[visit] + delay + clock
        if family == "correlated_batching":
            timestamp = float(round(timestamp / 10.0) * 10)
        events.append((timestamp, visit))
        if name == "repeated_stop" and visit in (0, 3):
            events.append((timestamp + 1.0, visit))
    events.sort()
    stops: tuple[str, ...] = _STOPS
    verified = name != "wrong_pattern_unverified"
    if not verified:
        stops = tuple(f"wrong_{stop}" for stop in stops)
    primary = Pattern("out", "outbound", stops, (30.0,) * 7, historical_verified=verified)
    reverse = Pattern(
        "in", "inbound", tuple(reversed(stops)), (30.0,) * 7, historical_verified=verified
    )
    anchors: tuple[Anchor, ...] = ()
    if name not in ("no_anchor", "sparse_directions"):
        external_visit = events[0][1]
        if name == "corrupted_anchor":
            external_visit = min(7, external_visit + 1)
        anchors = (
            Anchor(
                "out",
                0,
                external_visit,
                "synthetic_external_anchor_corrupted"
                if name == "corrupted_anchor"
                else "synthetic_independent_visit_observation",
                True,
            ),
        )
    scenarios: tuple[DelayScenario, ...] = (DelayScenario("broad_payment_prior", 0.0, 20.0),)
    if name == "clock_offset":
        scenarios += (DelayScenario("clock_plus_17", 0.0, 20.0, 17.0),)
    observation = SyntheticObservation(
        tuple(t for t, _ in events), (primary, reverse), anchors, scenarios
    )
    truth = SyntheticTruth(tuple(_STOPS[v] for _, v in events), tuple(v for _, v in events), "out")
    challenge = (
        "Renewal travel intervals and independent bounded payment jitter."
        if family == "renewal"
        else "Correlated travel disturbances and quantized common-clock batches."
    )
    if name == "payment_delays":
        challenge += " Long payments can cross visit order and violate the decoder prior."
    if name == "corrupted_anchor":
        challenge += " Incorrect external anchor tests reliance on supplied anchor validity."
    return SyntheticCase(
        name,
        family,
        observation,
        truth,
        name in ("no_anchor", "sparse_directions", "wrong_pattern_unverified"),
        missing,
        challenge,
    )


def generate_synthetic_cases(seed: int = 20260926) -> tuple[SyntheticCase, ...]:
    """Two fixed generator families, 24 cases; no fitted priors or decoder-derived truth."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("integer seed required")
    rng = random.Random(seed)
    return tuple(
        _case(family, name, rng) for family in ("renewal", "correlated_batching") for name in _CASES
    )


def predict_synthetic(
    observation: SyntheticObservation,
    config: DecodeConfig = _DEFAULT_BUDGET,
) -> dict[str, AlignmentResult]:
    """Accept observations only, with no evaluation labels or gold-path parameter."""
    return {
        "B0": no_stop_baseline(len(observation.timestamps)),
        "B2": greedy_baseline(
            observation.timestamps,
            observation.patterns[0],
            anchor=observation.anchors[0] if observation.anchors else None,
            scenario=observation.scenarios[0],
            tolerance_seconds=20.0,
        ),
        "exact_monotone": decode(
            observation.timestamps,
            observation.patterns,
            anchors=observation.anchors,
            scenarios=observation.scenarios,
            config=config,
        ),
    }


def _metrics(result: AlignmentResult, truth: SyntheticTruth) -> dict[str, Any]:
    total = len(truth.stop_ids)
    assigned = sum(stop is not None for stop in result.stop_ids)
    exact_stops = sum(
        predicted == actual
        for predicted, actual in zip(result.stop_ids, truth.stop_ids, strict=True)
    )
    exact_visits = 0
    stop_correct_visit_wrong = 0
    visit_errors: list[int] = []
    if result.candidates:
        candidate = result.candidates[0]
        exact_visits = sum(
            stop is not None and candidate.pattern_id == truth.pattern_id and visit == gold
            for stop, visit, gold in zip(
                result.stop_ids, candidate.visits, truth.visits, strict=True
            )
        )
        stop_correct_visit_wrong = sum(
            predicted == stop and (candidate.pattern_id != truth.pattern_id or visit != gold)
            for predicted, stop, visit, gold in zip(
                result.stop_ids, truth.stop_ids, candidate.visits, truth.visits, strict=True
            )
        )
        if candidate.pattern_id == truth.pattern_id:
            visit_errors = [
                abs(visit - gold)
                for stop, visit, gold in zip(
                    result.stop_ids, candidate.visits, truth.visits, strict=True
                )
                if stop is not None and visit is not None
            ]
    return {
        "synthetic_visit_sequence_exact_all_events": bool(total and exact_visits == total),
        "synthetic_visit_sequence_errors_assigned": assigned - exact_visits,
        "stop_correct_but_visit_wrong_events": stop_correct_visit_wrong,
        "mean_absolute_visit_index_error_same_pattern_assigned": (
            sum(visit_errors) / len(visit_errors) if visit_errors else None
        ),
        "total_events": total,
        "assigned_events": assigned,
        "unknown_events": total - assigned,
        "coverage": assigned / total if total else 0.0,
        "correct_stop_events": exact_stops,
        "correct_visit_events": exact_visits,
        "synthetic_stop_accuracy_assigned": exact_stops / assigned if assigned else None,
        "synthetic_visit_accuracy_assigned": exact_visits / assigned if assigned else None,
        "synthetic_stop_correct_fraction_all_events": exact_stops / total if total else None,
        "synthetic_visit_correct_fraction_all_events": exact_visits / total if total else None,
        "status": result.status,
        "reasons": list(result.reasons),
        "transitions": result.transitions,
        "budget_exhausted": "budget_exhausted" in result.reasons,
        "calibrated": result.calibrated,
    }


def evaluate_synthetic(
    seed: int = 20260926,
    config: DecodeConfig = _DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Reproducible JSON-compatible evidence; synthetic success cannot promote real labels."""
    cases: list[dict[str, Any]] = []
    for case in generate_synthetic_cases(seed):
        predictions = predict_synthetic(case.observations, config)
        cases.append(
            {
                "case": case.name,
                "family": case.family,
                "challenge": case.challenge,
                "expected_abstention": case.expected_abstention,
                "missing_consecutive_visits": case.missing_consecutive_visits,
                "anchor_source": [anchor.source for anchor in case.observations.anchors],
                "metrics": {
                    method: _metrics(result, case.truth) for method, result in predictions.items()
                },
            }
        )
    summary = {}
    for method in ("B0", "B2", "exact_monotone"):
        rows = [case["metrics"][method] for case in cases]
        total = sum(row["total_events"] for row in rows)
        assigned = sum(row["assigned_events"] for row in rows)
        stops = sum(row["correct_stop_events"] for row in rows)
        visits = sum(row["correct_visit_events"] for row in rows)
        summary[method] = {
            "total_events": total,
            "assigned_events": assigned,
            "unknown_events": total - assigned,
            "coverage": assigned / total if total else 0.0,
            "synthetic_stop_accuracy_assigned": stops / assigned if assigned else None,
            "synthetic_visit_accuracy_assigned": visits / assigned if assigned else None,
            "synthetic_stop_correct_fraction_all_events": stops / total if total else None,
            "synthetic_visit_correct_fraction_all_events": visits / total if total else None,
            "budget_exhausted_cases": sum(row["budget_exhausted"] for row in rows),
        }
    return {
        "schema_version": "boarding.synthetic-evaluation.v1",
        "seed": seed,
        "config": {"decoder": asdict(config), "B2_tolerance_seconds": 20.0},
        "case_count": len(cases),
        "evidence_kind": "synthetic_independent_generator_truth",
        "real_stop_accuracy": "unverified",
        "real_label_promotion_allowed": False,
        "limitations": [
            "Hand-selected adversarial cases are not representative of Moscow transit.",
            "Generator parameters and anchor provenance are synthetic assumptions.",
            "Fixed-duration decoder is intentionally misspecified against both generators.",
            "Scores and conditional synthetic accuracy are not calibrated probabilities.",
            "No HMM or HSMM promotion without independent real gold.",
        ],
        "summary": summary,
        "cases": cases,
    }

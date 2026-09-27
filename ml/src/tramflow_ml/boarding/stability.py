"""Bounded perturbation diagnostics for device-level partitions, not stop accuracy."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .bursts import Payment, detect

_SAMPLE_CAP = 10_000
Group = tuple[str, str, str, str]


def _group(event: Payment) -> Group:
    return event.device, event.vehicle, event.route, event.exit


def _rank(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value!r}".encode()).hexdigest()


def _partition(events: Sequence[Payment]) -> dict[str, int]:
    return {key: number for number, burst in enumerate(detect(events)) for key in burst.keys}


def _compare(
    original: Sequence[Payment],
    perturbed: Sequence[Payment],
    baseline: dict[str, int],
) -> dict[str, Any]:
    predicted = _partition(perturbed)
    surviving = set(predicted)
    ordered: dict[Group, list[Payment]] = defaultdict(list)
    for event in original:
        if event.event_key in surviving:
            ordered[_group(event)].append(event)
    expected_edges: set[tuple[str, str]] = set()
    actual_edges: set[tuple[str, str]] = set()
    pairs = 0
    for group in ordered.values():
        group.sort(key=lambda event: (event.second, event.event_key))
        for first, second in zip(group, group[1:], strict=False):
            pairs += 1
            pair = (first.event_key, second.event_key)
            if baseline[pair[0]] == baseline[pair[1]]:
                expected_edges.add(pair)
            if predicted[pair[0]] == predicted[pair[1]]:
                actual_edges.add(pair)
    intersection = len(expected_edges & actual_edges)
    union = len(expected_edges | actual_edges)
    original_success = sum(event.success for event in original)
    retained_success = sum(event.success for event in perturbed)
    return {
        "sample_events": len(original),
        "retained_events": len(perturbed),
        "dropped_events": len(original) - len(perturbed),
        "retained_fraction": len(perturbed) / len(original) if original else None,
        "sample_successful": original_success,
        "retained_successful": retained_success,
        "dropped_successful": original_success - retained_success,
        "adjacent_pairs_evaluated": pairs,
        "baseline_same_burst_edges": len(expected_edges),
        "perturbed_same_burst_edges": len(actual_edges),
        "preserved_same_burst_edges": intersection,
        "same_burst_edge_union": union,
        "adjacency_retention": intersection / len(expected_edges) if expected_edges else None,
        "adjacency_jaccard": intersection / union if union else None,
        "adjacency_agreement": (
            (pairs - len(expected_edges ^ actual_edges)) / pairs if pairs else None
        ),
        "retained_bursts": len(set(predicted.values())),
        "geographic_accuracy": "unverified",
    }


def evaluate_stability(events: Sequence[Payment], seed: int = 20260926) -> dict[str, Any]:
    """Fixed perturbations; selection includes whole declared identity groups only.

    Comparisons use adjacent retained events in the baseline temporal order within
    each declared group. Unknown identities remain unknown. Dropped observations
    do not count as matched, and their denominators are reported separately.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("integer seed required")
    if len({event.event_key for event in events}) != len(events):
        raise ValueError("duplicate event key")
    counts = Counter(_group(event) for event in events)
    selected: set[Group] = set()
    remaining = _SAMPLE_CAP
    for group in sorted(counts, key=lambda group: (_rank(group, seed), group)):
        if counts[group] <= remaining:
            selected.add(group)
            remaining -= counts[group]
    sample = sorted(
        (event for event in events if _group(event) in selected),
        key=lambda event: (*_group(event), event.second, event.event_key),
    )
    baseline = _partition(sample)
    cases: list[dict[str, Any]] = []

    def add(name: str, changed: Sequence[Payment], config: dict[str, Any]) -> None:
        cases.append(
            {"perturbation": name, "config": config, "metrics": _compare(sample, changed, baseline)}
        )

    add("identity", sample, {})
    for amplitude in (5, 15, 30):
        rng = random.Random(f"{seed}:jitter:{amplitude}")
        changed = [
            replace(event, second=event.second + rng.uniform(-amplitude, amplitude))
            for event in sample
        ]
        add(
            f"jitter_{amplitude}s",
            changed,
            {"uniform_min_seconds": -amplitude, "uniform_max_seconds": amplitude},
        )
    rng = random.Random(f"{seed}:thinning")
    add(
        "thin_30_percent",
        [event for event in sample if rng.random() >= 0.3],
        {"independent_drop_probability": 0.3},
    )
    burst_ids = sorted(set(baseline.values()))
    dropped_burst = random.Random(f"{seed}:burst").choice(burst_ids) if burst_ids else None
    add(
        "drop_one_whole_burst",
        [event for event in sample if baseline[event.event_key] != dropped_burst],
        {"applied": dropped_burst is not None},
    )
    devices = sorted(
        {event.device for event in sample if event.device},
        key=lambda device: (_rank(device, seed), device),
    )
    device = devices[0] if devices else None
    add(
        "drop_one_device",
        [event for event in sample if event.device != device],
        {"applied": device is not None, "device_selection": "first seeded hash rank"},
    )
    shifted = [
        replace(event, second=event.second + 30)
        if device is not None and event.device == device
        else event
        for event in sample
    ]
    add(
        "shift_one_device_clock",
        shifted,
        {
            "applied": device is not None,
            "offset_seconds": 30,
            "expected_invariance": "D0 compares within-group gaps; cannot validate clock accuracy",
        },
    )
    add(
        "global_start_shift",
        [replace(event, second=event.second + 3600) for event in sample],
        {"offset_seconds": 3600, "expected_invariance": "constant shift preserves intervals"},
    )
    return {
        "schema_version": "boarding-stability.v1",
        "seed": seed,
        "detector": {"method": "D0", "gap_seconds": 30, "max_span_seconds": 90},
        "selection": {
            "method": "whole declared device/vehicle/route/exit groups in seeded SHA256 order",
            "max_sample_events": _SAMPLE_CAP,
            "original_events": len(events),
            "original_successful": sum(event.success for event in events),
            "sample_events": len(sample),
            "sample_successful": sum(event.success for event in sample),
            "omitted_events": len(events) - len(sample),
            "original_groups": len(counts),
            "selected_groups": len(selected),
            "omitted_groups": len(counts) - len(selected),
            "oversized_groups": sum(count > _SAMPLE_CAP for count in counts.values()),
            "sampling_truncated": len(sample) < len(events),
            "partial_groups": False,
            "empty_reason": (
                "all_groups_exceed_budget"
                if counts and not sample
                else "no_input"
                if not events
                else None
            ),
            "selected_event_keys_sha256": hashlib.sha256(
                repr(tuple(event.event_key for event in sample)).encode()
            ).hexdigest(),
        },
        "comparison": "same-burst decisions on adjacent retained events in baseline group order",
        "undefined_metric": "null when no relevant adjacent pairs or same-burst edges exist",
        "geographic_accuracy": "unverified",
        "calibrated": False,
        "fit_performed": False,
        "cases": cases,
    }

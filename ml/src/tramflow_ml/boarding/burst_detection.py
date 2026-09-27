"""Compare validation onsets against local rate; inferred pulses are not door telemetry."""

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]

from .dense_audit import bursts

METHODS = ("capped30_90", "gap15", "gap30", "dbscan8", "rate_dbscan", "decay_onset")


def intensity(times: Any) -> tuple[Any, Any, Any, float]:
    times = np.asarray(times, dtype=float)
    if (
        not len(times)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) < 0)
        or times[-1] - times[0] > 86400
    ):
        raise ValueError("sorted finite events spanning at most one day required")
    origin = float(np.floor(times[0] / 5) * 5)
    bins = ((times - origin) // 5).astype(int)
    counts = np.bincount(bins, minlength=61).astype(float)
    kernel = np.ones(61)
    baseline = np.convolve(counts, kernel, mode="same") / np.convolve(
        np.ones(len(counts)), kernel, mode="same"
    )
    return counts, np.maximum(baseline, 0.1), bins, origin


def _clusters(times: Any, labels: Any) -> dict[str, Any]:
    start, count, duration = [], [], []
    for label in sorted(set(labels) - {-1}):
        selected = times[labels == label]
        if selected[-1] - selected[0] > 180:
            continue
        start.append(float(selected[0]))
        count.append(len(selected))
        duration.append(float(selected[-1] - selected[0]))
    order = np.argsort(start)
    return {
        "times": np.array(start)[order],
        "counts": np.array(count)[order],
        "durations": np.array(duration)[order],
        "forced_boundaries": np.zeros(max(0, len(start) - 1), dtype=bool),
    }


def detect(times: Any, method: str) -> dict[str, Any]:
    times = np.asarray(times, dtype=float)
    counts, baseline, bins, origin = intensity(times)
    if method == "capped30_90":
        result = bursts(times, 30, 90)
    elif method in {"gap15", "gap30"}:
        result = bursts(times, 15 if method == "gap15" else 30, None)
    elif method in {"dbscan8", "rate_dbscan"}:
        if method == "dbscan8":
            values, eps, minimum = times - times[0], 8.0, 3
        else:
            cumulative = np.r_[0, np.cumsum(baseline)]
            values = cumulative[bins] + ((times - origin) % 5) / 5 * baseline[bins]
            eps, minimum = 0.75, 4
        labels = DBSCAN(eps=eps, min_samples=minimum, algorithm="kd_tree").fit_predict(
            values[:, None]
        )
        result = _clusters(times, labels)
    elif method == "decay_onset":
        weights = np.exp(-np.arange(12) * 5 / 15)
        score = np.zeros(len(counts))
        for i in range(4, len(counts) - 12):
            future = counts[i : i + 12]
            expected = baseline[i : i + 12]
            excess = float(np.dot(future - expected, weights))
            sigma = np.sqrt(float(np.dot(expected, weights**2)))
            early, late = float(future[:3].sum()), float(future[3:6].sum())
            before = float(counts[i - 4 : i].mean())
            if future[:2].sum() >= 3 and early >= late and early / 3 > 1.5 * max(0.1, before):
                score[i] = excess / sigma
        candidates = np.flatnonzero(score >= 2.5)
        chosen: list[int] = []
        for i in sorted(candidates.tolist(), key=lambda i: (-score[i], i)):
            if all(abs(i - j) >= 6 for j in chosen):
                chosen.append(int(i))
        chosen.sort()
        labels = np.full(len(times), -1)
        for label, i in enumerate(chosen):
            end = min(
                origin + (i + 12) * 5,
                origin + chosen[label + 1] * 5 if label + 1 < len(chosen) else np.inf,
            )
            labels[(times >= origin + i * 5) & (times < end)] = label
        result = _clusters(times, labels)
    else:
        raise ValueError("unknown burst detector")
    result["source_events"] = len(times)
    result["covered_events"] = int(sum(result["counts"]))
    if result["covered_events"] > len(times):
        raise ValueError("detector duplicated source events")
    return result


def randomize_local_rate(times: Any, seed: int) -> Any:
    """Destroy sub-five-minute structure while preserving counts in each fixed 300s bin."""
    times = np.asarray(times, dtype=float)
    rng = np.random.default_rng(seed)
    return np.sort(np.floor(times / 300) * 300 + rng.uniform(0, 300, len(times)))


def match_onsets(reference: Any, detected: Any, tolerance: float = 15) -> dict[str, Any]:
    reference, detected = np.sort(reference), np.sort(detected)
    i = j = hits = 0
    while i < len(reference) and j < len(detected):
        if abs(reference[i] - detected[j]) <= tolerance:
            hits += 1
            i += 1
            j += 1
        elif reference[i] < detected[j]:
            i += 1
        else:
            j += 1
    return {
        "reference": len(reference),
        "detected": len(detected),
        "matched": hits,
        "precision": hits / len(detected) if len(detected) else None,
        "recall": hits / len(reference) if len(reference) else None,
    }


def heldout_shape(onsets: Any, heldout: Any, start: float, end: float) -> dict[str, Any]:
    """Measure rise/decay on events not used to detect onsets; no door-time claim."""
    onsets, heldout = np.asarray(onsets), np.asarray(heldout)
    onsets = onsets[(onsets >= start + 30) & (onsets <= end - 60)]
    before = early = late = 0
    for t in onsets:
        before += int(np.count_nonzero((heldout >= t - 30) & (heldout < t)))
        early += int(np.count_nonzero((heldout >= t) & (heldout < t + 15)))
        late += int(np.count_nonzero((heldout >= t + 30) & (heldout < t + 60)))
    return {
        "onsets": len(onsets),
        "before_events": before,
        "early_events": early,
        "late_events": late,
        "early_over_before_rate": 2 * early / before if before else None,
        "early_over_late_rate": 2 * early / late if late else None,
    }

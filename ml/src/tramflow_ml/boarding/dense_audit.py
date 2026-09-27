"""Test whether dense payment bursts identify ordered stop-time fingerprints."""

from typing import Any

import numpy as np


def bursts(times: Any, gap: float, span: float | None) -> dict[str, Any]:
    times = np.asarray(times, dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("nonempty sorted finite times required")
    if (
        gap <= 0
        or not np.isfinite(gap)
        or (span is not None and (span <= 0 or not np.isfinite(span)))
    ):
        raise ValueError("positive finite burst thresholds required")
    starts, forced = [0], []
    for i in range(1, len(times)):
        quiet = times[i] - times[i - 1] > gap
        cap = span is not None and times[i] - times[starts[-1]] > span
        if quiet or cap:
            starts.append(i)
            forced.append(bool(cap and not quiet))
    ends = np.array([*starts[1:], len(times)], dtype=int)
    return {
        "times": times[starts],
        "counts": ends - starts,
        "durations": times[ends - 1] - times[starts],
        "forced_boundaries": np.array(forced, dtype=bool),
    }


def dense_window(group: dict[str, Any], length: int = 12) -> int | None:
    """Pick by observed counts only; retain sparse bursts within the window."""
    counts, times = group["counts"], group["times"]
    if len(times) < length:
        return None
    candidates = []
    for start in range(len(times) - length + 1):
        mass = counts[start : start + length]
        interval = np.diff(times[start : start + length])
        duration = group["durations"][start : start + length]
        if (
            np.count_nonzero(mass >= 3) >= 8
            and max(interval) <= 600
            and min(interval) >= 15
            and max(duration) <= 180
        ):
            candidates.append((int(mass.sum()), -start))
    return -max(candidates)[1] if candidates else None


def align(
    gaps: Any,
    edges: Any,
    boardable: Any,
    *,
    max_step: int = 3,
    scales: tuple[float, ...] = (0.8, 1.0, 1.2),
    skip_penalty: float = 15,
) -> dict[str, Any]:
    """Exhaust all starts, bounded forward paths and scales; no absolute clock anchor."""
    gaps, edges = np.asarray(gaps, dtype=float), np.asarray(edges, dtype=float)
    mask = np.asarray(boardable, dtype=bool)
    n, t = len(edges), len(gaps)
    if (
        not 1 <= t <= 64
        or not 2 <= n <= 256
        or mask.shape != (n,)
        or not mask.any()
        or not np.isfinite(gaps).all()
        or np.any(gaps < 0)
        or not np.isfinite(edges).all()
        or np.any(edges <= 0)
        or not 1 <= max_step <= 8
        or not scales
        or any(not np.isfinite(s) or s <= 0 for s in scales)
        or not np.isfinite(skip_penalty)
        or skip_penalty < 0
    ):
        raise ValueError("invalid or unbounded alignment inputs")
    states = np.arange(n)
    steps = np.arange(1, max_step + 1)
    sources = (states[None, :] - steps[:, None]) % n
    solutions = []
    phase_costs = np.full(n, np.inf)
    for scale in scales:
        cumulative = np.zeros((max_step, n))
        for k in range(max_step):
            cumulative[k] = sum(np.roll(edges, -j) for j in range(k + 1)) * scale
        score = np.full((n, n), np.inf)
        score[states[mask], states[mask]] = 0
        back = np.zeros((t, n, n), dtype=np.int8)
        for i, delta in enumerate(gaps):
            cost = abs(delta - cumulative[np.arange(max_step)[:, None], sources])
            cost += skip_penalty * (steps[:, None] - 1)
            choices = np.stack([score[:, sources[k]] + cost[k] for k in range(max_step)])
            back[i] = np.argmin(choices, axis=0) + 1
            score = np.min(choices, axis=0)
            score[:, ~mask] = np.inf
        ends = np.argmin(score, axis=1)
        values = score[states, ends]
        phase_costs = np.minimum(phase_costs, values)
        start = int(np.argmin(values))
        if not np.isfinite(values[start]):
            continue
        path = [int(ends[start])]
        selected_steps = []
        for i in range(t - 1, -1, -1):
            k = int(back[i, start, path[-1]])
            selected_steps.append(k)
            path.append((path[-1] - k) % n)
        path.reverse()
        selected_steps.reverse()
        predicted = [float(cumulative[k - 1, path[i]]) for i, k in enumerate(selected_steps)]
        solutions.append(
            {
                "cost_seconds": float(values[start]),
                "scale": scale,
                "path": path,
                "steps": selected_steps,
                "predicted_seconds": predicted,
                "residual_seconds": (gaps - predicted).tolist(),
            }
        )
    if not solutions:
        raise ValueError("no feasible path")
    best = min(solutions, key=lambda s: (s["cost_seconds"], s["scale"], s["path"][0]))
    ordered = np.sort(phase_costs[np.isfinite(phase_costs)])
    residual = np.abs(best["residual_seconds"])
    best.update(
        mae_seconds=float(residual.mean()),
        within_30=float(np.mean(residual <= 30)),
        maximum_residual_seconds=float(residual.max()),
        competing_phases_within_10s=int(np.sum(phase_costs <= ordered[0] + 10 * t)),
        phase_margin_seconds_per_gap=float((ordered[1] - ordered[0]) / t)
        if len(ordered) > 1
        else None,
        skipped_visits=sum(k - 1 for k in best["steps"]),
        phase_cost_seconds_per_gap=[float(v / t) if np.isfinite(v) else None for v in phase_costs],
    )
    return best


def prefix_prediction(gaps: Any, edges: Any, boardable: Any, prefix: int = 6) -> dict[str, Any]:
    """Freeze prefix-selected phase/scale; predict next-stop durations before seeing holdout."""
    gaps = np.asarray(gaps, dtype=float)
    if not 1 <= prefix < len(gaps):
        raise ValueError("prefix must leave held-out intervals")
    fit = align(gaps[:prefix], edges, boardable, max_step=1)
    start = fit["path"][-1]
    prediction = []
    current = start
    for _ in range(len(gaps) - prefix):
        seconds = float(edges[current])
        current = (current + 1) % len(edges)
        while not boardable[current]:
            seconds += float(edges[current])
            current = (current + 1) % len(edges)
        prediction.append(seconds * fit["scale"])
    residual = abs(gaps[prefix:] - prediction)
    return {
        "fit_mae_seconds": fit["mae_seconds"],
        "holdout_mae_seconds": float(residual.mean()),
        "holdout_within_30": float(np.mean(residual <= 30)),
        "predicted_seconds": prediction,
        "scale": fit["scale"],
        "phase": fit["path"][0],
    }


def diagnose(times: Any, edges: Any, boardable: Any, seed: int = 20260926) -> dict[str, Any]:
    times = np.asarray(times, dtype=float)
    gaps = np.diff(times)
    fit = align(gaps, edges, boardable)
    direct = align(gaps, edges, boardable, max_step=1)
    prefix = prefix_prediction(gaps, edges, boardable)
    rng = np.random.default_rng(seed)
    null = []
    null_holdout = []
    for _ in range(9):
        shuffled = rng.permutation(edges)
        null.append(align(gaps, shuffled, boardable)["mae_seconds"])
        null_holdout.append(prefix_prediction(gaps, shuffled, boardable)["holdout_mae_seconds"])
    keep = np.array([0, 1, 3, 5, 7, 9, len(times) - 1])
    keep = np.unique(keep[keep < len(times)])
    sparse = align(np.diff(times[keep]), edges, boardable, max_step=6)
    retained_agreement = float(np.mean(np.array(fit["path"])[keep] == sparse["path"]))
    return {
        "fit": fit,
        "consecutive": direct,
        "prefix": prefix,
        "shuffled_mae_seconds": null,
        "shuffled_median_mae_seconds": float(np.median(null)),
        "shuffled_prefix_holdout_median": float(np.median(null_holdout)),
        "beats_all_9_shuffles": fit["mae_seconds"] < min(null) - 1e-9,
        "beats_shuffled_median_by_20pct": bool(fit["mae_seconds"] < 0.8 * np.median(null)),
        "retained_positions_after_masking": retained_agreement,
        "masked_fit_mae_seconds": sparse["mae_seconds"],
        "strong_candidate": bool(
            fit["competing_phases_within_10s"] == 1
            and fit["mae_seconds"] <= 30
            and fit["mae_seconds"] < 0.8 * np.median(null)
            and prefix["holdout_mae_seconds"] <= 45
            and retained_agreement >= 0.8
        ),
    }

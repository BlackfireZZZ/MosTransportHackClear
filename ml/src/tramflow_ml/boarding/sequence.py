"""Conditional cycle alignment; posterior mass is model support, never stop accuracy."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CycleTemplate:
    """Ordered cyclic visits; edge i leaves visit i, including terminal connectors."""

    route: str
    pattern_ids: tuple[str, ...]
    stop_ids: tuple[str, ...]
    directions: tuple[str, ...]
    edge_lengths_m: tuple[float, ...]
    terminal_after: tuple[bool, ...]
    provenance: str


@dataclass(frozen=True)
class SequenceConfig:
    speed_mps: float = 5.0
    dwell_seconds: float = 20.0
    terminal_seconds: float = 120.0
    relative_sigma: float = 0.35
    noise_seconds: float = 25.0
    max_steps: int = 128
    reset_gap_seconds: float = 1800.0
    outlier_sigma: float = 8.0
    max_cells: int = 2_000_000
    max_operations: int = 100_000_000


DEFAULT_CONFIG = SequenceConfig()


@dataclass(frozen=True)
class SequenceResult:
    """All probabilities condition on the template, timing scenario and step bound.

    Transition arrays have T entries: entry 0 and resets use step=-1, other
    numeric diagnostics=0. expected_cycles counts cycle-boundary crossings;
    expected_skipped counts unobserved visits between observations. Neither
    quantity claims actual service. log_score is the conditional path potential,
    without Gaussian constants, and is not comparable across reset patterns.
    """

    path_states: IntArray
    posterior: FloatArray
    top_states: IntArray
    top_probabilities: FloatArray
    path_steps: IntArray
    expected_seconds: FloatArray
    residual_seconds: FloatArray
    expected_skipped: FloatArray
    expected_cycles: FloatArray
    log_score: float
    reset_indices: tuple[int, ...]
    flags: tuple[str, ...]
    operations: int
    calibrated: bool = False


def _validate(template: CycleTemplate, config: SequenceConfig, times: FloatArray) -> None:
    n = len(template.stop_ids)
    if n < 2 or not template.route or not template.provenance:
        raise ValueError("cycle needs at least two visits, route and provenance")
    if any(
        len(v) != n
        for v in (
            template.pattern_ids,
            template.directions,
            template.edge_lengths_m,
            template.terminal_after,
        )
    ):
        raise ValueError("cycle columns must have equal lengths")
    if any(not v for v in (*template.stop_ids, *template.pattern_ids, *template.directions)):
        raise ValueError("visit identity cannot be empty")
    if any(type(v) is not bool for v in template.terminal_after):
        raise ValueError("terminal flags must be boolean")
    if any(not isfinite(v) or v < 0 for v in template.edge_lengths_m):
        raise ValueError("edge lengths must be finite and nonnegative")
    if not any(v > 0 for v in template.edge_lengths_m):
        raise ValueError("cycle must have a positive length")
    if any(
        not isfinite(v) or v <= 0
        for v in (
            config.speed_mps,
            config.noise_seconds,
            config.reset_gap_seconds,
            config.outlier_sigma,
        )
    ):
        raise ValueError("positive finite timing parameters required")
    if any(
        not isfinite(v) or v < 0
        for v in (
            config.dwell_seconds,
            config.terminal_seconds,
            config.relative_sigma,
        )
    ):
        raise ValueError("nonnegative finite timing parameters required")
    if any(
        type(v) is not int or v <= 0
        for v in (
            config.max_steps,
            config.max_cells,
            config.max_operations,
        )
    ):
        raise ValueError("positive integer budgets and step bound required")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("times must be a nonempty finite one-dimensional sequence")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted absolute seconds, without midnight wrapping")


def _lse(values: FloatArray, axis: int = 0) -> FloatArray:
    maximum = np.max(values, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0)
    with np.errstate(divide="ignore"):
        return np.asarray(
            np.squeeze(safe_maximum, axis=axis)
            + np.log(np.exp(values - safe_maximum).sum(axis=axis)),
            dtype=np.float64,
        )


def infer_sequence(
    times: Sequence[float] | FloatArray,
    template: CycleTemplate,
    config: SequenceConfig = DEFAULT_CONFIG,
    *,
    log_emissions: FloatArray | None = None,
    edge_seconds: FloatArray | None = None,
) -> SequenceResult:
    """Exact sum/max-product over bounded forward transitions with a uniform start.

    Pairwise potential for a forward step count k is Gaussian-shaped residual
    support divided by timing sigma and by max_steps+1. All k in [0,max_steps]
    participate, including repeated cycle traversals; top-two is display only.
    Long gaps and globally impossible residuals start independent blocks.
    Complexity is O(T*N*max_steps); exhausted budgets raise before allocation.
    Optional edge_seconds overrides complete local travel/standing durations.
    Optional external log evidence has shape (T,N), is nonpositive, and may use
    -inf for impossible states. Its source/calibration belongs to the caller.
    """
    timestamps = np.asarray(times, dtype=np.float64)
    _validate(template, config, timestamps)
    t, n = len(timestamps), len(template.stop_ids)
    k = config.max_steps + 1
    operations = 2 * max(t - 1, 1) * n * k
    if t * n > config.max_cells or n * k > config.max_cells:
        raise ValueError("sequence cell budget exhausted")
    if operations > config.max_operations:
        raise ValueError("sequence operation budget exhausted")
    emissions = np.zeros((t, n)) if log_emissions is None else np.asarray(log_emissions)
    if emissions.shape != (t, n):
        raise ValueError("log emissions must have shape (observations, visits)")
    if np.any(np.isnan(emissions)) or np.any(emissions > 0):
        raise ValueError("log emissions must be nonpositive or negative infinity")
    if not np.all(np.any(np.isfinite(emissions), axis=1)):
        raise ValueError("each emission row must have a feasible state")
    edge_times = (
        np.asarray(template.edge_lengths_m) / config.speed_mps
        + config.dwell_seconds
        + np.asarray(template.terminal_after) * config.terminal_seconds
    )
    if edge_seconds is not None:
        edge_times = np.asarray(edge_seconds, dtype=np.float64)
        if edge_times.shape != (n,) or not np.isfinite(edge_times).all() or np.any(edge_times <= 0):
            raise ValueError("edge seconds must be a positive finite vector matching visits")
    states = np.arange(n, dtype=np.int64)
    steps = np.arange(k, dtype=np.int64)[:, None]
    targets = (states + steps) % n
    sources = (states - steps) % n
    mean = np.zeros((k, n), dtype=np.float64)
    for advance in range(1, k):
        mean[advance] = mean[advance - 1] + edge_times[(states + advance - 1) % n]
    sigma = np.hypot(config.noise_seconds, config.relative_sigma * mean)
    if not np.isfinite(mean).all() or not np.isfinite(sigma).all():
        raise ValueError("timing scenario overflows")
    prior = log(k)

    def weights(delta: float) -> tuple[FloatArray, bool]:
        residual = (delta - mean) / sigma
        reset = delta > config.reset_gap_seconds or np.min(np.abs(residual)) > config.outlier_sigma
        if reset:
            return np.zeros_like(mean), True
        return np.asarray(-0.5 * residual**2 - np.log(sigma) - prior), False

    alpha = np.empty((t, n), dtype=np.float64)
    alpha[0] = -log(n) + emissions[0]
    best = alpha[0].copy()
    alpha[0] -= float(_lse(alpha[0]))
    back_steps = np.zeros((t, n), dtype=np.int64)
    reset_at = np.zeros(t, dtype=np.bool_)
    differences = np.diff(timestamps)
    score = 0.0
    for i, delta in enumerate(differences, start=1):
        weight, reset = weights(float(delta))
        reset_at[i] = reset
        if reset:
            alpha[i] = -log(n) + emissions[i]
            alpha[i] -= float(_lse(alpha[i]))
            back_steps[i] = int(np.argmax(best))
            score += float(best.max())
            best = -log(n) + emissions[i]
            continue
        incoming = weight[steps, sources] + emissions[i][None, :]
        alpha[i] = _lse(alpha[i - 1][sources] + incoming)
        normalizer = float(_lse(alpha[i]))
        if not isfinite(normalizer):
            raise ValueError("external evidence has no feasible bounded forward path")
        alpha[i] -= normalizer
        choices = best[sources] + incoming
        back_steps[i] = np.argmax(choices, axis=0)
        best = np.max(choices, axis=0)
        offset = float(best.max())
        score += offset
        best -= offset

    path = np.empty(t, dtype=np.int64)
    path[-1] = np.argmax(best)
    path_steps = np.full(t, -1, dtype=np.int64)
    expected_seconds = np.zeros(t, dtype=np.float64)
    residual_seconds = np.zeros(t, dtype=np.float64)
    for i in range(t - 1, 0, -1):
        value = back_steps[i, path[i]]
        if reset_at[i]:
            path[i - 1] = value
        else:
            path_steps[i] = value
            path[i - 1] = (path[i] - value) % n
            expected_seconds[i] = mean[value, path[i - 1]]
            residual_seconds[i] = differences[i - 1] - expected_seconds[i]
    score += float(best.max())

    posterior = np.empty_like(alpha)
    beta = np.zeros(n, dtype=np.float64)
    expected_skipped = np.zeros(t, dtype=np.float64)
    expected_cycles = np.zeros(t, dtype=np.float64)
    bound_mass = 0.0
    for i in range(t - 1, -1, -1):
        joint = alpha[i] + beta
        posterior[i] = np.exp(joint - float(_lse(joint)))
        if i == 0:
            break
        if reset_at[i]:
            beta.fill(0)
            continue
        weight, _ = weights(float(differences[i - 1]))
        following = emissions[i][targets] + beta[targets]
        edge_joint = alpha[i - 1][None, :] + weight + following
        edge_mass = np.exp(edge_joint - float(_lse(edge_joint.ravel())))
        bound_mass = max(bound_mass, float(edge_mass[-1].sum()))
        expected_skipped[i] = np.sum(edge_mass * np.maximum(steps - 1, 0))
        expected_cycles[i] = np.sum(edge_mass * ((states + steps) // n))
        beta = _lse(weight + following)
        beta -= float(_lse(beta))
    top = np.argsort(-posterior, axis=1, kind="stable")[:, :2]
    flags = ["inferred_uncertified", "conditional_uncalibrated", "bounded_steps", "open_endpoints"]
    if reset_at.any():
        flags.append("gap_or_outlier_reset")
    if bound_mass > 0.05:
        flags.append("step_bound_sensitive")
    if log_emissions is not None:
        flags.append("external_log_evidence")
    if np.any(np.max(posterior, axis=1) <= 0.5 + 1e-10):
        flags.append("ambiguous_states")
    return SequenceResult(
        path,
        posterior,
        top,
        np.take_along_axis(posterior, top, axis=1),
        path_steps,
        expected_seconds,
        residual_seconds,
        expected_skipped,
        expected_cycles,
        score,
        tuple(int(v) for v in np.flatnonzero(reset_at)),
        tuple(flags),
        operations,
    )

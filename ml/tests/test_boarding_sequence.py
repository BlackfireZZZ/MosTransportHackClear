from dataclasses import replace
from itertools import product
from math import exp, log, sqrt

import numpy as np
import pytest

from tramflow_ml.boarding.sequence import CycleTemplate, SequenceConfig, infer_sequence


def cycle(lengths: tuple[float, ...] = (10.0, 20.0, 30.0)) -> CycleTemplate:
    n = len(lengths)
    return CycleTemplate(
        "1",
        tuple("p" for _ in lengths),
        tuple(str(i) for i in range(n)),
        tuple("out" if i < n // 2 else "in" for i in range(n)),
        lengths,
        tuple(False for _ in lengths),
        "synthetic:test",
    )


def config(**kwargs: object) -> SequenceConfig:
    return replace(
        SequenceConfig(
            speed_mps=1,
            dwell_seconds=0,
            terminal_seconds=0,
            noise_seconds=2,
            relative_sigma=0,
            max_steps=4,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("use_emissions", [False, True])
def test_posterior_and_max_path_equal_independent_exhaustive_enumeration(
    use_emissions: bool,
) -> None:
    template = cycle()
    cfg = config()
    times = [0.0, 20.0, 50.0]
    emissions = np.array([[0, -1, -2], [-0.5, -np.inf, 0], [-2, 0, -1]])
    if not use_emissions:
        emissions.fill(0)
    actual = infer_sequence(times, template, cfg, log_emissions=emissions)
    masses = np.zeros_like(actual.posterior)
    skip = np.zeros(3)
    cycles = np.zeros(3)
    total = 0.0
    best_score = -float("inf")
    best_path = None
    for initial, first, second in product(range(3), range(5), range(5)):
        path = [initial]
        score = -log(3) + emissions[0, initial]
        for i, (delta, step) in enumerate(
            zip(np.diff(times), (first, second), strict=True),
            start=1,
        ):
            mean = sum(template.edge_lengths_m[(path[-1] + j) % 3] for j in range(step))
            sigma = sqrt(cfg.noise_seconds**2 + (cfg.relative_sigma * mean) ** 2)
            score += -0.5 * ((delta - mean) / sigma) ** 2 - log(sigma) - log(5)
            path.append((path[-1] + step) % 3)
            score += emissions[i, path[-1]]
        weight = exp(score)
        total += weight
        for i, state in enumerate(path):
            masses[i, state] += weight
        for i, step in enumerate((first, second), start=1):
            skip[i] += weight * max(step - 1, 0)
            cycles[i] += weight * ((path[i - 1] + step) // 3)
        if score > best_score:
            best_score, best_path = score, path
    np.testing.assert_allclose(actual.posterior, masses / total, atol=1e-14)
    np.testing.assert_allclose(actual.expected_skipped, skip / total, atol=1e-14)
    np.testing.assert_allclose(actual.expected_cycles, cycles / total, atol=1e-14)
    assert actual.path_states.tolist() == best_path
    assert actual.log_score == pytest.approx(best_score)


def test_cycle_skip_many_to_one_and_terminal_timing() -> None:
    template = replace(cycle(), terminal_after=(False, False, True))
    actual = infer_sequence([0, 0, 20, 60, 90], template, config(terminal_seconds=10))
    assert actual.path_states.tolist() == [1, 1, 2, 0, 2]
    assert actual.path_steps.tolist() == [-1, 0, 1, 1, 2]
    assert actual.expected_seconds.tolist() == [0, 0, 20, 40, 30]
    assert actual.expected_skipped[-1] > 0.99
    assert actual.expected_cycles[3] > 0.99


def test_identical_geometry_cannot_invent_start_or_direction_certainty() -> None:
    actual = infer_sequence([0, 10, 20, 40, 70], cycle((10, 10, 10, 10)), config())
    np.testing.assert_allclose(actual.posterior, 0.25, atol=1e-14)
    np.testing.assert_allclose(actual.top_probabilities, 0.25, atol=1e-14)
    assert "ambiguous_states" in actual.flags
    assert "conditional_uncalibrated" in actual.flags
    assert not actual.calibrated
    assert len(actual.path_states) == 5


def test_top_two_never_renormalized_or_used_as_search_beam() -> None:
    actual = infer_sequence([0], cycle(), config())
    assert actual.top_probabilities.sum() == pytest.approx(2 / 3)
    assert actual.posterior.sum() == pytest.approx(1)


@pytest.mark.parametrize("gap", [500, 5000])
def test_large_gaps_and_impossible_travel_reset_to_independent_blocks(gap: int) -> None:
    template, cfg = cycle(), config()
    actual = infer_sequence([0, 20, gap, gap + 30], template, cfg)
    left = infer_sequence([0, 20], template, cfg)
    right = infer_sequence([gap, gap + 30], template, cfg)
    assert actual.reset_indices == (2,)
    assert actual.path_steps[2] == -1
    assert actual.expected_cycles[2] == 0
    np.testing.assert_allclose(actual.posterior[:2], left.posterior)
    np.testing.assert_allclose(actual.posterior[2:], right.posterior)
    assert actual.log_score == pytest.approx(left.log_score + right.log_score)


def test_midnight_and_epoch_shift_preserve_inference_and_input() -> None:
    times = np.array([86395.0, 86415.0, 86445.0])
    original = times.copy()
    actual = infer_sequence(times, cycle(), config())
    shifted = infer_sequence(times + 1_750_000_000, cycle(), config())
    np.testing.assert_array_equal(times, original)
    np.testing.assert_array_equal(actual.path_states, shifted.path_states)
    np.testing.assert_allclose(actual.posterior, shifted.posterior)


@pytest.mark.parametrize("times", [[], [0, float("nan")], [0, float("inf")], [10, 0], [[0]]])
def test_invalid_times_rejected(times: list[object]) -> None:
    with pytest.raises(ValueError):
        infer_sequence(times, cycle(), config())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "cfg",
    [
        config(max_cells=1),
        config(max_operations=1),
        config(max_steps=0),
        config(speed_mps=0),
        config(relative_sigma=-1),
        config(noise_seconds=float("nan")),
        config(reset_gap_seconds=float("inf")),
        config(max_steps=1.5),
    ],
)
def test_invalid_config_or_budget_fails_explicitly(cfg: SequenceConfig) -> None:
    with pytest.raises(ValueError):
        infer_sequence([0, 10], cycle(), cfg)


@pytest.mark.parametrize(
    "template",
    [
        replace(cycle(), provenance=""),
        replace(cycle(), directions=("out",)),
        replace(cycle(), edge_lengths_m=(1, -2, 3)),
        replace(cycle(), edge_lengths_m=(1, float("nan"), 3)),
        replace(cycle(), edge_lengths_m=(0, 0, 0)),
        replace(cycle(), stop_ids=("a", "", "c")),
    ],
)
def test_invalid_templates_rejected(template: CycleTemplate) -> None:
    with pytest.raises(ValueError):
        infer_sequence([0, 10], template, config())


def test_long_sequence_has_finite_normalized_mass_with_bounded_work() -> None:
    times = np.cumsum(np.resize([10, 20, 30], 2000)).astype(float)
    actual = infer_sequence(times, cycle(), config())
    assert np.isfinite(actual.posterior).all()
    assert np.isfinite(actual.log_score)
    np.testing.assert_allclose(actual.posterior.sum(axis=1), 1, atol=1e-12)
    assert actual.operations == 2 * 1999 * 3 * 5


def test_external_emission_single_row_matches_analytic_posterior() -> None:
    emissions = np.log(np.array([[0.2, 0.3, 0.5]]))
    actual = infer_sequence([0], cycle(), config(), log_emissions=emissions)
    np.testing.assert_allclose(actual.posterior, [[0.2, 0.3, 0.5]])
    assert actual.path_states.tolist() == [2]
    assert actual.log_score == pytest.approx(log(0.5 / 3))


def test_boardable_emission_mask_applies_to_path_smoothing_and_resets() -> None:
    emissions = np.array(
        [[0, -np.inf, -np.inf], [-np.inf, 0, -np.inf], [-np.inf, -np.inf, 0], [0, -np.inf, -np.inf]]
    )
    actual = infer_sequence([0, 10, 5000, 5030], cycle(), config(), log_emissions=emissions)
    assert actual.path_states.tolist() == [0, 1, 2, 0]
    np.testing.assert_allclose(actual.posterior, np.isfinite(emissions).astype(float))
    assert actual.reset_indices == (2,)


def test_future_external_evidence_updates_earlier_marginal() -> None:
    emissions = np.array([[0, 0, 0], [-np.inf, -np.inf, 0]])
    actual = infer_sequence([0, 20], cycle(), config(), log_emissions=emissions)
    assert actual.posterior[0, 1] > 0.99
    assert actual.posterior[1, 2] == 1
    assert actual.path_states.tolist() == [1, 2]


@pytest.mark.parametrize(
    "emissions",
    [
        np.zeros((1, 2)),
        np.array([[0, 1, 0]]),
        np.array([[0, np.nan, 0]]),
        np.array([[-np.inf, -np.inf, -np.inf]]),
        np.array([[0, np.inf, 0]]),
    ],
)
def test_invalid_external_evidence_fails_explicitly(emissions: np.ndarray) -> None:
    with pytest.raises(ValueError):
        infer_sequence([0], cycle(), config(), log_emissions=emissions)


def test_disconnected_emission_support_fails_instead_of_fabricating_mass() -> None:
    emissions = np.array([[0, -np.inf, -np.inf], [-np.inf, -np.inf, 0]])
    with pytest.raises(ValueError, match="no feasible"):
        infer_sequence([0, 20], cycle(), config(max_steps=1), log_emissions=emissions)


def test_step_bound_saturation_is_visible() -> None:
    actual = infer_sequence([0, 30], cycle((10, 10, 10)), config(max_steps=3))
    assert "step_bound_sensitive" in actual.flags

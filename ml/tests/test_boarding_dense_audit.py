import numpy as np
import pytest

from tramflow_ml.boarding.burst_detection import METHODS, detect, match_onsets, randomize_local_rate
from tramflow_ml.boarding.dense_audit import align, bursts, prefix_prediction


def test_span_cap_can_create_false_periodicity() -> None:
    times = np.arange(0, 601, 10)
    capped = bursts(times, 30, 90)
    natural = bursts(times, 30, None)
    assert len(natural["times"]) == 1
    assert capped["forced_boundaries"].all()
    assert np.all(np.diff(capped["times"]) == 100)


def test_unique_pattern_recovers_phase_and_skips() -> None:
    edges = np.array([60.0, 180, 300, 90, 240, 420, 120])
    fit = align([300, 330, 420, 120, 60], edges, np.ones(7, bool), scales=(1.0,))
    assert fit["path"] == [2, 3, 5, 6, 0, 1]
    assert fit["steps"] == [1, 2, 1, 1, 1]
    assert fit["mae_seconds"] == 0


def test_equal_edges_cannot_identify_phase() -> None:
    fit = align([60] * 11, [60] * 6, [True] * 6)
    assert fit["competing_phases_within_10s"] == 6
    assert fit["phase_margin_seconds_per_gap"] == 0


def test_prefix_does_not_read_future_intervals() -> None:
    edges = [60.0, 180, 300, 90, 240, 420, 120]
    a = prefix_prediction([60, 180, 300, 90, 240, 420, 120, 60, 180], edges, [True] * 7)
    b = prefix_prediction([60, 180, 300, 90, 240, 420, 500, 800, 300], edges, [True] * 7)
    assert a["phase"] == b["phase"]
    assert a["predicted_seconds"] == b["predicted_seconds"]
    assert a["holdout_mae_seconds"] == 0


def test_random_control_preserves_local_mass() -> None:
    times = np.arange(0, 1800, 7.0)
    changed = randomize_local_rate(times, 42)
    np.testing.assert_array_equal(np.sort((times // 300).astype(int)), (changed // 300).astype(int))


@pytest.mark.parametrize("method", METHODS)
def test_detectors_do_not_duplicate_events(method: str) -> None:
    times = np.sort(np.r_[np.arange(0, 600, 10.0), [100, 100, 101, 102, 103, 300, 301, 302]])
    result = detect(times, method)
    assert result["covered_events"] <= len(times)
    assert np.all(np.diff(result["times"]) >= 0)


def test_onset_matching_is_one_to_one() -> None:
    result = match_onsets([100, 110], [105], 15)
    assert result["matched"] == 1
    assert result["recall"] == 0.5


def test_heldout_shape_uses_exposure_lengths() -> None:
    from tramflow_ml.boarding.burst_detection import heldout_shape

    result = heldout_shape([100], [80, 90, 101, 102, 103, 104, 135, 145], 0, 200)
    assert result["early_over_before_rate"] == 4
    assert result["early_over_late_rate"] == 4


def test_forbidden_states_are_json_serializable() -> None:
    import json

    result = align([60, 120], [60] * 4, [True, True, False, True])
    assert result["phase_cost_seconds_per_gap"][2] is None
    json.dumps(result, allow_nan=False)


def test_full_diagnostic_roundtrips_json() -> None:
    import json

    from tramflow_ml.boarding.dense_audit import diagnose

    result = diagnose(np.arange(12) * 60.0, [60.0, 90, 120, 60, 180, 120], [True] * 6)
    json.dumps(result, allow_nan=False)


def test_bounded_alignment_matches_exhaustive_search() -> None:
    from itertools import product

    edges = [40.0, 130, 80, 200, 60]
    gaps = [140.0, 270, 65]
    best = float("inf")
    for start, steps in product(range(5), product(range(1, 4), repeat=3)):
        state, cost = start, 0.0
        for delta, step in zip(gaps, steps, strict=True):
            expected = sum(edges[(state + k) % 5] for k in range(step))
            cost += abs(delta - expected) + 15 * (step - 1)
            state = (state + step) % 5
        best = min(best, cost)
    result = align(gaps, edges, [True] * 5, scales=(1.0,))
    assert result["cost_seconds"] == pytest.approx(best)

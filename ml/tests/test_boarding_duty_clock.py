from copy import deepcopy

import numpy as np
import pytest

from tramflow_ml.boarding.duty_clock import (
    calibrate,
    causal_onsets,
    clocks,
    direction_summary,
    directions,
    measure,
    nearest_errors,
    split_profiles,
)


def profile(direction, values):
    return {"direction": direction, "times": values, "profile_id": direction}


def test_onsets_are_causal_and_do_not_invent_first_window_event():
    prefix = [10, 30, 90, 91, 160]
    assert causal_onsets(prefix) == [90, 160]
    assert [x for x in causal_onsets(prefix + [170, 300, 400]) if x <= 160] == [90, 160]
    assert causal_onsets([]) == [] and causal_onsets([100]) == []
    assert causal_onsets([1, 1, 61]) == [61]


def test_nearest_errors_match_exhaustive_oracle():
    rng = np.random.default_rng(84)
    for _ in range(100):
        schedule = sorted(rng.integers(0, 10000, 50).astype(float).tolist())
        observed = rng.integers(-100, 12000, 100).astype(float).tolist()
        shift = 120
        expected = [min(abs(t - s - shift) for s in schedule) for t in observed]
        np.testing.assert_array_equal(nearest_errors(observed, schedule, shift), expected)


def test_known_offset_frozen_across_disjoint_whole_trips():
    early = [0, 1000, 1300, 2400, 3100, 3900]
    late = [20000, 20500, 21500, 22300]
    fit = calibrate([x + 120 for x in early], early)
    assert fit["shift"] == 120 and fit["loss"] == 0
    assert measure([x + 120 for x in late], late, fit["shift"])["capped_mae"] == 0
    changed_late = [x - 900 for x in late]
    assert calibrate([x + 120 for x in early], early) == fit
    assert measure(changed_late, late, fit["shift"])["capped_mae"] > 0


def test_whole_trip_guard_and_extended_service_clock():
    cutoff = 86400 + 12 * 3600
    p = [
        profile("0", [cutoff - 4000, cutoff - 1801]),
        profile("1", [cutoff + 1800, cutoff + 5000]),
        profile("0", [cutoff - 1800, cutoff + 100]),
    ]
    out = split_profiles(p, cutoff)
    assert out["early"] == [p[0]] and out["late"] == [p[1]] and out["crossing"] == 1
    assert all(t + 1800 < cutoff for t in out["early"][0]["times"])
    assert all(t - 1800 >= cutoff for t in out["late"][0]["times"])


def test_missing_evidence_is_not_zero_error():
    assert calibrate([], [1])["shift"] is None
    assert calibrate([1], [])["shift"] is None
    assert measure([], [], 0)["capped_mae"] is None
    assert measure([1, 2], [], 0)["capped_mae"] == 300


def test_direction_overlap_and_opposite_clock_ties_abstain():
    profiles = [profile("0", [100, 200, 300]), profile("1", [300, 400, 500])]
    assert directions([100, 200, 250, 300, 400, 700], profiles, 0) == [0, 0, -1, -1, 1, -1]
    overlapping = [profile("0", [0, 100, 200]), profile("1", [50, 150, 250])]
    assert directions([100], overlapping, 0) == [-1]


def test_near_optimal_shift_uncertainty_is_not_dropped():
    profiles = [profile("0", [0, 60, 120]), profile("1", [180, 240, 300])]
    result = direction_summary([60, 240], profiles, {"shift": 0, "near_shifts": [0, 180]})
    assert result["assigned"] == 2 and result["stable"] == 0


def test_translation_and_profile_order_invariance():
    p = [profile("0", [0, 120, 200]), profile("1", [400, 600, 700])]
    shifted = deepcopy(p)
    for row in shifted:
        row["times"] = [t + 86400 for t in row["times"]]
    assert directions([120, 600], p, 0) == directions([86520, 87000], shifted[::-1], 0)
    assert clocks(p) == clocks(p[::-1])


@pytest.mark.parametrize("times", [[2, 1], [0, float("nan")]])
def test_bad_observation_clocks_fail(times):
    with pytest.raises(ValueError):
        causal_onsets(times)


def test_periodic_aliases_remain_in_near_shift_set():
    fit = calibrate([3000, 3600, 4200], [float(x) for x in range(0, 8000, 600)])
    assert fit["shift"] == 0
    assert -600 in fit["near_shifts"] and 600 in fit["near_shifts"]


def test_guarded_shift_accepts_repeated_nonperiodic_early_signal():
    from tramflow_ml.boarding.duty_clock import guarded_calibration

    rng = np.random.default_rng(85)
    schedule = np.cumsum(rng.integers(100, 600, 60)).astype(float).tolist()
    observed = [t + 120 for t in schedule]
    out = guarded_calibration(observed, schedule, (observed[29] + observed[30]) / 2)
    assert out["accepted"] and out["shift"] == 120
    assert out["early_validation_gain"] >= 5


def test_guarded_shift_rejects_change_between_early_halves():
    from tramflow_ml.boarding.duty_clock import guarded_calibration

    rng = np.random.default_rng(86)
    schedule = np.cumsum(rng.integers(100, 600, 60)).astype(float).tolist()
    observed = [t + 120 for t in schedule[:30]] + [t - 300 for t in schedule[30:]]
    out = guarded_calibration(observed, schedule, (observed[29] + observed[30]) / 2)
    assert not out["accepted"] and out["shift"] == 0


def test_guarded_shift_no_evidence_and_aliases_keep_zero():
    from tramflow_ml.boarding.duty_clock import guarded_calibration

    assert guarded_calibration([], [], 100)["shift"] == 0
    clocks = [float(x) for x in range(0, 30000, 300)]
    out = guarded_calibration(clocks[10:80], clocks, 15000)
    assert out["reason"] == "ambiguous_early_shift" and out["shift"] == 0

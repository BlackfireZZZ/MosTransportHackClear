import pytest

from tramflow_ml.boarding.absolute_clock import (
    continue_profile,
    fit_profiles,
    reconstruct_profiles,
)


def profile(times, identity=0):
    return {"profile_id": identity, "times": times, "inferred": True}


def test_clock_breaks_identical_interval_alias():
    profiles = [profile([100, 160, 220]), profile([500, 560, 620], 1)]
    assert fit_profiles([510, 570, 630], profiles)["best"]["profile_id"] == 1
    baseline = fit_profiles(
        [510, 570, 630], profiles, clock_weight=0, start_prior=0, residual_weight=0
    )
    assert baseline["gap"] == 0


def test_terminal_prior_soft_not_absolute():
    profiles = [profile([100, 160, 220])]
    assert fit_profiles([150, 210], profiles, start_prior=60)["best"]["start_is_terminal"]
    result = fit_profiles([160, 220], profiles, start_prior=10)["best"]
    assert result["stop_indices"] == [1, 2]
    assert result["initial_offset"] == 0


def test_midnight_profile_extends_without_negative_modulo():
    profiles = reconstruct_profiles([[86340], [0], [60]], [60, 60], tolerance=0)
    assert profiles[0]["times"] == [86340, 86400, 86460]
    best = fit_profiles([86350, 86410, 86470], profiles)["best"]
    assert best["local_residuals"] == [0, 0]
    assert best["initial_offset"] == 10
    with pytest.raises(ValueError):
        fit_profiles([86350, 10, 70], profiles)


def test_missing_stop_clocks_does_not_fabricate_trip():
    assert reconstruct_profiles([[100], [], [220]], [60, 60]) == []
    assert fit_profiles([100], [])["best"] is None


def test_reconstruction_bounded_reproducible_and_forward_only():
    departures = [[100, 500], [155, 160, 165, 560], [220, 225, 620]]
    a = reconstruct_profiles(departures, [60, 60], tolerance=10, beam_width=2)
    assert a == reconstruct_profiles(departures, [60, 60], tolerance=10, beam_width=2)
    assert len(a) <= 4
    assert all(p["inferred"] for p in a)
    assert reconstruct_profiles([[100], [90]], [60], tolerance=10) == []


def test_one_global_offset_never_reset_per_stop():
    best = fit_profiles([110, 180, 290], [profile([100, 160, 220])])["best"]
    assert best["initial_offset"] == 10
    assert best["predicted_times"] == [110, 170, 230]
    assert best["clock_residuals"] == [0, 10, 60]
    assert best["local_residuals"] == [10, 50]


def test_skipped_stop_allowed_and_penalized():
    best = fit_profiles([100, 220], [profile([100, 160, 220])], max_offset=0)["best"]
    assert best["stop_indices"] == [0, 2]
    assert best["steps"] == [2]
    assert best["score"] == 15
    limited = fit_profiles([100, 220], [profile([100, 160, 220])], max_offset=0, max_step=1)
    assert limited["best"]["local_residuals"] == [60]


def test_infeasible_sequence_is_explicit():
    result = fit_profiles([100, 160, 220], [profile([100, 160])])
    assert result["best"] is None
    assert result["feasible_count"] == 0
    assert fit_profiles([10000], [profile([100])])["candidate_count"] == 0


def test_prefix_locked_continuation_cannot_reselect_profile_or_offset():
    p = profile([100, 160, 220, 280])
    prefix = fit_profiles([110, 170], [p])["best"]
    future = continue_profile([235, 295], p, prefix)
    assert future["initial_offset"] == 10
    assert future["stop_indices"] == [1, 2, 3]
    assert future["clock_residuals"] == [0, 5, 5]
    with pytest.raises(ValueError):
        continue_profile([235], profile([100, 160, 220], 1), prefix)
    assert continue_profile([235, 295, 350], p, prefix) is None


@pytest.mark.parametrize("times", [[], [1, 1], [2, 1], [-1], [float("nan")]])
def test_invalid_observations(times):
    with pytest.raises(ValueError):
        fit_profiles(times, [profile([100, 160])])


@pytest.mark.parametrize(
    "kwargs",
    [{"max_step": 0}, {"max_offset": -1}, {"clock_weight": float("nan")}, {"alternatives": -1}],
)
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        fit_profiles([100], [profile([100])], **kwargs)


def test_invalid_profiles():
    with pytest.raises(ValueError):
        fit_profiles([100], [profile([100, 90])])
    with pytest.raises(ValueError):
        fit_profiles([100], [profile([100]), profile([110])])
    with pytest.raises(ValueError):
        reconstruct_profiles([[100], [160]], [0])


def test_missing_times_explicit_opt_in_retains_terminal_identity():
    profiles = reconstruct_profiles([[100], [], [220]], [60, 60], allow_missing=True)
    assert profiles[0]["times"] == [100, 160, 220]
    assert profiles[0]["estimated_stops"] == [1]
    assert reconstruct_profiles([[], [160]], [60], allow_missing=True) == []


def test_forced_terminal_is_separate_from_soft_prior():
    profiles = [profile([100, 160, 220])]
    assert fit_profiles([160, 220], profiles, start_prior=0)["best"]["stop_indices"] == [1, 2]
    forced = fit_profiles([160, 220], profiles, start_mode="terminal")
    assert forced["best"]["stop_indices"] == [0, 1]
    assert forced["candidate_count"] == 1
    assert fit_profiles([160, 220], profiles, max_offset=0, start_mode="terminal")["best"] is None
    with pytest.raises(ValueError):
        fit_profiles([160], profiles, start_mode="bad")


def test_returned_path_cannot_mutate_cache():
    profiles = [profile([100, 160, 220])]
    first = fit_profiles([100, 160], profiles)
    first["best"]["stop_indices"][0] = 99
    assert fit_profiles([100, 160], profiles)["best"]["stop_indices"] == [0, 1]


def test_changed_prefix_schedule_is_rejected():
    p = profile([100, 160, 220])
    prefix = fit_profiles([110, 170], [p])["best"]
    with pytest.raises(ValueError):
        continue_profile([230], profile([100, 161, 220]), prefix)


def test_equal_adjacent_profile_clocks_preserve_both_stop_identities():
    p = profile([100, 100, 160])
    best = fit_profiles([100, 102, 160], [p], start_mode="terminal")["best"]
    assert best["stop_indices"] == [0, 1, 2]
    assert best["expected_intervals"] == [0, 60]
    assert best["local_residuals"] == [2, -2]
    prefix = fit_profiles([100, 102], [p], start_mode="terminal")["best"]
    assert continue_profile([160], p, prefix)["stop_indices"] == [1, 2]

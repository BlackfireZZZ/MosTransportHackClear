import numpy as np
import pytest

from tramflow_ml.boarding.wave_merge import coalesce


def test_one_continuous_wave_has_one_representative_and_preserves_members() -> None:
    r = coalesce(np.arange(20), [0, 5, 12], [[0, 20]])
    assert r["times"] == [0]
    assert r["groups"][0]["members"] == [0, 5, 12]
    assert r["merged_candidates"] == 2


def test_nearby_distinct_supports_are_never_joined() -> None:
    r = coalesce(np.arange(20), [0, 5], [[0, 5], [5, 10]])
    assert r["times"] == [0, 5]


def test_quiet_gap_protects_two_close_six_person_waves_inside_broad_support() -> None:
    times = np.array([0, 0, 0, 1, 1, 1, 10, 10, 10, 11, 11, 11])
    r = coalesce(times, [0, 10], [[0, 20]])
    assert r["times"] == [0, 10]
    assert r["decisions"][0]["decision"] == "keep_quiet_gap"


def test_no_support_means_no_merge_and_no_transitive_cross_support_merge() -> None:
    assert coalesce([0, 1, 2], [0, 1], [])["times"] == [0, 1]
    r = coalesce(np.arange(25), [0, 8, 10, 19, 21], [[0, 10], [10, 20]])
    assert r["times"] == [0, 10, 21]


def test_shift_invariant_and_idempotent() -> None:
    r = coalesce(np.arange(20), [0, 5, 12], [[0, 20]])
    s = coalesce(np.arange(20) + 100, [100, 105, 112], [[100, 120]])
    assert np.allclose(np.array(s["times"]) - 100, r["times"])
    assert coalesce(np.arange(20), r["times"], [[0, 20]])["times"] == r["times"]


@pytest.mark.parametrize(
    "c,s", [([2, 1], []), ([0, 0], []), ([float("nan")], []), ([0], [[0, 5], [4, 7]]), ([30], [])]
)
def test_invalid_candidates_or_supports_rejected(c: list, s: list) -> None:
    with pytest.raises(ValueError):
        coalesce(np.arange(20), c, s)


def test_quiet_gap_overlapping_fractional_consensus_interval_is_not_missed() -> None:
    r = coalesce([0, 0, 1, 1, 10, 10, 11, 11], [6, 10], [[0, 20]])
    assert r["times"] == [6, 10]
    assert r["decisions"][0]["decision"] == "keep_quiet_gap"


@pytest.mark.parametrize("candidates", [[], [10], [0, 10]])
def test_first_observation_is_mandatory_and_idempotent(candidates):
    result = coalesce([0, 0, 1, 10], candidates, [], first_observation_anchor=True)
    assert result["times"][0] == 0
    assert result["times"].count(0) == 1
    assert result["anchor"]["route_stop_index"] is None
    assert result["anchor"]["trip_start_confirmed"] is False
    assert result["input_candidates"] + result["anchor"]["added"] == (
        len(result["times"]) + result["merged_candidates"]
    )
    again = coalesce([0, 0, 1, 10], result["times"], [], first_observation_anchor=True)
    assert again["times"] == result["times"]


def test_anchor_does_not_split_one_supported_wave():
    result = coalesce(range(20), [3, 8], [[0, 20]], first_observation_anchor=True)
    assert result["times"] == [0]
    assert result["groups"][0]["members"] == [0, 3, 8]


def test_anchor_preserves_close_wave_after_silence():
    result = coalesce(
        [0, 0, 0, 1, 1, 1, 10, 10, 10, 11, 11, 11],
        [10], [[0, 20]], first_observation_anchor=True,
    )
    assert result["times"] == [0, 10]


def test_anchor_is_actual_payment_not_window_origin():
    result = coalesce([103, 103, 104, 222], [222], [], first_observation_anchor=True)
    assert result["times"] == [103, 222]
    shifted = coalesce([1103, 1103, 1104, 1222], [1222], [], first_observation_anchor=True)
    assert shifted["times"] == [1103, 1222]

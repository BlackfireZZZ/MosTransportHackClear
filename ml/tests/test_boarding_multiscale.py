import numpy as np
import pytest

from tramflow_ml.boarding.multiscale import (
    FeatureMixture,
    consensus,
    controls,
    hdbscan_onsets,
    peaks,
    representation,
    validate,
)


def test_counts_include_simultaneous_devices_without_double_counting() -> None:
    r = representation([100, 100, 101, 103, 110], ["a", "b", "a", "c", "b"])
    assert r["counts"][0, :4].tolist() == [3, 3, 4, 4]
    assert r["devices"][0, :4].tolist() == [2, 2, 3, 3]
    assert np.isfinite(r["features"]).all()


@pytest.mark.parametrize("kind", ["uniform", "device_shift"])
def test_controls_preserve_device_bin_mass_and_integer_precision(kind: str) -> None:
    t = np.array([1, 2, 2, 20, 299, 301, 302, 501.0])
    d = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
    u, v = controls(t, d, 9, kind)
    assert sorted(zip(t // 300, d, strict=True)) == sorted(zip(u // 300, v, strict=True))
    assert np.all(u == np.floor(u))
    assert np.all(np.diff(u) >= 0)
    assert np.array_equal(u, controls(t, d, 9, kind)[0])


def test_shift_preserves_device_circular_gaps() -> None:
    t = np.array([1.0, 3, 40, 200])
    u, _ = controls(t, np.array(["a"] * 4), 10, "device_shift")
    assert sorted(np.diff(np.r_[t, t[0] + 300])) == sorted(np.diff(np.r_[u, u[0] + 300]))


def test_consensus_no_single_family_votes_or_transitive_chaining() -> None:
    assert len(consensus({"a": [0, 1, 2]}, 5, 2)) == 0
    assert len(consensus({"a": [0], "b": [4], "c": [8]}, 5, 3)) == 0
    assert consensus({"a": [0, 100], "b": [2, 103]}, 5, 2).tolist() == [1, 101.5]


def test_variable_support_suppression_has_no_fixed_fifteen_seconds() -> None:
    ids = peaks(np.array([0, 3, 10, 25]), np.array([3, 2, 4, 5]), np.array([2, 2, 20, 5]), 1)
    assert ids.tolist() == [0, 1, 3]


def test_hdbscan_handles_variable_pulse_density() -> None:
    t = np.r_[np.arange(6), 100 + np.arange(15) * 2, 250 + np.arange(8)]
    onsets = hdbscan_onsets(t, 3, False)
    assert len(onsets) >= 3
    assert all(any(abs(onsets - t0) <= 2) for t0 in [0, 100, 250])


def test_feature_fit_rejects_future_data() -> None:
    with pytest.raises(ValueError, match="Jan-Apr"):
        FeatureMixture().fit(np.ones((3, 14)), [True, False, True], ["2025-05-01"])


@pytest.mark.parametrize(
    "times,devices",
    [
        ([2, 1], ["a", "a"]),
        ([1, float("nan")], ["a", "a"]),
        ([0, 86401], ["a", "a"]),
        ([1], []),
        ([], []),
    ],
)
def test_invalid_inputs(times: list, devices: list) -> None:
    with pytest.raises(ValueError):
        validate(times, devices)


def test_representation_translation_invariant_and_device_relabelling_invariant() -> None:
    t = np.array([100, 101, 101, 150, 180, 200.0])
    a = representation(t, ["a", "b", "a", "b", "a", "b"])
    b = representation(t + 86400, ["c", "d", "c", "d", "c", "d"])
    for key in ["counts", "scan", "sync", "features", "devices"]:
        np.testing.assert_allclose(a[key], b[key])


def test_synthetic_queue_covers_background_and_interpulse_arrivals() -> None:
    from tramflow_ml.boarding.multiscale_experiment import queued_times

    t, d = queued_times([10, 10.1, 11, 12, 10], [0, 0, 0, 0, 1])
    assert t[d == 0].tolist() == [10, 11, 13, 14]
    assert t[d == 1].tolist() == [10]

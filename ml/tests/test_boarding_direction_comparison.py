"""Equal-clock opportunities and independence of externally fixed direction labels."""

from copy import deepcopy
from pathlib import Path
from runpy import run_path

import pytest

from tramflow_ml.boarding.direction_comparison import (
    compare_relative,
    relative_profiles,
    scheduled_direction,
)


def profile(direction, times, key=None):
    return {"direction": direction, "times": times, "profile_id": key or direction,
            "stop_ids": [str(i) for i in range(len(times))]}


def fixtures():
    times = [1000, 1060, 1180, 1360, 1600, 1900, 2260, 2680]
    other = [9000, 9420, 9780, 10080, 10320, 10500, 10620, 10680]
    return times, [profile("0", times), profile("1", other)]


def test_opposite_direction_available_despite_distant_clock():
    times, profiles = fixtures()
    out = compare_relative(times, profiles)
    assert out["0"]["candidate_starts"] == out["1"]["candidate_starts"] == 8
    assert out["0"]["full"]["mae"] == 0
    assert out["1"]["full"]["mae"] > 0


def test_translation_and_profile_order_invariance():
    times, profiles = fixtures()
    shifted = deepcopy(profiles)
    for p in shifted:
        p["times"] = [t+123456 for t in p["times"]]
    translated = compare_relative([t+9876 for t in times], shifted[::-1])
    assert compare_relative(times, profiles) == translated


def test_direction_label_swap():
    times, profiles = fixtures()
    swapped = [{**p, "direction": str(1-int(p["direction"]))} for p in profiles]
    a, b = compare_relative(times, profiles), compare_relative(times, swapped)
    for direction in ("0", "1"):
        assert a[direction]["full"] == b[str(1-int(direction))]["full"]


def test_suffix_never_changes_prefix():
    times, profiles = fixtures()
    a = compare_relative(times, profiles)
    b = compare_relative([*times[:4], 2000, 3000, 4000, 5000], profiles)
    assert [a[d]["prefix"] for d in ("0", "1")] == [b[d]["prefix"] for d in ("0", "1")]


def test_dedup_preserves_visits_and_source_ids():
    _, profiles = fixtures()
    same = {**profiles[0], "profile_id": "copy", "times": [t+100 for t in profiles[0]["times"]]}
    result = relative_profiles([profiles[0], same])
    assert len(result) == 1
    assert result[0]["source_profile_ids"] == ["0", "copy"]
    same["stop_ids"] = list(reversed(same["stop_ids"]))
    assert len(relative_profiles([profiles[0], same])) == 2


def test_tie_order_does_not_depend_on_direction_name():
    _, profiles = fixtures()
    candidates = [{**p, "direction": "0"} for p in profiles]
    renamed = [{**p, "direction": "1"} for p in candidates]
    a, b = relative_profiles(candidates), relative_profiles(renamed)
    assert [p["times"] for p in a] == [p["times"] for p in b]
    assert [p["profile_id"].split(":")[1] for p in a] == [
        p["profile_id"].split(":")[1] for p in b
    ]


def test_equal_prefix_paths_remain_invariant_when_labels_swap():
    times = [0, 60, 120, 180, 240, 360, 480, 600]
    other = [0, 60, 120, 180, 300, 480, 720, 1000]
    profiles = [profile(d, t, f"{d}/{i}") for d in ("0", "1")
                for i, t in enumerate([times, other])]
    swapped = [{**p, "direction": str(1-int(p["direction"]))} for p in profiles]
    a, b = compare_relative(times, profiles), compare_relative(times, swapped)
    for d in ("0", "1"):
        opposite = str(1-int(d))
        assert a[d]["prefix_tied"]
        assert a[d]["prefix"]["stop_indices"] == b[opposite]["prefix"]["stop_indices"]
        assert a[d]["suffix"] == b[opposite]["suffix"]


def test_single_date_cannot_produce_uncertainty_interval():
    script = Path(__file__).resolve().parents[2] / "scripts/audit_direction_pairs.py"
    interval = run_path(str(script))["cluster_interval"]([{"date": "2025-07-08"}], [10])
    assert interval["percentile95"] is None


def test_equal_directions_have_equal_loss():
    times, _ = fixtures()
    out = compare_relative(times, [profile(d, times) for d in ("0", "1")])
    assert out["0"]["full"] == out["1"]["full"]
    assert out["0"]["suffix"] == out["1"]["suffix"]


def test_no_fit_is_charged_and_no_clock_label_is_invented():
    times, profiles = fixtures()
    out = compare_relative(times, [profiles[0]])
    assert out["1"]["suffix"]["capped_loss"] == 900
    assert scheduled_direction(5000, profiles)["direction"] is None
    assert scheduled_direction(1000, [profiles[0], profiles[0]])["direction"] is None
    assert scheduled_direction(1000, profiles)["direction"] == "0"


@pytest.mark.parametrize("times", [[1, 1, 2, 3, 4], [1, 2, 3, 4, float("nan")]])
def test_invalid_onsets_rejected(times):
    with pytest.raises(ValueError):
        compare_relative(times, [])

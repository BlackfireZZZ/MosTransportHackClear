from copy import deepcopy
from itertools import combinations

import pytest

from tramflow_ml.boarding.direction_ambiguity import compare_ambiguity, prefix_states
from tramflow_ml.boarding.direction_comparison import compare_relative


def profile(direction, times, key=None):
    return {
        "direction": direction,
        "times": times,
        "profile_id": key or direction,
        "stop_ids": [str(i) for i in range(len(times))],
    }


def test_exact_states_match_exhaustive_paths():
    clocks = [0, 30, 60, 100, 130, 200, 260]
    observed = [10, 45, 100, 165]
    costs = {}
    for path in combinations(range(len(clocks)), len(observed)):
        if max(b - a for a, b in zip(path, path[1:], strict=False)) > 3:
            continue
        cost = sum(
            abs(observed[j + 1] - observed[j] - (clocks[b] - clocks[a])) + 15 * (b - a - 1)
            for j, (a, b) in enumerate(zip(path, path[1:], strict=False))
        )
        costs[path[-1]] = min(costs.get(path[-1], float("inf")), cost)
    actual = prefix_states(observed, [profile("0", clocks)], max_step=3)["0"]
    assert actual["cost"] == min(costs.values())
    assert {s["endpoint"] for s in actual["states"]} == {
        end for end, cost in costs.items() if cost == min(costs.values())
    }


def test_single_start_tied_endpoints_are_retained_before_failure():
    times = [0, 10, 25, 40]
    p = profile("0", [0, 10, 20, 30, 40])
    states = prefix_states(times[:3], [p], max_step=1)["0"]["states"]
    assert {s["endpoint"] for s in states} == {2, 3, 4}
    out = compare_ambiguity(times, [p], prefix_size=3, max_step=1)["0"]
    assert out["feasible_states"] == 2
    assert out["mean_capped_loss"] == pytest.approx((5 + 5 + 900) / 3)
    assert out["max_capped_loss"] == 900
    assert not out["all_feasible"] and out["any_feasible"]


def test_tied_endpoints_from_same_start_are_not_lost():
    p = profile("0", [0, 1000, 1010, 1040, 2000])
    out = prefix_states([0, 1000, 1032.5], [p], max_step=2)["0"]
    assert out["cost"] == 22.5
    assert {s["endpoint"] for s in out["states"]} == {2, 3}
    audit = compare_ambiguity([0, 1000, 1032.5, 1072.5, 2000], [p], prefix_size=3, max_step=2)["0"]
    assert audit["state_count"] == 2 and audit["feasible_states"] == 1


def test_suffix_changes_neither_states_nor_prefix_cost():
    profiles = [profile("0", list(range(0, 600, 60))), profile("1", list(range(0, 900, 90)))]
    a = compare_ambiguity([0, 60, 120, 180, 240, 300], profiles)
    b = compare_ambiguity([0, 60, 120, 180, 1000, 2000], profiles)
    for d in ("0", "1"):
        assert a[d]["prefix_cost"] == b[d]["prefix_cost"]
        assert [(s["profile_id"], s["endpoint"]) for s in a[d]["states"]] == [
            (s["profile_id"], s["endpoint"]) for s in b[d]["states"]
        ]


def test_translation_duplication_order_and_label_invariance():
    times = [0, 60, 120, 180, 240]
    profiles = [profile(d, [0, 60, 120, 180, 240, 300]) for d in ("0", "1")]
    a = compare_ambiguity(times, profiles)
    shifted = deepcopy(profiles)
    for p in shifted:
        p["times"] = [x + 9000 for x in p["times"]]
    shifted += [{**shifted[0], "profile_id": "duplicate"}]
    assert a == compare_ambiguity([x + 1000 for x in times], shifted[::-1])
    assert a["0"]["mean_capped_loss"] == a["1"]["mean_capped_loss"]


def test_unique_state_reproduces_deterministic_baseline():
    times = [0, 60, 180, 360, 600, 900]
    profiles = [profile("0", times)]
    a = compare_ambiguity(times, profiles, max_step=1)
    b = compare_relative(times, profiles, max_step=1)
    assert a["0"]["state_count"] == 1
    assert a["0"]["mean_capped_loss"] == b["0"]["suffix"]["capped_loss"]
    assert a["1"]["state_count"] == 0
    assert a["1"]["mean_capped_loss"] == 900


@pytest.mark.parametrize("times", [[0, 1, 1, 2, 3], [0, 1, 2, 3, float("nan")]])
def test_bad_observations_rejected(times):
    with pytest.raises(ValueError):
        compare_ambiguity(times, [])


def test_direction_relabeling_preserves_losses_and_state_endpoints():
    times = [0, 60, 120, 180, 250, 320]
    profiles = [
        profile("0", [0, 60, 120, 180, 240, 300, 360]),
        profile("1", [0, 90, 180, 270, 360, 450, 540]),
    ]
    original = compare_ambiguity(times, profiles)
    renamed = compare_ambiguity(
        times, [{**p, "direction": str(1 - int(p["direction"]))} for p in profiles]
    )
    for d in ("0", "1"):
        a, b = original[d], renamed[str(1 - int(d))]
        assert a["mean_capped_loss"] == b["mean_capped_loss"]
        assert a["prefix_cost"] == b["prefix_cost"]
        assert [r["endpoint"] for r in a["states"]] == [r["endpoint"] for r in b["states"]]

"""Direction identifiability, prefix causality and failure-denominator checks."""

import runpy
from pathlib import Path

import pandas as pd
import pytest

from tramflow_ml.boarding.direction_probe import compare_directions, duty_component, shuffle_gaps


def profile(direction: str, times: list[int]) -> dict:
    return {
        "profile_id": direction, "direction": direction, "times": times,
        "stop_names": [f"stop-{i}" for i in range(len(times))],
        "stop_ids": [str(i) for i in range(len(times))],
    }


def test_equal_directions_abstain():
    times = [1000, 1060, 1180, 1360, 1600, 1900]
    result = compare_directions(times, [profile(d, times) for d in ("0", "1")])
    assert result["winner"] is None
    assert result["margin"] == 0


def test_asymmetric_sequence_and_label_swap():
    times = [1000, 1060, 1180, 1360, 1600, 1900]
    reverse = [1000, 1300, 1540, 1720, 1840, 1900]
    a = compare_directions(times, [profile("0", times), profile("1", reverse)])
    b = compare_directions(times, [profile("1", times), profile("0", reverse)])
    assert a["winner"] == "0"
    assert b["winner"] == "1"
    assert a["fits"]["0"]["suffix_mae"] == 0
    assert a["chosen_suffix_advantage"] == b["chosen_suffix_advantage"]


def test_suffix_cannot_change_prefix_winner():
    times = [1000, 1060, 1180, 1360, 1600, 1900]
    profiles = [profile("0", times), profile("1", [1000, 1300, 1540, 1720, 1840, 1900])]
    a = compare_directions(times, profiles)
    b = compare_directions([*times[:4], 5000, 9000], profiles)
    assert a["winner"] == b["winner"]
    assert a["fits"]["0"]["prefix"] == b["fits"]["0"]["prefix"]


def test_failed_direction_keeps_failure_loss():
    times = [1000, 1060, 1180, 1360, 1600, 1900]
    result = compare_directions(times, [profile("0", times), profile("1", times[:4])])
    assert result["fits"]["1"]["continuation"] is None
    assert result["fits"]["1"]["capped_loss"] == 900
    assert result["suffix_bilateral"] is False


def test_shuffle_preserves_intervals_and_span():
    times = [1000, 1010, 1040, 1100, 1200]
    shuffled = shuffle_gaps(times, 4)
    assert shuffled[0] == times[0] and shuffled[-1] == times[-1]
    assert sorted(b-a for a, b in zip(shuffled, shuffled[1:], strict=False)) == [10, 30, 60, 100]
    assert shuffled == shuffle_gaps(times, 4)
    with pytest.raises(ValueError):
        shuffle_gaps([1, 1], 0)


@pytest.mark.parametrize("value,expected", [("375_3161086_18_210", "210"),
                                           ("375_18_210", None), ("x_1_2_3", None)])
def test_duty_shape_is_not_semantic_certification(value, expected):
    assert duty_component(value) == expected


def test_enrollment_preserves_unknown_denominator_and_boundary_ambiguity():
    script = Path(__file__).resolve().parents[2] / "scripts/audit_direction_evidence.py"
    enrollment = runpy.run_path(str(script))["clock_enrollment"]
    block = pd.DataFrame({"exit_key": ["known", "known", "missing"],
                          "second": [7*3600+100, 7*3600+200, 7*3600+300]})
    profiles = [
        {"trip_id": "1_2_1_3", "direction": "0", "times": [111600, 111800]},
        {"trip_id": "1_2_2_3", "direction": "1", "times": [111800, 112200]},
    ]
    result = enrollment(block, profiles, {"known": "3"})
    assert result["periods"]["AM"]["all_events"] == 3
    assert result["counts"]["unknown_duty"] == 1
    assert result["counts"]["unique_trip"] == 1
    assert result["counts"]["ambiguous_trip"] == 1


def test_aggregate_charges_abstention():
    script = Path(__file__).resolve().parents[2] / "scripts/audit_direction_evidence.py"
    namespace = runpy.run_path(str(script))
    times = [1000, 1060, 1180, 1360, 1600, 1900]
    tie = compare_directions(times, [profile(d, times) for d in ("0", "1")])
    summary = namespace["aggregate"]([{"fits": {m: tie for m in namespace["METHODS"]}}])
    assert summary["duty"]["decided"] == 0
    assert summary["duty"]["all_window_capped_loss"] == 900

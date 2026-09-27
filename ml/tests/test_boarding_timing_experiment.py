import pandas as pd
import pytest

from tramflow_ml.boarding.timing_experiment import compare_events, compare_flows


def events() -> pd.DataFrame:
    return pd.DataFrame({"event_key": ["a", "b", "c"], "route": ["1"] * 3,
                         "event_at": ["2025-01-01 10:00:00"] * 3,
                         "stop_id": ["x", None, "y"], "direction": ["0"] * 3,
                         "raw_stable": [False] * 3})


def test_candidate_null_transitions_are_not_hidden_by_agreement() -> None:
    before = events()
    after = events().assign(stop_id=[None, "x", "y"])
    result = compare_events(before, after)
    assert result["events"] == 3
    assert result["candidate_to_null"] == result["null_to_candidate"] == 1
    assert result["both_have_candidate"] == result["same_stop_direction"] == 1


@pytest.mark.parametrize("column,value", [("route", "7"), ("event_at", "2025-01-02 10:00:00")])
def test_changed_source_identity_is_rejected(column: str, value: str) -> None:
    changed = events()
    changed.loc[0, column] = value
    with pytest.raises(ValueError, match="source route or event time"):
        compare_events(events(), changed)


def test_flow_redistribution_has_correct_mass_and_detects_hour_leakage() -> None:
    before = pd.DataFrame({"route": ["1", "1"], "event_hour": ["2025-01-01 10"] * 2,
                           "stop_id": ["x", "y"], "direction": ["0", "0"],
                           "expected_count": [600_000., 400_000.]})
    after = before.assign(expected_count=[400_000., 600_000.])
    assert compare_flows(before, after)["mass_moved_fraction"] == pytest.approx(.2)
    after.loc[0, "expected_count"] -= 1
    with pytest.raises(ValueError, match="mass mismatch"):
        compare_flows(before, after)
    after = before.copy()
    after.loc[0, "event_hour"] = "2025-01-01 11"
    with pytest.raises(ValueError, match="route/hour"):
        compare_flows(before, after)

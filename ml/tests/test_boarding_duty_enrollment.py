from pathlib import Path
from runpy import run_path

import pandas as pd
import pytest

from tramflow_ml.boarding.duty_clock import calibrate


@pytest.fixture
def runner(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return run_path(str(root / "scripts/audit_duty_clock.py"))


def frame(early=True, late=True, key="v"):
    base = 86400
    times = (
        ([base + 6 * 3600 - 60] + [base + 6 * 3600 + i * 600 for i in range(24)]) if early else []
    )
    if late:
        times += [base + 14 * 3600, base + 15 * 3600]
    return pd.DataFrame(
        {
            "second": times,
            "event_key": [str(i) for i in range(len(times))],
            "vehicle_key": key,
            "exit_key": "exit",
            "identity_conflict": False,
        }
    )


def test_late_only_vehicle_does_not_change_enrollment_ids_or_fit(runner):
    source = frame()
    augmented = pd.concat([source, frame(early=False, key="a")])
    assert runner["early_vehicle_keys"](source) == runner["early_vehicle_keys"](augmented) == ["v"]
    allowed = {h: True for h in range(24)}
    early = runner["prepare_vehicle"](source, {"exit": "duty"}, allowed)
    changed = source.copy()
    changed.loc[changed.second >= 86400 + 14 * 3600, "second"] += 1000
    later = runner["prepare_vehicle"](changed, {"exit": "duty"}, allowed)
    assert early["reason"] == later["reason"] == "enrolled"
    assert early["early"] == later["early"]
    schedule = [t - 120 for t in early["early"]]
    assert calibrate(early["early"], schedule) == calibrate(later["early"], schedule)


def test_multiple_early_duties_reject_but_late_change_is_retained(runner):
    source = frame()
    allowed = {h: True for h in range(24)}
    bad = source.copy()
    bad.loc[2, "exit_key"] = "other"
    assert (
        runner["prepare_vehicle"](bad, {"exit": "duty"}, allowed)["reason"]
        == "multiple_early_duties"
    )
    late = source.copy()
    late.loc[late.second >= 86400 + 14 * 3600, "exit_key"] = "other"
    result = runner["prepare_vehicle"](late, {"exit": "duty"}, allowed)
    assert result["reason"] == "enrolled" and result["exit_key"] == "exit"
    assert "other" in result["onset_rows"].exit_key.to_list()


def test_no_late_observations_do_not_remove_enrolled_group(runner):
    result = runner["prepare_vehicle"](
        frame(late=False), {"exit": "duty"}, {h: True for h in range(24)}
    )
    assert result["reason"] == "enrolled"
    assert not (result["onset_rows"].second >= 86400 + 14 * 3600).any()


def test_early_policy_and_identity_fail_closed(runner):
    source = frame()
    allowed = {h: h != 7 for h in range(24)}
    assert (
        runner["prepare_vehicle"](source, {"exit": "duty"}, allowed)["reason"]
        == "early_service_exclusion"
    )
    source.loc[2, "identity_conflict"] = True
    assert (
        runner["prepare_vehicle"](source, {"exit": "duty"}, {h: True for h in range(24)})["reason"]
        == "early_identity_conflict"
    )


def test_complete_group_late_perturbation_cannot_change_clock_or_guard(runner):
    base = 86400
    early1 = [base + 6 * 3600 + 100 + i * 127 for i in range(16)]
    early2 = [base + 8 * 3600 + 100 + i * 149 for i in range(16)]
    later = [base + 14 * 3600 + 100 + i * 137 for i in range(20)]
    profiles = [
        {"times": t, "direction": str(i % 2), "profile_id": str(i)}
        for i, t in enumerate([early1, early2, later])
    ]
    early = [t + 120 for t in early1 + early2]
    first = runner["run_group"](
        ({"group_id": "synthetic"}, early, [t + 120 for t in later], profiles, profiles)
    )
    changed = runner["run_group"](
        ({"group_id": "synthetic"}, early, [t + 900 for t in later], profiles, profiles)
    )
    assert first["fit"] == changed["fit"]
    assert first["guard"] == changed["guard"]
    assert first["wrong_fit"] == changed["wrong_fit"]
    assert first["methods"]["fitted"]["capped_mae"] != changed["methods"]["fitted"]["capped_mae"]

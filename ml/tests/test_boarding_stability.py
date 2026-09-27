import json
from dataclasses import asdict

import pytest

from tramflow_ml.boarding.bursts import Payment
from tramflow_ml.boarding.stability import evaluate_stability


def payments() -> tuple[Payment, ...]:
    return tuple(
        Payment(f"e{i}", second, device, "vehicle", "1", "exit", i % 3 != 0)
        for i, (second, device) in enumerate(
            [
                (0, "a"),
                (29, "a"),
                (58, "a"),
                (120, "a"),
                (140, "a"),
                (1, "b"),
                (10, "b"),
                (200, "b"),
            ]
        )
    )


def test_deterministic_order_independent_and_no_mutation() -> None:
    events = payments()
    before = [asdict(event) for event in events]
    report = evaluate_stability(events, 42)
    assert report == evaluate_stability(tuple(reversed(events)), 42)
    assert [asdict(event) for event in events] == before
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["geographic_accuracy"] == "unverified"
    assert report["fit_performed"] is False
    assert report["calibrated"] is False
    assert len(report["cases"]) == 9


def test_mass_denominators_shift_invariance_and_explicit_drops() -> None:
    report = evaluate_stability(payments(), 42)
    cases = {case["perturbation"]: case["metrics"] for case in report["cases"]}
    for metrics in cases.values():
        assert metrics["retained_events"] + metrics["dropped_events"] == 8
        assert metrics["retained_successful"] + metrics["dropped_successful"] == 5
    for name in ("identity", "shift_one_device_clock", "global_start_shift"):
        assert cases[name]["adjacency_retention"] == 1
        assert cases[name]["adjacency_jaccard"] == 1
        assert cases[name]["adjacency_agreement"] == 1
    assert cases["drop_one_whole_burst"]["dropped_events"] >= 1
    assert cases["drop_one_device"]["dropped_events"] in (3, 5)
    assert cases["thin_30_percent"]["dropped_events"] >= 1
    assert any(cases[f"jitter_{seconds}s"]["adjacency_jaccard"] < 1 for seconds in (5, 15, 30))


def test_sampling_preserves_whole_groups_and_skips_oversized() -> None:
    events = tuple(Payment(f"big{i}", i, "big") for i in range(10_001)) + (
        Payment("small1", 0, "small"),
        Payment("small2", 2, "small"),
    )
    selection = evaluate_stability(events)["selection"]
    assert selection["original_events"] == 10_003
    assert selection["sample_events"] == 2
    assert selection["omitted_events"] == 10_001
    assert selection["oversized_groups"] == 1
    assert selection["sampling_truncated"] is True
    assert selection["partial_groups"] is False
    assert selection["selected_groups"] == 1
    empty = evaluate_stability(events[:-2])
    assert empty["selection"]["empty_reason"] == "all_groups_exceed_budget"
    assert empty["selection"]["sample_events"] == 0


def test_all_unknown_device_events_stay_singletons() -> None:
    report = evaluate_stability((Payment("a", 1, ""), Payment("b", 2, "")))
    cases = {case["perturbation"]: case for case in report["cases"]}
    assert cases["identity"]["metrics"]["retained_bursts"] == 2
    assert cases["identity"]["metrics"]["adjacency_jaccard"] is None
    assert cases["drop_one_device"]["config"]["applied"] is False
    assert cases["drop_one_device"]["metrics"]["dropped_events"] == 0


def test_empty_and_duplicate_inputs() -> None:
    empty = evaluate_stability(())
    assert empty["selection"]["empty_reason"] == "no_input"
    assert all(case["metrics"]["adjacency_retention"] is None for case in empty["cases"])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_stability((Payment("a", 0, "a"), Payment("a", 1, "b")))
    with pytest.raises(ValueError, match="seed"):
        evaluate_stability((), True)

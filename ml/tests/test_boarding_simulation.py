import inspect
import json
from dataclasses import replace

import pytest

from tramflow_ml.boarding.alignment import DecodeConfig
from tramflow_ml.boarding.simulation import (
    evaluate_synthetic,
    generate_synthetic_cases,
    predict_synthetic,
)


def test_two_families_cases_determinism_and_json() -> None:
    first = evaluate_synthetic(42)
    second = evaluate_synthetic(42)
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )
    assert first["case_count"] == 24
    assert {case["family"] for case in first["cases"]} == {"renewal", "correlated_batching"}
    assert first["real_stop_accuracy"] == "unverified"
    assert first["real_label_promotion_allowed"] is False
    assert generate_synthetic_cases(42) != generate_synthetic_cases(43)


def test_denominator_is_all_events_and_nonidentifiable_cases_abstain() -> None:
    report = evaluate_synthetic(7)
    for case in report["cases"]:
        for method, metrics in case["metrics"].items():
            assert metrics["assigned_events"] + metrics["unknown_events"] == metrics["total_events"]
            assert metrics["coverage"] == metrics["assigned_events"] / metrics["total_events"]
            assert metrics["correct_visit_events"] <= metrics["assigned_events"]
            assert metrics["correct_stop_events"] <= metrics["assigned_events"]
            assert not metrics["calibrated"]
            if case["expected_abstention"] or method == "B0":
                assert metrics["assigned_events"] == 0
                assert metrics["synthetic_stop_accuracy_assigned"] is None
    totals = {row["total_events"] for row in report["summary"].values()}
    assert len(totals) == 1


def test_truth_is_excluded_from_decoder_inputs() -> None:
    case = generate_synthetic_cases(10)[0]
    changed_truth = replace(case.truth, stop_ids=("hidden",) * len(case.truth.stop_ids))
    changed = replace(case, truth=changed_truth)
    assert predict_synthetic(case.observations) == predict_synthetic(changed.observations)
    assert set(inspect.signature(predict_synthetic).parameters) == {"observation", "config"}
    assert "truth" not in case.observations.__dataclass_fields__


def test_missing_runs_open_ends_repeated_stops_and_delay_order() -> None:
    cases = {case.name: case for case in generate_synthetic_cases(42) if case.family == "renewal"}
    for count in (1, 3, 5):
        case = cases[f"missing_{count}"]
        assert case.missing_consecutive_visits == count
        assert case.truth.visits == (0, *range(count + 1, 8))
    assert cases["open_ends"].truth.visits == (2, 3, 4, 5)
    assert len(cases["sparse_directions"].truth.visits) == 1
    repeated = cases["repeated_stop"].truth
    assert repeated.visits.count(0) == 2 and repeated.visits.count(3) == 2
    assert repeated.stop_ids.count("s0") == 4
    delayed = cases["payment_delays"].truth.visits
    assert tuple(sorted(delayed)) != delayed


def test_wrong_anchor_can_reduce_accuracy_and_is_not_hidden() -> None:
    report = evaluate_synthetic(42)
    rows = [case for case in report["cases"] if case["case"] == "corrupted_anchor"]
    assert all("corrupted" in case["anchor_source"][0] for case in rows)
    assert any(
        case["metrics"]["B2"]["correct_stop_events"] < case["metrics"]["B2"]["assigned_events"]
        for case in rows
    )
    assert any(
        case["metrics"]["exact_monotone"]["assigned_events"]
        < case["metrics"]["exact_monotone"]["total_events"]
        for case in report["cases"]
    )


def test_budget_failure_stays_in_denominator() -> None:
    report = evaluate_synthetic(42, DecodeConfig(max_transitions=1))
    rows = [case["metrics"]["exact_monotone"] for case in report["cases"]]
    assert all(row["budget_exhausted"] and row["assigned_events"] == 0 for row in rows)
    assert report["summary"]["exact_monotone"]["budget_exhausted_cases"] == 24
    assert report["summary"]["exact_monotone"]["total_events"] > 0


def test_seed_type_rejected() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_cases(True)

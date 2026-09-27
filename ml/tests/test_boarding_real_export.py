import csv
import gzip
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_boarding_service_policy import notice, registry

from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.composition import RouteEvidence, timetable_emissions
from tramflow_ml.boarding.real_export import export_real_dataset
from tramflow_ml.boarding.sequence import CycleTemplate, infer_sequence

DAYS = ("2025-06-30", "2025-07-01", "2025-08-31", "2025-09-01")


def gz_csv(path: Path, rows: list[list[Any]]) -> None:
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    path.write_bytes(gzip.compress(text.getvalue().encode(), mtime=0))


def bind_day(run: Path, day: str) -> None:
    folder = run / day
    receipt = {"files": {p.name: digest(p) for p in folder.iterdir()
                         if p.name != "receipt.json"}}
    write_json(folder / "receipt.json", receipt)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["day_receipts"][day] = digest(folder / "receipt.json")
    write_json(run / "manifest.json", manifest)


def source(run: Path, days: tuple[str, ...] = DAYS) -> Path:
    run.mkdir()
    reports = []
    for day in days:
        folder = run / day
        folder.mkdir()
        report = {"day": day, "source_success": 4, "assigned_weak": 1, "ambiguous": 2,
                  "unassigned": 1, "source_rejected": 1, "soft_expected_count": 3}
        reports.append(report)
        write_json(folder / "evaluation.json", report)
        (folder / "event_assignments.csv.gz").write_bytes(b"extra hash-bound source artifact")
        (folder / "mass_ledger.csv").write_text(
            "route,event_hour,inferred_stable,ambiguous,unassigned\n"
            f"1,{day} 00,1,2,1\n"
        )
        (folder / "stop_hour_weak_counts.csv").write_text(
            "route,event_hour,stop_id,direction,inferred_count\n"
            f"1,{day} 00,a,0,1\n"
        )
        gz_csv(folder / "stop_hour_soft_counts.csv.gz", [
            ["route", "event_hour", "stop_id", "direction", "expected_count", "applicable_pattern"],
            ["1", f"{day} 00", "b", "1", 1, False],
            ["1", f"{day} 00", "a", "0", 1, True],
            ["1", f"{day} 00", "a", "0", 1, True],
        ])
    write_json(run / "manifest.json", {
        "schema_version": "boarding-real-run.v1", "complete": True,
        "implementation": {"composition.py": "a" * 64},
        "config": {"dates": list(reversed(days)), "routes": ["1"]},
        "reports": reports, "day_receipts": {}, "source_success": 4 * len(days),
        "assigned_weak": len(days), "ambiguous": 2 * len(days), "unassigned": len(days),
    })
    for day in days:
        bind_day(run, day)
    return run


def read_output(out: Path, name: str) -> list[dict[str, str]]:
    with gzip.open(out / f"{name}.csv.gz", "rt") as handle:
        return list(csv.DictReader(handle))


def test_chronological_boundaries_calendar_and_experimental_label_contract(tmp_path: Path) -> None:
    run = source(tmp_path / "run")
    out = tmp_path / "out"
    manifest = export_real_dataset(run, out)
    assert [r["day"] for r in read_output(out, "soft_train")] == [DAYS[0]] * 2
    development = read_output(out, "soft_development")
    assert [r["day"] for r in development] == [DAYS[1]] * 2 + [DAYS[2]] * 2
    assert [r["day"] for r in read_output(out, "soft_diagnostic")] == [DAYS[3]] * 2
    row = development[0]
    assert row["event_hour"] == "2025-07-01T00:00:00+03:00"
    assert row["weekday"] == "1" and row["hour"] == "0"
    assert row["stop_id"] == "a" and float(row["expected_count"]) == 2
    assert row["label_origin"] == "inferred_conditional_uncalibrated"
    assert row["training_eligible"] == "True"
    assert development[1]["applicable_pattern"] == "False"
    assert development[1]["training_eligible"] == "False"
    assert not any("lag" in field for field in row)
    assert manifest["point_in_time_sources_verified"] is False
    assert manifest["real_stop_accuracy"] == "unverified"
    assert manifest["date_count"] == 4
    assert manifest["dates"] == list(DAYS)


def test_mass_blocked_patterns_and_partial_strict_counts_remain_separate(tmp_path: Path) -> None:
    run = source(tmp_path / "run")
    out = tmp_path / "out"
    manifest = export_real_dataset(run, out)
    soft = [r for split in ("train", "development", "diagnostic")
            for r in read_output(out, "soft_" + split)]
    strict = [r for split in ("train", "development", "diagnostic")
              for r in read_output(out, "weak_" + split)]
    assert sum(float(r["expected_count"]) for r in soft) == manifest["source_decoded"] == 12
    assert sum(int(r["inferred_count"]) for r in strict) == manifest["source_strict_accepted"] == 4
    assert len(strict) == 4 and all(r["stop_id"] == "a" for r in strict)
    assert sum(float(r["expected_count"]) for r in soft if r["training_eligible"] == "False") == 4
    assert sum(f["blocked_pattern_count"] for f in manifest["files"].values()) == 4
    assert manifest["source_success"] == 16
    assert manifest["source_unassigned"] == manifest["source_rejected"] == 4
    assert manifest["source_outside_selected_dates_or_routes"] is None
    assert manifest["out_of_scope_count_available"] is False
    assert "not zero" in manifest["strict_coverage"]


def test_output_is_byte_reproducible_and_empty_splits_have_headers(tmp_path: Path) -> None:
    run = source(tmp_path / "run", (DAYS[0],))
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = export_real_dataset(run, first), export_real_dataset(run, second)
    assert a == b
    for name in ["manifest.json", *[f["name"] for f in a["files"].values()]]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert read_output(first, "soft_diagnostic") == []
    assert read_output(first, "weak_development") == []


@pytest.mark.parametrize("name", ["receipt.json", "event_assignments.csv.gz",
                                 "stop_hour_soft_counts.csv.gz", "evaluation.json"])
def test_tampering_any_receipt_bound_file_fails_before_output(tmp_path: Path, name: str) -> None:
    run = source(tmp_path / "run")
    path = run / DAYS[0] / name
    path.write_bytes(path.read_bytes() + b" ")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="missing or changed"):
        export_real_dataset(run, out)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["incomplete", "missing_receipt", "missing_report",
                                     "aggregate_mismatch", "wrong_schema"])
def test_incomplete_or_inconsistent_manifest_refused(tmp_path: Path, mutation: str) -> None:
    run = source(tmp_path / "run")
    manifest = json.loads((run / "manifest.json").read_text())
    if mutation == "incomplete":
        manifest["complete"] = False
    elif mutation == "missing_receipt":
        del manifest["day_receipts"][DAYS[0]]
    elif mutation == "missing_report":
        manifest["reports"].pop()
    elif mutation == "aggregate_mismatch":
        manifest["source_success"] += 1
    else:
        manifest["schema_version"] = "other"
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        export_real_dataset(run, tmp_path / "out")


@pytest.mark.parametrize("value", ["4", "-1", "nan", "inf"])
def test_rehashed_invalid_soft_mass_cannot_become_training_data(tmp_path: Path, value: str) -> None:
    run = source(tmp_path / "run", (DAYS[0],))
    gz_csv(run / DAYS[0] / "stop_hour_soft_counts.csv.gz", [
        ["route", "event_hour", "stop_id", "direction", "expected_count", "applicable_pattern"],
        ["1", f"{DAYS[0]} 00", "a", "0", value, True],
    ])
    bind_day(run, DAYS[0])
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        export_real_dataset(run, out)
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("wrong_day", [True, False])
def test_rehashed_wrong_day_and_strict_mass_are_rejected(tmp_path: Path, wrong_day: bool) -> None:
    run = source(tmp_path / "run", (DAYS[0],))
    strict = run / DAYS[0] / "stop_hour_weak_counts.csv"
    original = strict.read_text()
    bad = original.replace(DAYS[0], DAYS[1]) if wrong_day else original.replace(",a,0,1", ",a,0,2")
    strict.write_text(bad)
    bind_day(run, DAYS[0])
    with pytest.raises(ValueError, match="civil date" if wrong_day else "route-hour mass"):
        export_real_dataset(run, tmp_path / "out")


def test_nonempty_output_and_absent_manifest_are_refused(tmp_path: Path) -> None:
    run = source(tmp_path / "run")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="empty"):
        export_real_dataset(run, out)
    assert (out / "keep.txt").read_text() == "user data"
    with pytest.raises(ValueError, match="manifest"):
        export_real_dataset(tmp_path / "absent", tmp_path / "new")


def test_explicit_policy_retains_all_soft_and_strict_mass_but_blocks_training(
    tmp_path: Path,
) -> None:
    run = source(tmp_path / "run", (DAYS[0],))
    write_json(run / DAYS[0] / "patterns.json", {"1": {"warnings": []}})
    bind_day(run, DAYS[0])
    source_hash = digest(run / "manifest.json")
    policy = registry(tmp_path, [notice("fixture", ["1"], "2025-01-01", None, "daily")])
    out = tmp_path / "out"
    result = export_real_dataset(run, out, service_policy=policy)
    soft, strict = read_output(out, "soft_train"), read_output(out, "weak_train")
    assert sum(float(r["expected_count"]) for r in soft) == 3
    assert sum(int(r["inferred_count"]) for r in strict) == 1
    assert all(r["training_eligible"] == "False" for r in soft + strict)
    assert {r["engine_applicable_pattern"] for r in soft} == {"True", "False"}
    assert all("notice_exclusion:fixture" in r["service_policy_reasons"] for r in soft + strict)
    assert result["service_policy"]["policy_blocked_soft_mass"] == 3
    assert result["service_policy"]["raw_engine_blocked_soft_mass"] == 1
    assert result["service_policy"]["policy_blocked_strict_mass"] == 1
    assert result["service_policy"]["registry_sha256"] == digest(policy)
    assert digest(run / "manifest.json") == source_hash


def test_policy_requires_hash_bound_patterns_and_snapshots(tmp_path: Path) -> None:
    run = source(tmp_path / "run", (DAYS[0],))
    policy = registry(tmp_path, [notice("fixture", ["1"], "2025-01-01", None, "daily")])
    with pytest.raises(ValueError, match="receipt"):
        export_real_dataset(run, tmp_path / "out", service_policy=policy)
    write_json(run / DAYS[0] / "patterns.json", {"1": {"warnings": []}})
    bind_day(run, DAYS[0])
    (tmp_path / "notice.html").write_text("changed")
    with pytest.raises(ValueError, match="snapshot"):
        export_real_dataset(run, tmp_path / "out", service_policy=policy)


def test_policy_restoration_releases_only_soft_rows_without_inventing_strict_counts(
    tmp_path: Path,
) -> None:
    day = "2025-06-21"
    run = source(tmp_path / "run", (day,))
    folder = run / day
    write_json(folder / "patterns.json", {"12": {
        "warnings": ["weekend_entuziastov_diversion_end_unknown",
                     "incomplete_timetable_coverage_clock_evidence_disabled"],
        "applicable_pattern": False,
    }})
    (folder / "mass_ledger.csv").write_text(
        "route,event_hour,inferred_stable,ambiguous,unassigned\n" + f"12,{day} 00,0,3,1\n"
    )
    (folder / "stop_hour_weak_counts.csv").write_text(
        "route,event_hour,stop_id,direction,inferred_count\n"
    )
    gz_csv(folder / "stop_hour_soft_counts.csv.gz", [
        ["route", "event_hour", "stop_id", "direction", "expected_count", "applicable_pattern"],
        ["12", f"{day} 00", "a", "0", 3, False],
    ])
    report = json.loads((folder / "evaluation.json").read_text())
    report.update(assigned_weak=0, ambiguous=3)
    write_json(folder / "evaluation.json", report)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(assigned_weak=0, ambiguous=3, reports=[report])
    manifest["config"]["routes"] = ["12"]
    write_json(run / "manifest.json", manifest)
    bind_day(run, day)
    policy = registry(tmp_path, [notice("7177", ["12"], day, None, "restoration")])
    out = tmp_path / "out"
    result = export_real_dataset(run, out, service_policy=policy)
    soft = read_output(out, "soft_train")
    assert len(soft) == 1 and float(soft[0]["expected_count"]) == 3
    assert soft[0]["applicable_pattern"] == soft[0]["engine_applicable_pattern"] == "False"
    assert soft[0]["training_eligible"] == "True"
    assert "restored_warning:7177" in soft[0]["service_policy_reasons"]
    assert "informational_warning:" in soft[0]["service_policy_reasons"]
    assert read_output(out, "weak_train") == []
    assert result["service_policy"]["raw_engine_blocked_soft_mass"] == 3
    assert result["service_policy"]["restored_soft_mass"] == 3
    assert result["service_policy"]["policy_blocked_soft_mass"] == 0


def test_mixed_clock_precision_breaks_equal_state_symmetry() -> None:
    template = CycleTemplate("7", ("p", "p"), ("a", "b"), ("0", "1"),
                             (300, 300), (False, False), "synthetic:precision-regression")
    evidence = RouteEvidence(
        template, ({"boardable": True}, {"boardable": True}), ((3600,), (3600,)),
        ("historical_exact_day", "current_proxy_transfer"), (), (), True,
    )
    times = np.array([3600.0 - 10800 + 15 + 180])
    emissions = timetable_emissions(times, evidence)
    result = infer_sequence(times, template, log_emissions=emissions)
    np.testing.assert_allclose(result.posterior[0], [0.2998321212, 0.7001678788], atol=1e-9)
    neutral = infer_sequence(times, template,
                             log_emissions=timetable_emissions(times, evidence, use_schedule=False))
    np.testing.assert_allclose(neutral.posterior[0], [0.5, 0.5])


@pytest.mark.parametrize("with_policy", [False, True])
def test_mixed_precision_day_guard_is_always_applied_without_losing_any_mass(
    tmp_path: Path, with_policy: bool,
) -> None:
    days = ("2025-07-29", "2025-07-30")
    run = source(tmp_path / "run", days)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["routes"] = ["1", "7"]
    for day in days:
        folder = run / day
        for name in ("mass_ledger.csv", "stop_hour_weak_counts.csv"):
            lines = (folder / name).read_text().splitlines()
            (folder / name).write_text("\n".join(
                lines + ["7," + row.split(",", 1)[1] for row in lines[1:]]
            ) + "\n")
        with gzip.open(folder / "stop_hour_soft_counts.csv.gz", "rt") as handle:
            rows = list(csv.reader(handle))
        gz_csv(folder / "stop_hour_soft_counts.csv.gz",
               rows + [["7", *row[1:]] for row in rows[1:]])
        write_json(folder / "patterns.json", {"1": {"warnings": []}, "7": {"warnings": []}})
        report = next(r for r in manifest["reports"] if r["day"] == day)
        for field in ("source_success", "assigned_weak", "ambiguous", "unassigned",
                      "source_rejected", "soft_expected_count"):
            report[field] *= 2
        write_json(folder / "evaluation.json", report)
    for field in ("source_success", "assigned_weak", "ambiguous", "unassigned"):
        manifest[field] *= 2
    write_json(run / "manifest.json", manifest)
    for day in days:
        bind_day(run, day)
    policy = registry(tmp_path, [notice("other", ["99"], "2025-01-01", None, "daily")])
    out = tmp_path / "out"
    result = export_real_dataset(run, out, service_policy=policy if with_policy else None)
    soft, strict = read_output(out, "soft_development"), read_output(out, "weak_development")
    assert sum(float(r["expected_count"]) for r in soft) == 12
    assert sum(int(r["inferred_count"]) for r in strict) == 4
    for row in soft + strict:
        affected = row["route"] == "7" and row["day"] == "2025-07-29"
        assert json.loads(row["algorithmic_quality_reasons"]) == (
            ["mixed_schedule_precision_uncalibrated"] if affected else []
        )
        if affected:
            assert row["training_eligible"] == "False"
        elif row["applicable_pattern"] == "True":
            assert row["training_eligible"] == "True"
    exclusion = result["algorithmic_exclusions"][0]
    assert exclusion["affected_soft_mass"] == 3 and exclusion["affected_strict_mass"] == 1
    assert exclusion["source_composition_sha256"] == "a" * 64
    assert exclusion["official_service_notice"] is False
    if with_policy:
        assert result["service_policy"]["policy_blocked_strict_mass"] == 0
        assert result["service_policy"]["policy_blocked_soft_mass"] == 4

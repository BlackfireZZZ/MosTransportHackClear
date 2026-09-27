import json
import sys
from pathlib import Path

import pytest

from tramflow_ml.intake import IntakeRequest, profile_sample, render
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.data_v1 import ValidationEvent  # noqa: E402, I001


@pytest.fixture
def fixture_dir(tmp_path):
    generate_dataset(SyntheticConfig(), tmp_path / "in")
    return tmp_path / "in"


@pytest.fixture
def report(fixture_dir):
    return profile_sample(IntakeRequest(source=fixture_dir))


def test_report_counts_every_physical_row_of_both_streams(report):
    validations = report["streams"]["validations"]
    telemetry = report["streams"]["telemetry"]

    assert (validations["rows"], validations["readable"], validations["unreadable"]) == (67, 67, {})
    assert (telemetry["rows"], telemetry["readable"], telemetry["unreadable"]) == (12, 12, {})


def test_duplicates_split_repeats_from_conflicts_the_way_ingestion_does(report):
    duplicates = report["streams"]["validations"]["duplicates"]

    assert duplicates == {
        "conflict_rate": 0.0,
        "conflicting_rows": 0,
        "distinct_keys": 64,
        "key": "event_id",
        "repeat_rate": round(3 / 67, 6),
        "repeated_rows": 3,
        "unkeyed_rows": 0,
    }


def test_date_span_reports_observed_dates_and_the_gap_between_them(report):
    assert report["streams"]["validations"]["date_span"] == {
        "coverage_ratio": round(64 / 731, 6),
        "first_date": "2024-01-01",
        "first_instant": "2024-01-01T18:57:00+03:00",
        "last_date": "2025-12-31",
        "last_instant": "2025-12-31T18:00:00+03:00",
        "observed_dates": 64,
        "span_days": 731,
        "unobserved_dates_in_span": 667,
    }


def test_fields_are_summarised_by_their_classification(report):
    fields = report["streams"]["validations"]["fields"]

    assert fields["event_id"]["classification"] == "identifier"
    assert fields["event_id"]["distinct_values"] == 64
    assert fields["event_at"]["classification"] == "timestamp"
    assert fields["event_at"]["first"] == "2024-01-01T18:57:00+03:00"
    assert fields["stop_sequence"] == {
        "classification": "measure",
        "maximum": 3,
        "minimum": 0,
        "missing": 0,
        "missing_rate": 0.0,
        "out_of_contract": 0,
        "present": 67,
        "sum": 99,
        "unparsed": 0,
    }
    assert fields["target"]["values"] == {
        "other": 0,
        "synthetic_boardings": 67,
        "validation_count": 0,
    }


def test_join_coverage_speaks_the_identity_vocabulary(report):
    coverage = report["streams"]["validations"]["join_coverage"]

    assert coverage["crosswalk"] == "identity"
    assert coverage["matched"] == {"by_kind": {"exact_id": 67}, "total": 67}
    assert coverage["unmatched"] == {"by_reason": {}, "total": 0}
    assert (coverage["ambiguous"], coverage["stale"]) == (0, 0)
    assert coverage["unalignable"]["total"] == 0
    assert coverage["rates"]["matched"] == 1.0


def test_join_coverage_is_not_measured_without_a_catalog(fixture_dir, tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("validations.jsonl", "telemetry.jsonl"):
        (bare / name).write_bytes((fixture_dir / name).read_bytes())

    stream = profile_sample(IntakeRequest(source=bare))["streams"]["validations"]

    assert stream["join_coverage"]["measured"] is False
    assert "--catalog" in stream["join_coverage"]["reason"]


def test_year_horizon_is_refused_with_the_two_numbers_that_refused_it(report):
    year = report["supportability"]["horizons"]["year"]

    assert year["supportable"] is False
    assert (year["observed_complete_buckets"], year["required_complete_buckets"]) == (22, 48)
    assert year["blockers"][0].startswith("insufficient_history: policy 'year/month' reads 36")
    assert "yields 22" in year["blockers"][0]


def test_only_whole_buckets_count_so_a_partial_first_day_is_not_a_day(report):
    """The fixture starts at 18:57 and ends at 18:00, so neither end date is complete."""
    horizons = report["supportability"]["horizons"]

    assert horizons["month"]["observed_complete_buckets"] == 729
    assert horizons["day"]["observed_complete_buckets"] == 17519
    assert horizons["year"]["observed_complete_buckets"] == 22


def test_a_measure_outside_its_contract_is_counted_not_waved_through(fixture_dir, tmp_path):
    rows = [
        json.loads(line)
        for line in (fixture_dir / "validations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["stop_sequence"] = -1
    rows[1]["stop_sequence"] = 2.5
    (fixture_dir / "validations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    summary = profile_sample(IntakeRequest(source=fixture_dir))["streams"]["validations"]["fields"]

    assert summary["stop_sequence"]["unparsed"] == 0
    assert summary["stop_sequence"]["out_of_contract"] == 2
    assert summary["stop_sequence"]["sum"] is None
    assert summary["stop_sequence"]["minimum"] == -1


def test_every_horizon_reports_its_policy_and_thresholds(report):
    horizons = report["supportability"]["horizons"]

    assert set(horizons) == {"day", "month", "year"}
    assert [horizons[name]["policy"] for name in ("day", "month", "year")] == [
        "day/hour",
        "month/day",
        "year/month",
    ]
    assert all(horizon["required_coverage_ratio"] == 0.5 for horizon in horizons.values())
    assert all("sparse_coverage" in horizon["blockers"][-1] for horizon in horizons.values())


def test_targets_name_the_one_that_is_present_and_the_one_that_is_not(report):
    targets = report["supportability"]["targets"]

    assert targets["observed"]["synthetic_boardings"] == {
        "blockers": [],
        "rows": 67,
        "supportable": True,
    }
    assert targets["observed"]["validation_count"]["supportable"] is False
    assert targets["observed"]["validation_count"]["blockers"] == [
        "absent: no row declares the target 'validation_count'"
    ]
    assert targets["units"] == {"event_count": 67, "other": 0}


def test_source_section_records_the_mapping_that_was_used(report):
    source = report["source"]

    assert source["profile"] == "canonical"
    assert source["format"] == "jsonl"
    assert source["catalog"] == {
        "entity_version": "synthetic-entities.v1",
        "file": "entities.json",
    }
    assert source["streams"]["validations"]["file"] == "validations.jsonl"
    assert source["streams"]["validations"]["unused_columns"] == []


def test_rendered_report_is_json_with_sorted_keys_and_one_trailing_newline(report):
    serialized = render(report)

    assert serialized.endswith("}\n")
    assert json.loads(serialized) == report
    assert serialized == json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_golden_counts_describe_contract_valid_rows(fixture_dir, report):
    rows = [
        ValidationEvent.model_validate_json(line)
        for line in (fixture_dir / "validations.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == report["streams"]["validations"]["rows"]
    assert len({row.event_id for row in rows}) == (
        report["streams"]["validations"]["duplicates"]["distinct_keys"]
    )

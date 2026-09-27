import json

import pytest

from tramflow_ml.intake import IntakeRequest, SchemaError, load_profile, profile_sample
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset

LATE_FIELD = "vehicle_id"
ROWS = 400
ROWS_WITHOUT_FIELD = 250


def write_profile(path, columns, source_format="jsonl", files=None, has_header=True):
    payload = {
        "schema_version": "intake-profile.v1",
        "name": "declared",
        "source_format": source_format,
        "files": files
        or {"validations": "validations.jsonl", "telemetry": "telemetry.jsonl"},
        "columns": columns,
        "constants": {},
        "timestamp_format": None,
        "assume_timezone": None,
    }
    if source_format == "csv":
        payload["has_header"] = has_header
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_profile(path)


@pytest.fixture
def fixture_dir(tmp_path):
    generate_dataset(SyntheticConfig(), tmp_path / "source")
    return tmp_path / "source"


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.fixture
def late_field_sample(fixture_dir, tmp_path):
    """400 rows where vehicle_id appears only from row 251 onward."""
    sample = tmp_path / "late"
    sample.mkdir()
    (sample / "entities.json").write_bytes((fixture_dir / "entities.json").read_bytes())
    base = _rows(fixture_dir / "validations.jsonl")
    rows = []
    for index in range(ROWS):
        row = dict(base[index % len(base)])
        row["event_id"] = f"synthetic:validation:{index + 1}"
        if index < ROWS_WITHOUT_FIELD:
            row.pop(LATE_FIELD, None)
        rows.append(row)
    _write(sample / "validations.jsonl", rows)
    (sample / "telemetry.jsonl").write_bytes((fixture_dir / "telemetry.jsonl").read_bytes())
    return sample


def test_a_field_that_first_appears_late_in_the_file_is_still_found(late_field_sample):
    report = profile_sample(IntakeRequest(source=late_field_sample))

    summary = report["streams"]["validations"]["fields"][LATE_FIELD]
    assert report["streams"]["validations"]["rows"] == ROWS
    assert summary["present"] == ROWS - ROWS_WITHOUT_FIELD
    assert summary["missing_rate"] == round(ROWS_WITHOUT_FIELD / ROWS, 6)


@pytest.fixture
def absent_field_sample(fixture_dir, tmp_path):
    sample = tmp_path / "absent"
    sample.mkdir()
    (sample / "entities.json").write_bytes((fixture_dir / "entities.json").read_bytes())
    rows = [
        {key: value for key, value in row.items() if key != LATE_FIELD}
        for row in _rows(fixture_dir / "validations.jsonl")
    ]
    _write(sample / "validations.jsonl", rows)
    (sample / "telemetry.jsonl").write_bytes((fixture_dir / "telemetry.jsonl").read_bytes())
    return sample


def test_a_field_absent_from_every_row_refuses_with_the_checklist(absent_field_sample):
    """Reporting it as 100% missing would silently delete the checklist instead."""
    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=absent_field_sample))

    assert LATE_FIELD in str(raised.value)
    assert "unmapped, required" in str(raised.value)


def test_the_profile_the_checklist_asks_for_turns_it_into_a_missing_rate(
    absent_field_sample, tmp_path
):
    profile = write_profile(tmp_path / "declared.json", {LATE_FIELD: LATE_FIELD})

    report = profile_sample(IntakeRequest(source=absent_field_sample, profile=profile))

    summary = report["streams"]["validations"]["fields"][LATE_FIELD]
    assert summary["present"] == 0
    assert summary["missing_rate"] == 1.0
    assert report["source"]["streams"]["validations"]["declared_columns_never_seen"] == [
        LATE_FIELD
    ]


def test_an_explicit_mapping_takes_effect_even_for_a_key_seen_only_once(
    late_field_sample, tmp_path
):
    profile = write_profile(tmp_path / "explicit.json", {LATE_FIELD: LATE_FIELD})

    report = profile_sample(IntakeRequest(source=late_field_sample, profile=profile))

    assert report["streams"]["validations"]["fields"][LATE_FIELD]["present"] == 150


def test_a_repeated_csv_header_name_says_which_positions_collide(tmp_path):
    sample = tmp_path / "repeated"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.csv").write_text("a,b,a\n1,2,3\n", encoding="utf-8")

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=sample))

    message = str(raised.value)
    assert "positions 1 and 3 (a)" in message
    assert message.count("validations.csv:") == 1


def test_a_misspelled_profile_column_leads_the_refusal(fixture_dir, tmp_path):
    from test_intake_schemas import COLUMNS, FILES, write_alternate_schema

    alternate = write_alternate_schema(fixture_dir, tmp_path / "alternate")
    columns = {**COLUMNS, "event_id": "row_kye"}
    profile = write_profile(
        tmp_path / "typo.json", columns, source_format="csv", files=FILES
    )

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=alternate, profile=profile))

    assert "profile.columns names columns absent from" in str(raised.value)
    assert "row_kye" in str(raised.value)


def test_an_empty_file_is_diagnosed_as_empty(tmp_path):
    sample = tmp_path / "empty"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(SchemaError, match="is empty: it holds no rows at all"):
        profile_sample(IntakeRequest(source=sample))


def test_an_empty_csv_is_diagnosed_as_empty(tmp_path):
    sample = tmp_path / "empty-csv"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.csv").write_text("", encoding="utf-8")

    with pytest.raises(SchemaError, match="is empty: it holds no lines at all"):
        profile_sample(IntakeRequest(source=sample))

import csv
import json
from datetime import datetime

import pytest

from tramflow_ml.intake import IntakeRequest, SchemaError, load_profile, profile_sample
from tramflow_ml.synthetic import MOSCOW, SyntheticConfig, generate_dataset

COLUMNS = {
    "available_at": "published",
    "direction_id": "bound",
    "entity_version": "catalog_ver",
    "event_at": "happened",
    "event_id": "row_key",
    "latitude": "lat",
    "longitude": "lon",
    "route_id": "line_code",
    "schema_version": "contract",
    "source_version": "extract_ver",
    "stop_id": "platform_code",
    "stop_sequence": "visit_no",
    "synthetic": "is_generated",
    "target": "measure_name",
    "unit": "measure_unit",
    "vehicle_id": "car_code",
}
FILES = {"telemetry": "positions.csv", "validations": "boardings.csv"}
TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M:%S"
TIMESTAMP_FIELDS = ("event_at", "available_at")
INTEGER_FIELDS = ("stop_sequence",)
RECONCILING_SECTIONS = ("streams", "supportability")


def _cell(name, value):
    if name in TIMESTAMP_FIELDS:
        return datetime.fromisoformat(value).astimezone(MOSCOW).strftime(TIMESTAMP_FORMAT)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _rewrite(source, target, columns, float_integers=False):
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    names = sorted(columns)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([columns[name] for name in names])
        for row in rows:
            writer.writerow(
                [
                    f"{row[name]}.0"
                    if float_integers and name in INTEGER_FIELDS
                    else _cell(name, row[name])
                    for name in names
                ]
            )


def write_alternate_schema(fixture_dir, destination, float_integers=False):
    """The same events rendered as CSV: renamed columns, naive Moscow timestamps.

    ``float_integers`` spells every integer as ``0.0``, which is what a CSV written
    from a float-typed column looks like — the first encoding we did not author.
    """
    destination.mkdir(parents=True)
    (destination / "entities.json").write_bytes((fixture_dir / "entities.json").read_bytes())
    without = {
        "validations": ("latitude", "longitude"),
        "telemetry": ("target", "unit"),
    }
    for stream, absent in without.items():
        columns = {name: column for name, column in COLUMNS.items() if name not in absent}
        _rewrite(
            fixture_dir / f"{stream}.jsonl",
            destination / FILES[stream],
            columns,
            float_integers,
        )
    return destination


def write_profile(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": "intake-profile.v1",
                "name": "alternate-fixture",
                "source_format": "csv",
                "files": FILES,
                "columns": COLUMNS,
                "constants": {},
                "timestamp_format": TIMESTAMP_FORMAT,
                "assume_timezone": "Europe/Moscow",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def canonical(tmp_path):
    generate_dataset(SyntheticConfig(), tmp_path / "canonical")
    return tmp_path / "canonical"


@pytest.fixture
def alternate(canonical, tmp_path):
    return write_alternate_schema(canonical, tmp_path / "alternate")


@pytest.fixture
def reports(canonical, alternate, tmp_path):
    profile = load_profile(write_profile(tmp_path / "alternate.json"))
    return (
        profile_sample(IntakeRequest(source=canonical)),
        profile_sample(IntakeRequest(source=alternate, profile=profile)),
    )


def test_every_reconciling_section_is_identical_across_the_two_schemas(reports):
    first, second = reports

    for section in RECONCILING_SECTIONS:
        assert first[section] == second[section], section


def test_row_duplicate_and_span_counts_reconcile(reports):
    first, second = (report["streams"]["validations"] for report in reports)

    assert (first["rows"], second["rows"]) == (67, 67)
    assert first["duplicates"] == second["duplicates"]
    assert first["date_span"] == second["date_span"]
    assert first["join_coverage"] == second["join_coverage"]


def test_measures_reconcile_although_one_encoding_carries_them_as_text(reports):
    first, second = (report["streams"]["validations"]["fields"] for report in reports)

    assert first["stop_sequence"]["sum"] == second["stop_sequence"]["sum"] == 99
    assert first["synthetic"]["values"] == second["synthetic"]["values"]
    assert first["event_at"]["first"] == second["event_at"]["first"]


def test_only_the_encoding_section_differs(reports):
    first, second = reports

    assert first["source"] != second["source"]
    assert (first["source"]["format"], second["source"]["format"]) == ("jsonl", "csv")
    assert second["source"]["profile"] == "alternate-fixture"
    assert second["source"]["timestamp_format"] == TIMESTAMP_FORMAT
    assert second["source"]["streams"]["validations"]["columns"]["event_id"] == "row_key"


def test_alternate_schema_needs_its_profile(alternate):
    with pytest.raises(SchemaError) as failure:
        profile_sample(IntakeRequest(source=alternate))

    assert "cannot locate the streams" in str(failure.value)


@pytest.fixture
def float_spelled(canonical, tmp_path):
    return write_alternate_schema(canonical, tmp_path / "floats", float_integers=True)


def test_integers_spelled_as_floats_reconcile_with_the_same_facts(
    canonical, float_spelled, tmp_path
):
    """A CSV written from a float-typed column spells 0 as 0.0; it is the same 0."""
    profile = load_profile(write_profile(tmp_path / "floats.json"))

    first = profile_sample(IntakeRequest(source=canonical))
    second = profile_sample(IntakeRequest(source=float_spelled, profile=profile))

    assert first["streams"] == second["streams"]
    assert first["supportability"] == second["supportability"]
    assert second["streams"]["validations"]["fields"]["stop_sequence"]["sum"] == 99
    assert second["streams"]["validations"]["duplicates"]["conflicting_rows"] == 0


def test_a_headerless_rendering_reconciles_once_its_positions_are_declared(
    canonical, tmp_path
):
    destination = tmp_path / "headerless"
    destination.mkdir(parents=True)
    (destination / "entities.json").write_bytes((canonical / "entities.json").read_bytes())
    names = sorted(COLUMNS)
    for stream in ("validations", "telemetry"):
        rows = [
            json.loads(line)
            for line in (canonical / f"{stream}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        with (destination / FILES[stream]).open("w", encoding="utf-8", newline="") as stream_file:
            writer = csv.writer(stream_file, lineterminator="\n")
            for row in rows:
                writer.writerow([_cell(n, row[n]) if n in row else "" for n in names])
    path = tmp_path / "headerless.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "intake-profile.v1",
                "name": "headerless",
                "source_format": "csv",
                "has_header": False,
                "files": FILES,
                "columns": {name: f"column_{index + 1}" for index, name in enumerate(names)},
                "constants": {},
                "timestamp_format": TIMESTAMP_FORMAT,
                "assume_timezone": "Europe/Moscow",
            }
        ),
        encoding="utf-8",
    )

    first = profile_sample(IntakeRequest(source=canonical))
    second = profile_sample(IntakeRequest(source=destination, profile=load_profile(path)))

    assert second["streams"]["validations"]["rows"] == 67, "row one must be data, not a header"
    assert first["streams"] == second["streams"]

import json

import pytest

from tramflow_ml.intake import IntakeRequest, classify, profile_sample, render
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset

CARD = "4276-3801-5512-9043"
VEHICLE = "PASSENGER-CARD-9912345678"
EXTRACT = "TICKET-7781234512349999"
IDENTIFIER_VALUES = (CARD, VEHICLE, EXTRACT)
IDENTIFIER_SUMMARY_KEYS = {
    "classification",
    "distinct_values",
    "format_signatures",
    "max_length",
    "min_length",
    "missing",
    "missing_rate",
    "present",
}


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def carded_sample(tmp_path):
    """The canonical fixture with card-shaped values in three identifier columns."""
    generate_dataset(SyntheticConfig(), tmp_path / "source")
    sample = tmp_path / "carded"
    sample.mkdir()
    (sample / "entities.json").write_bytes((tmp_path / "source/entities.json").read_bytes())
    for stream in ("validations", "telemetry"):
        rows = _read(tmp_path / f"source/{stream}.jsonl")
        _write(
            sample / f"{stream}.jsonl",
            [
                {
                    **row,
                    "event_id": f"{CARD}-{index}",
                    "vehicle_id": VEHICLE,
                    "source_version": EXTRACT,
                }
                for index, row in enumerate(rows)
            ],
        )
    return sample


@pytest.fixture
def report(carded_sample):
    return profile_sample(IntakeRequest(source=carded_sample))


def test_no_identifier_value_appears_anywhere_in_the_report(report):
    serialized = render(report)

    for value in IDENTIFIER_VALUES:
        assert value not in serialized


def test_the_report_still_describes_those_fields_by_shape(report):
    fields = report["streams"]["validations"]["fields"]

    assert fields["event_id"]["distinct_values"] == 67
    assert fields["vehicle_id"]["distinct_values"] == 1
    assert fields["vehicle_id"]["min_length"] == len(VEHICLE)
    assert fields["source_version"]["format_signatures"] == {"alnum_punct": 67}


def test_an_identifier_summary_carries_only_the_permitted_keys(report):
    for stream in report["streams"].values():
        for name, summary in stream["fields"].items():
            if classify(name) == "identifier":
                assert set(summary) == IDENTIFIER_SUMMARY_KEYS, name


def test_no_summary_offers_a_sample_or_a_most_frequent_value(report):
    forbidden = {"examples", "mode", "most_frequent", "sample", "top", "values_seen"}

    for stream in report["streams"].values():
        for name, summary in stream["fields"].items():
            assert forbidden.isdisjoint(summary), name


def test_enumerated_fields_report_only_code_defined_members(report):
    fields = report["streams"]["validations"]["fields"]

    assert set(fields["target"]["values"]) == {"other", "synthetic_boardings", "validation_count"}
    assert set(fields["unit"]["values"]) == {"event_count", "other"}
    assert set(fields["synthetic"]["values"]) == {"false", "other", "true"}


def test_a_value_outside_the_allowlist_is_counted_but_never_quoted(carded_sample):
    rows = _read(carded_sample / "validations.jsonl")
    rows[0]["unit"] = "passengers-per-card-4276"
    _write(carded_sample / "validations.jsonl", rows)

    report = profile_sample(IntakeRequest(source=carded_sample))

    assert report["streams"]["validations"]["fields"]["unit"]["values"]["other"] == 1
    assert "passengers-per-card-4276" not in render(report)


HEADERLESS_CELLS = ("4276380155129043", "A-14", "Chistye Prudy", "T-2201")
HEADERLESS_ROW = ",".join((*HEADERLESS_CELLS, "2026-03-01T07:41:10+03:00"))


@pytest.fixture
def headerless_sample(tmp_path):
    """A CSV whose first line is data, which intake cannot know before reading it."""
    sample = tmp_path / "headerless"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.csv").write_text(
            f"{HEADERLESS_ROW}\n{HEADERLESS_ROW}\n", encoding="utf-8"
        )
    return sample


def test_a_headerless_csv_never_quotes_row_one_as_a_column_name(headerless_sample):
    from tramflow_ml.intake import SchemaError

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=headerless_sample))

    message = str(raised.value)
    for cell in (*HEADERLESS_CELLS, "2026-03-01T07:41:10+03:00"):
        assert cell not in message
    assert "<column 1: 16 digits>" in message
    assert "may have no header row" in message or "no header row looks like" in message


def test_a_headerless_csv_leaks_nothing_into_the_report_either(headerless_sample, tmp_path):
    """With a profile that maps the positions, the run succeeds and still quotes no cell."""
    profile_path = tmp_path / "headerless.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "intake-profile.v1",
                "name": "headerless",
                "source_format": "csv",
                "has_header": False,
                "files": {"validations": "validations.csv", "telemetry": "telemetry.csv"},
                "columns": {"event_id": "column_1", "route_id": "column_2"},
                "constants": {},
                "timestamp_format": None,
                "assume_timezone": None,
            }
        ),
        encoding="utf-8",
    )

    from tramflow_ml.intake import SchemaError, load_profile

    with pytest.raises(SchemaError) as raised:
        profile_sample(
            IntakeRequest(source=headerless_sample, profile=load_profile(profile_path))
        )

    message = str(raised.value)
    for cell in HEADERLESS_CELLS:
        assert cell not in message
    assert "column_1" in message and "column_2" in message


def test_a_json_row_keyed_by_an_identifier_is_redacted_too(tmp_path):
    sample = tmp_path / "keyed"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.jsonl").write_text(
            json.dumps({CARD: 1, "4276-3801-5512-9044": 2}) + "\n", encoding="utf-8"
        )

    from tramflow_ml.intake import SchemaError

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=sample))

    assert CARD not in str(raised.value)
    assert "digits" in str(raised.value) or "alnum_punct" in str(raised.value)


LEAD_CARD = "ZZLEADCARDZZ4276380155129040"
MIXED_CELLS = (
    "2026-03-01T07:41:10+03:00",
    LEAD_CARD,
    "A-14",
    "Chistye Prudy",
    "T-2200",
)


@pytest.fixture
def mixed_headerless_sample(tmp_path):
    """A headerless row where one cell IS word-like and the other four are not.

    ``ZZLEADCARDZZ4276380155129040`` is a valid Python identifier — letters then
    digits, no separator — so a per-cell test prints it. Only judging the line can
    redact it.
    """
    sample = tmp_path / "mixed"
    sample.mkdir()
    row = ",".join(MIXED_CELLS)
    for name in ("validations", "telemetry"):
        (sample / f"{name}.csv").write_text(f"{row}\n{row}\n", encoding="utf-8")
    return sample


def test_a_word_like_cell_beside_data_cells_is_redacted_with_the_line(
    mixed_headerless_sample,
):
    from tramflow_ml.intake import SchemaError, printable

    assert printable(LEAD_CARD), "the canary must pass the per-cell test to be a real probe"

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=mixed_headerless_sample))

    message = str(raised.value)
    for cell in MIXED_CELLS:
        assert cell not in message
    assert "4276380155129040" not in message
    assert "<column 2: 28 alnum>" in message


def test_the_headerless_diagnosis_fires_when_only_some_cells_fail(
    mixed_headerless_sample,
):
    from tramflow_ml.intake import SchemaError

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=mixed_headerless_sample))

    assert "Not every cell of the first line is word-like" in str(raised.value)


def test_a_word_like_cell_never_reaches_the_report_either(mixed_headerless_sample, tmp_path):
    from tramflow_ml.intake import load_profile

    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "intake-profile.v1",
                "name": "mixed",
                "source_format": "csv",
                "has_header": True,
                "files": {"validations": "validations.csv", "telemetry": "telemetry.csv"},
                "columns": {"event_id": LEAD_CARD},
                "constants": {},
                "timestamp_format": None,
                "assume_timezone": None,
            }
        ),
        encoding="utf-8",
    )

    from tramflow_ml.intake import SchemaError

    with pytest.raises(SchemaError) as raised:
        profile_sample(
            IntakeRequest(source=mixed_headerless_sample, profile=load_profile(path))
        )

    assert LEAD_CARD not in str(raised.value)


def test_a_genuine_header_is_untouched_by_the_line_rule(tmp_path):
    """Every cell word-like, so the line is a header and reads exactly as before."""
    sample = tmp_path / "real"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.csv").write_text("ticket_no,line,platform\na,b,c\n", encoding="utf-8")

    from tramflow_ml.intake import SchemaError

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=sample))

    message = str(raised.value)
    assert "columns found (3): ticket_no, line, platform" in message
    assert "<column" not in message
    assert "Not every cell" not in message

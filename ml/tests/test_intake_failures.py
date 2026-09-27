import json

import pytest

from tramflow_ml.intake import (
    CANONICAL_FIELDS,
    FIELD_CONTRACTS,
    IntakeError,
    IntakeRequest,
    SchemaError,
    load_profile,
    profile_sample,
)

FOREIGN_HEADER = "ticket_no,line,platform,ts,tram"
FOREIGN_ROW = "4276-3801-5512-9043,A-14,Chistye Prudy,01.03.2026 07:41:10,T-2201"
FOREIGN_COLUMNS = ("line", "platform", "ticket_no", "tram", "ts")


def _profile(path, files, columns, constants=None):
    path.write_text(
        json.dumps(
            {
                "schema_version": "intake-profile.v1",
                "name": "partial",
                "source_format": "csv",
                "files": files,
                "columns": columns,
                "constants": constants or {},
                "timestamp_format": None,
                "assume_timezone": None,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def foreign_sample(tmp_path):
    sample = tmp_path / "foreign"
    sample.mkdir()
    (sample / "validations.csv").write_text(f"{FOREIGN_HEADER}\n{FOREIGN_ROW}\n", encoding="utf-8")
    (sample / "telemetry.csv").write_text(
        f"{FOREIGN_HEADER},lat,lon\n{FOREIGN_ROW},55.76,37.64\n", encoding="utf-8"
    )
    return sample


@pytest.fixture
def failure(foreign_sample):
    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=foreign_sample))
    return str(raised.value)


def test_failure_names_every_column_it_found(failure):
    for column in FOREIGN_COLUMNS:
        assert column in failure
    assert "columns found (5): ticket_no, line, platform, ts, tram" in failure


def test_failure_names_every_canonical_field_it_needs_and_what_each_must_carry(failure):
    for name in sorted(CANONICAL_FIELDS - {"latitude", "longitude"}):
        assert name in failure
        assert FIELD_CONTRACTS[name] in failure
    assert "unmapped, required (14):" in failure


def test_failure_says_nothing_was_mapped_and_nothing_was_used(failure):
    assert "mapped (0): -" in failure
    assert "supplied as constants (0): -" in failure
    assert "columns not used (5): line, platform, ticket_no, tram, ts" in failure  # sorted


def test_failure_states_that_no_column_is_guessed_and_defers_the_real_mapping(failure):
    assert "No column is mapped by name similarity, position or a synonym list" in failure
    assert "TASK-049/TASK-050" in failure


def test_failure_carries_a_fill_in_profile_the_operator_can_complete(failure):
    skeleton = json.loads(failure[failure.index("{") : failure.rindex("}") + 1])

    assert skeleton["schema_version"] == "intake-profile.v1"
    assert skeleton["source_format"] == "csv"
    assert skeleton["files"] == {"telemetry": "telemetry.csv", "validations": "validations.csv"}
    assert set(skeleton["columns"]) == CANONICAL_FIELDS
    assert set(skeleton["columns"].values()) == {None}
    assert "--profile" in failure


def test_failure_quotes_no_value_from_the_sample(failure):
    for value in FOREIGN_ROW.split(","):
        assert value not in failure


def test_a_partial_mapping_leaves_only_the_rest_unmapped(foreign_sample, tmp_path):
    profile = _profile(
        tmp_path / "partial.json",
        files={"validations": "validations.csv", "telemetry": "telemetry.csv"},
        columns={"event_id": "ticket_no", "route_id": "line", "event_at": "ts"},
    )

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=foreign_sample, profile=load_profile(profile)))

    message = str(raised.value)
    assert "mapped (3): event_at <- ts, event_id <- ticket_no, route_id <- line" in message
    assert "unmapped, required (11):" in message
    assert "columns not used (2): platform, tram" in message


def test_constants_close_a_field_the_source_does_not_carry(foreign_sample, tmp_path):
    profile = _profile(
        tmp_path / "constants.json",
        files={"validations": "validations.csv", "telemetry": "telemetry.csv"},
        columns={"event_id": "ticket_no"},
        constants={"schema_version": "data.v1"},
    )

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=foreign_sample, profile=load_profile(profile)))

    message = str(raised.value)
    unmapped_block = message.split("unmapped, required")[1].split("columns not used")[0]
    assert "supplied as constants (1): schema_version" in message
    assert "schema_version" not in unmapped_block


def test_a_directory_without_recognisable_stream_files_lists_what_is_there(tmp_path):
    sample = tmp_path / "empty"
    sample.mkdir()
    (sample / "README.txt").write_text("nothing here\n", encoding="utf-8")

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=sample))

    message = str(raised.value)
    assert "files present (1): README.txt" in message
    assert "validations.jsonl" in message
    assert "--profile" in message


def test_a_profile_naming_an_absent_file_says_which_one(foreign_sample, tmp_path):
    profile = _profile(
        tmp_path / "absent.json",
        files={"validations": "nope.csv", "telemetry": "telemetry.csv"},
        columns={},
    )

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=foreign_sample, profile=load_profile(profile)))

    message = str(raised.value)
    assert "'nope.csv'" in message
    assert "files present (2): telemetry.csv, validations.csv" in message


def test_a_profile_naming_a_field_that_is_not_canonical_is_refused(tmp_path):
    profile = _profile(
        tmp_path / "bogus.json",
        files={"validations": "a.csv", "telemetry": "b.csv"},
        columns={"card_id": "ticket_no"},
    )

    with pytest.raises(IntakeError) as raised:
        load_profile(profile)

    assert "not canonical: ['card_id']" in str(raised.value)
    assert "event_id" in str(raised.value)


def test_both_spellings_present_is_an_ambiguity_intake_refuses_to_resolve(tmp_path):
    sample = tmp_path / "both"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.jsonl").write_text('{"event_id": "a"}\n', encoding="utf-8")
        (sample / f"{name}.csv").write_text("event_id\na\n", encoding="utf-8")

    with pytest.raises(SchemaError) as raised:
        profile_sample(IntakeRequest(source=sample))

    assert "is not intake's decision" in str(raised.value)


def test_a_source_that_is_not_a_directory_is_an_input_error(tmp_path):
    with pytest.raises(IntakeError, match="is not a directory"):
        profile_sample(IntakeRequest(source=tmp_path / "absent"))

import base64
import csv
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tramflow_ml.ingestion import (
    NORMALIZATION_VERSION,
    ColumnAdapter,
    IngestionConfig,
    IngestionError,
    InputChangedError,
    ingest,
    pipeline,
)
from tramflow_ml.ingestion import checkpoint as checkpoint_module
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.data_v1 import EntityCatalog, TelemetryEvent, ValidationEvent  # noqa: E402, I001

OUTPUT_FILES = ("validations.jsonl", "telemetry.jsonl", "quarantine.jsonl", "manifest.json")


@pytest.fixture
def fixture_dir(tmp_path):
    generate_dataset(SyntheticConfig(), tmp_path / "in")
    return tmp_path / "in"


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digests(output):
    return {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in OUTPUT_FILES}


def rewrite_validations(fixture_dir, transform):
    path = fixture_dir / "validations.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(transform(rows)) + "\n", encoding="utf-8")


def counts(manifest, stream):
    return manifest["streams"][stream]["counts"]


def test_tiny_fixture_reconciles_and_matches_contracts(fixture_dir, tmp_path):
    config = IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=10)

    manifest = ingest(config)

    assert counts(manifest, "validations") == {
        "input_rows": 67,
        "valid": 64,
        "duplicates": 3,
        "quarantined": 0,
    }
    assert counts(manifest, "telemetry") == {
        "input_rows": 12,
        "valid": 12,
        "duplicates": 0,
        "quarantined": 0,
    }
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
    assert manifest["source_manifest"]["synthetic"] is True
    catalog = EntityCatalog.model_validate_json((fixture_dir / "entities.json").read_text())
    validations = lines(tmp_path / "out/validations.jsonl")
    assert len({row["event_id"] for row in validations}) == 64
    for row in validations:
        ValidationEvent.model_validate_json(json.dumps(row)).validate_entities(catalog)
    for row in lines(tmp_path / "out/telemetry.jsonl"):
        TelemetryEvent.model_validate_json(json.dumps(row)).validate_entities(catalog)
    assert not (tmp_path / "out/checkpoint.json").exists()
    assert not (tmp_path / "out/dedup.sqlite").exists()


def test_valid_rows_are_byte_identical_to_the_canonical_fixture_encoding(fixture_dir, tmp_path):
    ingest(IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=7))

    source = [
        json.loads(line) for line in (fixture_dir / "validations.jsonl").read_bytes().splitlines()
    ]
    kept = list(dict((row["event_id"], row) for row in source).values())
    expected = b"".join(pipeline.encode(row) for row in kept)
    assert (tmp_path / "out/validations.jsonl").read_bytes() == expected


def test_corrupted_rows_are_quarantined_with_reasons(fixture_dir, tmp_path):
    def corrupt(rows):
        missing = json.loads(rows[1])
        del missing["vehicle_id"]
        unknown_stop = {**json.loads(rows[2]), "stop_id": "synthetic:stop:99"}
        early = json.loads(rows[3])
        early["available_at"] = "2023-12-31T00:00:00+03:00"
        wrong_type = {**json.loads(rows[4]), "stop_sequence": "1"}
        return [
            "{not json",
            json.dumps(missing),
            json.dumps(unknown_stop),
            json.dumps(early),
            json.dumps(wrong_type),
            *rows[5:],
        ]

    rewrite_validations(fixture_dir, corrupt)
    telemetry = fixture_dir / "telemetry.jsonl"
    rows = telemetry.read_text(encoding="utf-8").splitlines()
    telemetry.write_text(
        "\n".join([rows[0].replace('"latitude":55.75', '"latitude":NaN'), *rows[1:]]) + "\n"
    )

    manifest = ingest(IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=3))

    assert counts(manifest, "validations") == {
        "input_rows": 67,
        "valid": 59,
        "duplicates": 3,
        "quarantined": 5,
    }
    assert manifest["streams"]["validations"]["quarantine_reasons"] == {
        "availability_before_event": 1,
        "decode_error": 1,
        "invalid_type": 1,
        "missing_field": 1,
        "unknown_entity": 1,
    }
    assert manifest["streams"]["telemetry"]["quarantine_reasons"] == {"invalid_value": 1}
    quarantine = lines(tmp_path / "out/quarantine.jsonl")
    assert [row["row_index"] for row in quarantine if row["stream"] == "validations"] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert quarantine[0]["raw"] == "{not json"
    assert quarantine[1]["detail"] == "vehicle_id"


def test_conflicting_duplicate_is_quarantined_not_dropped(fixture_dir, tmp_path):
    def conflict(rows):
        changed = {**json.loads(rows[0]), "vehicle_id": "synthetic:vehicle:99"}
        return [*rows, json.dumps(changed)]

    rewrite_validations(fixture_dir, conflict)

    manifest = ingest(IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=100))

    assert counts(manifest, "validations") == {
        "input_rows": 68,
        "valid": 64,
        "duplicates": 3,
        "quarantined": 1,
    }
    [row] = lines(tmp_path / "out/quarantine.jsonl")
    assert row["reason"] == "conflicting_duplicate"
    assert row["row_index"] == 67
    assert json.loads(row["raw"])["vehicle_id"] == "synthetic:vehicle:99"


def interrupt_after(monkeypatch, commits):
    original = pipeline.commit_checkpoint
    seen = {"count": 0}

    def flaky(path, checkpoint):
        original(path, checkpoint)
        seen["count"] += 1
        if seen["count"] == commits:
            raise RuntimeError("injected crash after checkpoint")

    monkeypatch.setattr(pipeline, "commit_checkpoint", flaky)


@pytest.mark.parametrize("crash_after_commits", [1, 4, 7])
def test_interrupted_and_resumed_run_matches_uninterrupted(
    fixture_dir, tmp_path, monkeypatch, crash_after_commits
):
    straight = IngestionConfig(input=fixture_dir, output=tmp_path / "straight", chunk_size=10)
    resumed = replace(straight, output=tmp_path / "resumed")
    expected = ingest(straight)
    interrupt_after(monkeypatch, crash_after_commits)
    with pytest.raises(RuntimeError, match="injected"):
        ingest(resumed)
    assert (tmp_path / "resumed/checkpoint.json").exists()
    assert not (tmp_path / "resumed/manifest.json").exists()
    monkeypatch.undo()

    actual = ingest(resumed)

    assert actual == expected
    assert digests(tmp_path / "resumed") == digests(tmp_path / "straight")


def test_resume_drops_index_rows_committed_after_the_checkpoint(fixture_dir, tmp_path, monkeypatch):
    def conflict_free_duplicates(rows):
        return rows

    rewrite_validations(fixture_dir, conflict_free_duplicates)
    config = IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=17)
    original_commit = pipeline.DedupIndex.commit
    calls = {"count": 0}

    def crash_after_index_commit(self):
        original_commit(self)
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("injected crash between index commit and checkpoint")

    monkeypatch.setattr(pipeline.DedupIndex, "commit", crash_after_index_commit)
    with pytest.raises(RuntimeError, match="injected"):
        ingest(config)
    monkeypatch.undo()

    manifest = ingest(config)

    assert counts(manifest, "validations") == {
        "input_rows": 67,
        "valid": 64,
        "duplicates": 3,
        "quarantined": 0,
    }


def test_resume_refuses_changed_input(fixture_dir, tmp_path, monkeypatch):
    config = IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=10)
    interrupt_after(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        ingest(config)
    monkeypatch.undo()
    before = (tmp_path / "out/validations.jsonl").read_bytes()
    with (fixture_dir / "validations.jsonl").open("ab") as stream:
        stream.write(b'{"schema_version":"data.v1"}\n')

    with pytest.raises(InputChangedError, match="validations.jsonl"):
        ingest(config)

    assert (tmp_path / "out/validations.jsonl").read_bytes() == before
    assert not (tmp_path / "out/manifest.json").exists()


def test_resume_refuses_different_chunk_size(fixture_dir, tmp_path, monkeypatch):
    config = IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=10)
    interrupt_after(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        ingest(config)
    monkeypatch.undo()

    with pytest.raises(IngestionError, match="chunk_size"):
        ingest(replace(config, chunk_size=11))


def test_checkpoint_write_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    checkpoint_module.write_atomic(path, b"first\n")

    def fail_replace(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        checkpoint_module.write_atomic(path, b"second\n")

    assert path.read_bytes() == b"first\n"
    assert not path.with_name("checkpoint.json.tmp").exists()


def test_run_directory_rules(fixture_dir, tmp_path):
    completed = IngestionConfig(input=fixture_dir, output=tmp_path / "done", chunk_size=50)
    ingest(completed)
    with pytest.raises(IngestionError, match="completed"):
        ingest(completed)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep")
    with pytest.raises(IngestionError, match="notes.txt"):
        ingest(replace(completed, output=foreign))
    assert (foreign / "notes.txt").read_text() == "keep"
    with pytest.raises(IngestionError, match="differ"):
        ingest(replace(completed, output=fixture_dir))


def write_csv_fixture(fixture_dir, csv_dir):
    csv_dir.mkdir()
    for name in ("entities.json", "manifest.json"):
        (csv_dir / name).write_bytes((fixture_dir / name).read_bytes())
    renames = {
        "event_id": "id",
        "event_at": "ts",
        "available_at": "seen_ts",
        "vehicle_id": "vehicle",
    }
    for stream in ("validations", "telemetry"):
        rows = lines(fixture_dir / f"{stream}.jsonl")
        header = [
            renames.get(key, key) for key in rows[0] if key not in ("schema_version", "synthetic")
        ]
        with (csv_dir / f"{stream}.csv").open("w", newline="", encoding="utf-8") as stream_file:
            writer = csv.writer(stream_file)
            writer.writerow(header)
            for row in rows:
                writer.writerow(
                    [
                        row[key][:19].replace("T", " ")
                        if key in ("event_at", "available_at")
                        else row[key]
                        for key in row
                        if key not in ("schema_version", "synthetic")
                    ]
                )
    return ColumnAdapter(
        name="csv-fixture",
        columns=renames,
        constants={"schema_version": "data.v1", "synthetic": True},
        timestamp_format="%Y-%m-%d %H:%M:%S",
        assume_timezone="Europe/Moscow",
    )


def test_csv_adapter_round_trip_matches_jsonl_outputs(fixture_dir, tmp_path):
    adapter = write_csv_fixture(fixture_dir, tmp_path / "csv")
    jsonl = ingest(IngestionConfig(input=fixture_dir, output=tmp_path / "from-jsonl", chunk_size=9))

    csv_manifest = ingest(
        IngestionConfig(
            input=tmp_path / "csv",
            output=tmp_path / "from-csv",
            chunk_size=9,
            source_format="csv",
            adapter=adapter,
        )
    )

    assert csv_manifest["streams"] == {
        stream: {**report, "input_file": f"{stream}.csv"}
        for stream, report in jsonl["streams"].items()
    }
    assert csv_manifest["output_hash"] == jsonl["output_hash"]
    for name in ("validations.jsonl", "telemetry.jsonl"):
        assert (tmp_path / "from-csv" / name).read_bytes() == (
            tmp_path / "from-jsonl" / name
        ).read_bytes()


def test_csv_malformed_line_and_bad_timestamp_are_quarantined(fixture_dir, tmp_path):
    adapter = write_csv_fixture(fixture_dir, tmp_path / "csv")
    path = tmp_path / "csv/validations.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1] + ",extra"
    rows[2] = re.sub(r"\d{4}-\d{2}-\d{2} ", "2024-13-01 ", rows[2], count=1)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest = ingest(
        IngestionConfig(
            input=tmp_path / "csv",
            output=tmp_path / "out",
            chunk_size=5,
            source_format="csv",
            adapter=adapter,
        )
    )

    assert manifest["streams"]["validations"]["quarantine_reasons"] == {
        "decode_error": 1,
        "invalid_timestamp": 1,
    }
    assert counts(manifest, "validations")["input_rows"] == 67


def test_invalid_utf8_line_keeps_the_exact_bytes_in_quarantine(fixture_dir, tmp_path):
    path = fixture_dir / "validations.jsonl"
    rows = path.read_bytes().splitlines()
    undecodable = b'{"event_id":"\xff\xfe"}'
    path.write_bytes(b"\n".join([undecodable, *rows]) + b"\n")

    manifest = ingest(IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=4))

    assert manifest["streams"]["validations"]["quarantine_reasons"] == {"decode_error": 1}
    [record] = lines(tmp_path / "out/quarantine.jsonl")
    assert base64.b64decode(record["raw_base64"]) == undecodable
    assert "�" in record["raw"]


def test_valid_utf8_quarantine_records_carry_no_base64_copy(fixture_dir, tmp_path):
    rewrite_validations(fixture_dir, lambda rows: ["{not json", *rows])

    ingest(IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=4))

    [record] = lines(tmp_path / "out/quarantine.jsonl")
    assert record["raw"] == "{not json"
    assert "raw_base64" not in record


def test_csv_header_tolerates_a_utf8_byte_order_mark(fixture_dir, tmp_path):
    adapter = write_csv_fixture(fixture_dir, tmp_path / "csv")
    plain = IngestionConfig(
        input=tmp_path / "csv",
        output=tmp_path / "plain",
        chunk_size=9,
        source_format="csv",
        adapter=adapter,
    )
    expected = ingest(plain)
    for name in ("validations.csv", "telemetry.csv"):
        path = tmp_path / "csv" / name
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    manifest = ingest(replace(plain, output=tmp_path / "bom"))

    assert manifest["streams"] == expected["streams"]
    assert manifest["output_hash"] == expected["output_hash"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda payload: {**payload, "chunk_size": "10"}, "chunk_size"),
        (
            lambda payload: {key: value for key, value in payload.items() if key != "commit_seq"},
            "commit_seq",
        ),
        (lambda payload: {**payload, "streams": {}}, "streams.validations"),
        (
            lambda payload: {
                **payload,
                "streams": {
                    **payload["streams"],
                    "telemetry": {**payload["streams"]["telemetry"], "offset": None},
                },
            },
            "streams.telemetry",
        ),
    ],
)
def test_malformed_checkpoint_is_refused(fixture_dir, tmp_path, monkeypatch, mutate, expected):
    config = IngestionConfig(input=fixture_dir, output=tmp_path / "out", chunk_size=10)
    interrupt_after(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        ingest(config)
    monkeypatch.undo()
    path = tmp_path / "out/checkpoint.json"
    path.write_text(json.dumps(mutate(json.loads(path.read_text()))), encoding="utf-8")

    with pytest.raises(IngestionError, match=expected):
        ingest(config)

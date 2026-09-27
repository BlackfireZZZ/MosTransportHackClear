import json
import subprocess
import sys

import pytest

from tramflow_ml import cli
from tramflow_ml.ingestion import pipeline
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset


@pytest.fixture
def fixture_dir(tmp_path):
    generate_dataset(SyntheticConfig(), tmp_path / "in")
    return tmp_path / "in"


def ingest_command(fixture_dir, output, *extra):
    return [
        sys.executable,
        "-m",
        "tramflow_ml.cli",
        "ingest",
        "--input",
        str(fixture_dir),
        "--output",
        str(output),
        *extra,
    ]


def test_cli_ingest_happy_path_reports_counts(fixture_dir, tmp_path):
    command = ingest_command(fixture_dir, tmp_path / "out", "--chunk-size", "25")

    run = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)

    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout)
    assert summary["streams"]["validations"] == {
        "input_rows": 67,
        "valid": 64,
        "duplicates": 3,
        "quarantined": 0,
    }
    assert summary["streams"]["telemetry"]["valid"] == 12
    assert (tmp_path / "out/manifest.json").exists()


def test_cli_ingest_rejects_missing_input_as_usage_error(tmp_path):
    command = ingest_command(tmp_path / "absent", tmp_path / "out")

    run = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)

    assert run.returncode == 2
    assert "missing input files" in run.stderr
    assert not (tmp_path / "out").exists()


def test_cli_ingest_exits_one_when_counts_do_not_reconcile(
    fixture_dir, tmp_path, monkeypatch, capsys
):
    original = pipeline._report

    def unbalanced(state, input_file):
        return {**original(state, input_file), "reconciled": False}

    monkeypatch.setattr(pipeline, "_report", unbalanced)
    monkeypatch.setattr(
        sys, "argv", ["tramflow-ml", *ingest_command(fixture_dir, tmp_path / "out")[3:]]
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert "reconciliation failed" in capsys.readouterr().err
    assert not (tmp_path / "out/manifest.json").exists()

import hashlib
import json
import shutil
import subprocess
import sys

import pytest
from test_intake_schemas import write_alternate_schema, write_profile

from tramflow_ml import cli
from tramflow_ml.intake import join
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset


def intake_command(source, *extra):
    return [sys.executable, "-m", "tramflow_ml.cli", "intake", "--input", str(source), *extra]


def inventory(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def fixture_dir(tmp_path):
    generate_dataset(SyntheticConfig(), tmp_path / "in")
    return tmp_path / "in"


def test_intake_exits_zero_and_prints_one_json_report(fixture_dir, tmp_path):
    run = subprocess.run(intake_command(fixture_dir), cwd=tmp_path, capture_output=True, text=True)

    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["schema_version"] == "intake.v1"
    assert report["streams"]["validations"]["rows"] == 67


def test_the_sample_is_byte_identical_after_a_run(fixture_dir, tmp_path):
    before = inventory(fixture_dir)

    run = subprocess.run(
        intake_command(fixture_dir, "--output", str(tmp_path / "report.json")),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    assert inventory(fixture_dir) == before
    assert (tmp_path / "report.json").read_text(encoding="utf-8") == run.stdout


def test_two_runs_on_the_same_sample_produce_identical_bytes(fixture_dir, tmp_path):
    first = subprocess.run(
        intake_command(fixture_dir), cwd=tmp_path, capture_output=True, text=True
    )
    second = subprocess.run(
        intake_command(fixture_dir), cwd=fixture_dir, capture_output=True, text=True
    )

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


def test_a_copy_at_another_absolute_path_produces_identical_bytes(fixture_dir, tmp_path):
    elsewhere = tmp_path / "deeper" / "somewhere-else"
    shutil.copytree(fixture_dir, elsewhere)

    first = subprocess.run(
        intake_command(fixture_dir), cwd=tmp_path, capture_output=True, text=True
    )
    second = subprocess.run(intake_command(elsewhere), cwd=tmp_path, capture_output=True, text=True)

    assert first.stdout == second.stdout


def test_an_unknown_schema_exits_two_with_the_checklist_on_stderr(tmp_path):
    sample = tmp_path / "foreign"
    sample.mkdir()
    for name in ("validations", "telemetry"):
        (sample / f"{name}.csv").write_text("ticket_no,ts\nabc,2026-03-01\n", encoding="utf-8")

    run = subprocess.run(intake_command(sample), cwd=tmp_path, capture_output=True, text=True)

    assert run.returncode == 2
    assert run.stdout == ""
    assert "unmapped, required" in run.stderr
    assert "--profile" in run.stderr
    assert "usage:" not in run.stderr


def test_an_output_inside_the_sample_is_refused(fixture_dir, tmp_path):
    run = subprocess.run(
        intake_command(fixture_dir, "--output", str(fixture_dir / "report.json")),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 2
    assert "intake never writes into the sample it profiles" in run.stderr
    assert not (fixture_dir / "report.json").exists()


def test_a_profile_lets_the_cli_read_the_alternate_schema(fixture_dir, tmp_path):
    alternate = write_alternate_schema(fixture_dir, tmp_path / "alternate")
    profile = write_profile(tmp_path / "alternate.json")

    canonical = subprocess.run(
        intake_command(fixture_dir), cwd=tmp_path, capture_output=True, text=True
    )
    rendered = subprocess.run(
        intake_command(alternate, "--profile", str(profile)),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert rendered.returncode == 0, rendered.stderr
    first, second = json.loads(canonical.stdout), json.loads(rendered.stdout)
    assert first["streams"] == second["streams"]
    assert first["supportability"] == second["supportability"]


def test_a_failed_invariant_exits_one(fixture_dir, monkeypatch, capsys):
    monkeypatch.setattr(join.JoinCoverage, "reconciles", lambda self, rows: False)
    monkeypatch.setattr(sys, "argv", ["tramflow-ml", *intake_command(fixture_dir)[3:]])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert "intake invariant failed" in capsys.readouterr().err


def test_an_absent_catalog_named_explicitly_is_a_usage_error(fixture_dir, tmp_path):
    run = subprocess.run(
        intake_command(fixture_dir, "--catalog", str(tmp_path / "absent.json")),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 2
    assert "not a readable entity catalog" in run.stderr


def test_ingest_still_behaves_as_before(fixture_dir, tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "tramflow_ml.cli",
            "ingest",
            "--input",
            str(fixture_dir),
            "--output",
            str(tmp_path / "out"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["streams"]["validations"]["input_rows"] == 67


def test_an_unwritable_output_is_a_usage_error_not_a_traceback(fixture_dir, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()

    run = subprocess.run(
        intake_command(fixture_dir, "--output", str(blocked)),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 2
    assert "cannot write --output" in run.stderr
    assert "Traceback" not in run.stderr


def test_an_output_under_a_file_is_a_usage_error_not_a_traceback(fixture_dir, tmp_path):
    wall = tmp_path / "wall"
    wall.write_text("not a directory\n", encoding="utf-8")

    run = subprocess.run(
        intake_command(fixture_dir, "--output", str(wall / "report.json")),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 2
    assert "cannot write --output" in run.stderr
    assert "Traceback" not in run.stderr

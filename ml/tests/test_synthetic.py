import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from tramflow_ml import cli
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.data_v1 import EntityCatalog, TelemetryEvent, ValidationEvent  # noqa: E402, I001
from contracts.forecast_v1 import DatasetManifest  # noqa: E402, I001


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_hand_counted_injections_and_shared_schema(tmp_path):
    config = SyntheticConfig(
        events=12,
        duplicate_every=3,
        late_every=4,
        telemetry_every=4,
        gap_every_days=0,
        start=date(2024, 2, 28),
        end=date(2024, 3, 2),
    )
    output = tmp_path / "data"
    result = generate_dataset(config, output)
    rows = lines(output / "validations.jsonl")
    telemetry = lines(output / "telemetry.jsonl")
    assert len(rows) == 16
    assert len({row["event_id"] for row in rows}) == 12
    assert len(telemetry) == 3
    assert result["generation"]["counts"] == {
        "unique_validations": 12,
        "duplicate_validations": 4,
        "late_validations": 3,
        "telemetry": 3,
    }
    catalog = EntityCatalog.model_validate_json((output / "entities.json").read_text())
    manifest = DatasetManifest.model_validate_json((output / "manifest.json").read_text())
    assert manifest.synthetic and manifest.target.value == "synthetic_boardings"
    assert len({stop.name for stop in catalog.stops}) < len(catalog.stops)
    seen = {}
    late = 0
    for row in rows:
        event = ValidationEvent.model_validate_json(json.dumps(row))
        event.validate_entities(catalog)
        if event.event_id in seen:
            assert row == seen[event.event_id]
            continue
        seen[event.event_id] = row
        if event.available_at - event.event_at >= timedelta(days=1):
            late += 1
            assert not event.visible_at(event.event_at + timedelta(hours=1))
        assert event.visible_at(event.available_at)
    assert late == 3
    for row in telemetry:
        TelemetryEvent.model_validate_json(json.dumps(row)).validate_entities(catalog)
    assert sum(cell["count"] for cell in result["generation"]["cell_totals"]) == 12
    assert result["generation"]["hour_totals"] == {"03": 2, "08": 4, "12": 2, "18": 4}


def test_identical_config_reproduces_all_bytes_and_changed_seed_changes_content(tmp_path):
    config = SyntheticConfig(events=100)
    first, second, third = (tmp_path / name for name in ("a", "b", "c"))
    generate_dataset(config, first)
    generate_dataset(config, second)
    generate_dataset(replace(config, seed=config.seed + 1), third)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    assert (first / "validations.jsonl").read_bytes() != (third / "validations.jsonl").read_bytes()


def test_inventory_hashes_bind_exact_files(tmp_path):
    result = generate_dataset(SyntheticConfig(), tmp_path / "data")
    inventory = {}
    for name in ("entities.json", "validations.jsonl", "telemetry.jsonl", "generation.json"):
        raw = (tmp_path / "data" / name).read_bytes()
        inventory[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert result["files"] == inventory
    encoded = (
        json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    assert result["manifest"]["source_hash"] == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (date(2024, 2, 28), date(2024, 3, 2), {"2024-02-28", "2024-02-29", "2024-03-01"}),
        (date(2025, 2, 28), date(2025, 3, 2), {"2025-02-28", "2025-03-01"}),
        (date(2025, 12, 31), date(2026, 1, 2), {"2025-12-31", "2026-01-01"}),
    ],
)
def test_real_calendar_days(start, end, expected, tmp_path):
    generate_dataset(
        SyntheticConfig(start=start, end=end, events=24, duplicate_every=0, gap_every_days=0),
        tmp_path / "data",
    )
    actual = {row["event_at"][:10] for row in lines(tmp_path / "data/validations.jsonl")}
    assert actual == expected


def test_multiyear_endpoints_gaps_and_directions(tmp_path):
    result = generate_dataset(SyntheticConfig(events=1000, gap_every_days=3), tmp_path / "data")
    rows = lines(tmp_path / "data/validations.jsonl")
    dates = {datetime.fromisoformat(row["event_at"]).date() for row in rows}
    assert min(dates) == date(2024, 1, 1)
    assert max(dates) == date(2025, 12, 31)
    assert not dates & {date.fromisoformat(raw) for raw in result["generation"]["gap_dates"]}
    assert len({row["direction_id"] for row in rows}) == 2
    assert len({row["route_id"] for row in rows}) == 2
    assert result["generation"]["hour_totals"] == {"03": 125, "08": 375, "12": 125, "18": 375}


@pytest.mark.parametrize(
    "override",
    [
        {"events": 0},
        {"events": True},
        {"seed": -1},
        {"telemetry_every": 0},
        {"duplicate_every": -1},
        {"late_every": -1},
        {"gap_every_days": 1},
        {"start": date(2026, 1, 1), "end": date(2024, 1, 1)},
    ],
)
def test_invalid_config_does_not_create_output(override, tmp_path):
    with pytest.raises(ValueError):
        generate_dataset(SyntheticConfig(**override), tmp_path / "data")
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_destination_is_preserved(kind, tmp_path):
    output = tmp_path / "data"
    if kind == "directory":
        output.mkdir()
        (output / "marker").write_text("keep")
    elif kind == "file":
        output.write_text("keep")
    else:
        target = tmp_path / "target"
        target.mkdir()
        (target / "marker").write_text("keep")
        output.symlink_to(target)
    with pytest.raises(FileExistsError):
        generate_dataset(SyntheticConfig(), output)
    assert (
        (output / "marker").read_text() == "keep"
        if kind != "file"
        else output.read_text() == "keep"
    )


def test_cli_runs_outside_repository_and_rejects_overwrite(tmp_path):
    command = [
        sys.executable,
        "-m",
        "tramflow_ml.cli",
        "generate-synthetic",
        "--mode",
        "million",
        "--events",
        "12",
        "--output",
        str(tmp_path / "data"),
    ]
    run = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["counts"]["unique_validations"] == 12
    again = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 2


def test_cli_rejects_invalid_config_without_creating_output(tmp_path):
    command = [
        sys.executable,
        "-m",
        "tramflow_ml.cli",
        "generate-synthetic",
        "--gap-every-days",
        "1",
        "--output",
        str(tmp_path / "data"),
    ]
    run = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 2
    assert "gap_every_days" in run.stderr
    assert not (tmp_path / "data").exists()


def test_cli_reports_write_failure_as_usage_error(tmp_path, monkeypatch, capsys):
    def fail(config, output):
        raise OSError("injected fixture disk failure")

    monkeypatch.setattr(cli, "generate_dataset", fail)
    monkeypatch.setattr(
        sys, "argv", ["tramflow-ml", "generate-synthetic", "--output", str(tmp_path / "data")]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "injected fixture disk failure" in capsys.readouterr().err


def test_write_failure_never_leaves_completion_manifest(tmp_path, monkeypatch):
    original = Path.open

    def fail_telemetry(self, *args, **kwargs):
        if self.name == "telemetry.jsonl":
            raise OSError("injected fixture disk failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_telemetry)
    with pytest.raises(OSError, match="injected"):
        generate_dataset(SyntheticConfig(), tmp_path / "data")
    assert not (tmp_path / "data/manifest.json").exists()

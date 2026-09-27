import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from tramflow_ml.boarding.audit import audit, verify_run
from tramflow_ml.boarding.records import AuditConfig

HEADER = "tran_no;device_no;tran_date_time;validation_result;ngpt_route;garage_number;bus_exit_no\n"


def archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "train.csv",
            HEADER
            + "a;001;2025-08-31 23:59:59;1;1 трамвай;01;01\n"
            + "a;001;2025-09-01 00:00:00;1;1 трамвай;01;01\n"
            + "c;002;2025-09-01 00:00:00;0;5 трамвай;02;02\n",
        )
        z.writestr(
            "test.csv",
            HEADER
            + "a;001;2025-09-01 00:00:00;1;1 трамвай;02;01\n"
            + "e;;2025-09-01 01:00:00;1;50 трамвай;;\n"
            + "f;003;2025-11-01 00:00:00;1;1 трамвай;03;03\n",
        )
        z.writestr("labels/labels_day_train.csv", "route;date;hour;boardings\n1;2025-08-31;23;1\n")
        z.writestr(
            "labels/labels_day_test.csv",
            "route;date;hour;boardings\n1;2025-09-01;0;2\n50;2025-09-01;1;1\n",
        )
    return path


def test_mass_tails_duplicates_privacy_and_resume(tmp_path: Path) -> None:
    src = archive(tmp_path / "input.zip")
    cfg = AuditConfig(chunk_rows=2)
    out = tmp_path / "run"
    audit(src, out, cfg)
    report = verify_run(out)
    assert report["source_success"] == 4
    assert report["label_mismatches"] == 0
    inventory = json.loads((out / "inventory.json").read_text())
    assert inventory["raw_rows"] == 6
    assert inventory["outside_range_rows"] == 1
    assert inventory["rejected_in_range"] == 1
    assert json.loads((out / "identity_quality.json").read_text())["conflicting_device_hours"] == 1
    import gzip

    records = []
    for p in sorted((out / "shards").glob("*.csv.gz")):
        records.extend(csv.DictReader(io.StringIO(gzip.decompress(p.read_bytes()).decode())))
    assert len({r["event_key"] for r in records}) == 6
    assert all(r["stop_id"] == "" for r in records)
    assert all(r["device_key"] != "001" for r in records)
    assert sum(r["assignment_status"] == "out_of_scope" for r in records) == 1
    before = {p.name: p.read_bytes() for p in (out / "shards").glob("*")}
    (out / "manifest.json").unlink()
    audit(src, out, cfg)
    assert before == {p.name: p.read_bytes() for p in (out / "shards").glob("*")}
    with pytest.raises(ValueError, match="complete"):
        audit(src, out, cfg)


def test_gate_detects_corrupt_shard(tmp_path: Path) -> None:
    out = tmp_path / "run"
    audit(archive(tmp_path / "input.zip"), out, AuditConfig(chunk_rows=2))
    next((out / "shards").glob("*.csv.gz")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        verify_run(out)


def test_resume_rejects_changed_config(tmp_path: Path) -> None:
    out = tmp_path / "run"
    src = archive(tmp_path / "input.zip")
    audit(src, out, AuditConfig(chunk_rows=2))
    (out / "manifest.json").unlink()
    with pytest.raises(ValueError, match="configuration"):
        audit(src, out, AuditConfig(chunk_rows=3))


def test_invalid_success_timestamp_blocks_export(tmp_path: Path) -> None:
    src = tmp_path / "bad.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("train.csv", HEADER + "a;1;bad;1;1 трамвай;1;1\n")
        z.writestr("test.csv", HEADER)
    with pytest.raises(ValueError, match="invalid"):
        audit(src, tmp_path / "run", AuditConfig())
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_subrange_filters_labels(tmp_path: Path) -> None:
    from datetime import date

    out = tmp_path / "run"
    audit(archive(tmp_path / "input.zip"), out, AuditConfig(start=date(2025, 9, 1)))
    assert verify_run(out)["source_success"] == 3


def test_corrupt_checkpoint_never_publishes_complete(tmp_path: Path) -> None:
    import sqlite3

    out = tmp_path / "run"
    src = archive(tmp_path / "input.zip")
    audit(src, out, AuditConfig())
    (out / "manifest.json").unlink()
    with sqlite3.connect(out / "audit.sqlite") as db:
        name, stats = db.execute("SELECT name,stats FROM chunks LIMIT 1").fetchone()
        payload = json.loads(stats)
        payload["successful_in_range"] += 10
        db.execute("UPDATE chunks SET stats=? WHERE name=?", (json.dumps(payload), name))
    with pytest.raises(ValueError, match="G0"):
        audit(src, out, AuditConfig())
    assert not (out / "manifest.json").exists()


def test_noncanonical_success_time_rejected(tmp_path: Path) -> None:
    src = tmp_path / "bad.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("train.csv", HEADER + "a;1;2025-1-1 0:00:00;1;1 трамвай;1;1\n")
        z.writestr("test.csv", HEADER)
    with pytest.raises(ValueError, match="invalid"):
        audit(src, tmp_path / "run", AuditConfig())


def test_interruption_resume_matches_clean_and_chunk_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip
    import importlib

    module = importlib.import_module("tramflow_ml.boarding.audit")
    original_process = module.process
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_process(*args, **kwargs)

    src = archive(tmp_path / "input.zip")
    resumed = tmp_path / "resumed"
    monkeypatch.setattr(module, "process", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        audit(src, resumed, AuditConfig(chunk_rows=2))
    assert not (resumed / "manifest.json").exists()
    monkeypatch.setattr(module, "process", original_process)
    audit(src, resumed, AuditConfig(chunk_rows=2))
    clean = tmp_path / "clean"
    audit(src, clean, AuditConfig(chunk_rows=2))
    assert {p.name: p.read_bytes() for p in (resumed / "shards").glob("*")} == {
        p.name: p.read_bytes() for p in (clean / "shards").glob("*")
    }
    assert (resumed / "mass_ledger.csv").read_bytes() == (clean / "mass_ledger.csv").read_bytes()
    large = tmp_path / "large"
    audit(src, large, AuditConfig(chunk_rows=100))

    def read_rows(directory):
        rows = []
        for p in (directory / "shards").glob("*.gz"):
            with gzip.open(p, "rt") as f:
                rows.extend(csv.DictReader(f))
        return sorted(rows, key=lambda row: row["event_key"])

    assert read_rows(large) == read_rows(clean)

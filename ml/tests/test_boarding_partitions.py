from pathlib import Path

import pytest
from test_boarding_audit import archive

from tramflow_ml.boarding.audit import audit
from tramflow_ml.boarding.partitions import prepare, read_day
from tramflow_ml.boarding.records import AuditConfig


def test_date_union_preserves_tails_rejections_and_resume(tmp_path: Path) -> None:
    source = tmp_path / "source"
    audit(archive(tmp_path / "in.zip"), source, AuditConfig(chunk_rows=2))
    out = tmp_path / "days"
    m = prepare(source, out)
    assert m["successful"] == 4
    assert m["events"] == 5
    day = read_day(out, "2025-09-01")
    assert len(day) == 4
    assert int(day.success.astype(int).sum()) == 3
    assert len(read_day(out, "2025-09-01", ("1",))) == 2
    assert read_day(out, "2025-11-01").empty
    (out / "manifest.json").unlink()
    assert prepare(source, out) == m
    with pytest.raises(ValueError, match="complete"):
        prepare(source, out)


def test_changed_partition_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    audit(archive(tmp_path / "in.zip"), source, AuditConfig())
    out = tmp_path / "days"
    prepare(source, out)
    next((out / "days" / "2025-09-01").glob("*.gz")).write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        read_day(out, "2025-09-01")

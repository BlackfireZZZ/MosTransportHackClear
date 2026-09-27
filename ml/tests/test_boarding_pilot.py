from pathlib import Path

from test_boarding_audit import archive

from tramflow_ml.boarding.audit import audit
from tramflow_ml.boarding.pilot import pilot, select_cohort
from tramflow_ml.boarding.records import AuditConfig


def test_frozen_pilot_preserves_rejected_and_mass(tmp_path: Path) -> None:
    run = tmp_path / "run"
    audit(archive(tmp_path / "input.zip"), run, AuditConfig(chunk_rows=2))
    cohort = select_cohort(run)
    assert cohort["frozen_before_detection"]
    report = pilot(run, tmp_path / "pilot")
    assert report["successful"] == 3
    assert report["G0"] == "passed"
    assert report["certified_stop_labels"] == 0
    assert all(c["successful_mass"] == 3 for c in report["comparisons"])


def test_rejected_weights_and_untrusted_database(tmp_path: Path) -> None:
    import csv
    import gzip
    import zipfile

    original = archive(tmp_path / "input.zip")
    src = tmp_path / "with_reject.zip"
    with zipfile.ZipFile(original) as z, zipfile.ZipFile(src, "w") as target:
        for name in z.namelist():
            value = z.read(name)
            if name == "train.csv":
                value = value.replace(b"0;5 ", b"0;1 ")
            target.writestr(name, value)
    run = tmp_path / "run"
    audit(src, run, AuditConfig())
    (run / "audit.sqlite").write_bytes(b"untrusted diagnostic index")
    out = tmp_path / "pilot"
    report = pilot(run, out)
    with gzip.open(out / "event_to_burst.csv.gz", "rt") as f:
        assert sum(int(r["unresolved_weight"]) for r in csv.DictReader(f)) == report["successful"]
    assert report["rejected"] == 1

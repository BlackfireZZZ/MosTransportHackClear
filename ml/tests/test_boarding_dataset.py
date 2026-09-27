import csv
from pathlib import Path

from test_boarding_audit import archive

from tramflow_ml.boarding.audit import audit
from tramflow_ml.boarding.dataset import export_dataset
from tramflow_ml.boarding.records import AuditConfig


def test_chronological_export_missing_not_zero(tmp_path: Path) -> None:
    run = tmp_path / "run"
    audit(archive(tmp_path / "input.zip"), run, AuditConfig())
    out = tmp_path / "dataset"
    manifest = export_dataset(run, out)
    assert manifest["stop_labels_eligible"] is False
    with (out / "diagnostic.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert sum(int(r["target_validation_count"]) for r in rows) == 3
    assert all(r["lag_24h"] == "" for r in rows)
    assert all(r["stop_id"] == "" for r in rows)
    assert all(r["forecast_origin"] <= r["event_hour"] for r in rows)


def test_future_counts_do_not_change_past_features(tmp_path: Path) -> None:
    import zipfile

    from test_boarding_audit import HEADER

    outputs = []
    for name, future_count in [("original", 1), ("mutated", 20)]:
        src = tmp_path / (name + ".zip")
        with zipfile.ZipFile(src, "w") as z:
            z.writestr(
                "train.csv",
                HEADER
                + "a;d;2025-06-01 00:00:00;1;1 трамвай;v;e\n"
                + "b;d;2025-06-02 00:00:00;1;1 трамвай;v;e\n",
            )
            z.writestr(
                "test.csv", HEADER + "c;d;2025-09-01 00:00:00;1;1 трамвай;v;e\n" * future_count
            )
        run = tmp_path / (name + "-run")
        audit(src, run, AuditConfig())
        out = tmp_path / (name + "-dataset")
        export_dataset(run, out)
        outputs.append((out / "train.csv").read_bytes())
    assert outputs[0] == outputs[1]

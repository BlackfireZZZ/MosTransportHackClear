"""Retrospective route-hour training rows with fixed-origin historical features."""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .audit import digest, verify_run, write_json


def export_dataset(run: Path, out: Path) -> dict[str, object]:
    """Rejects incomplete/changed sources and never exports inferred stop labels."""
    verify_run(run)
    if out.exists() and any(out.iterdir()):
        raise ValueError("dataset output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    with (run / "mass_ledger.csv").open() as f:
        counts = {
            (
                r["route"],
                datetime.fromisoformat(r["event_hour"] + ":00:00").replace(
                    tzinfo=ZoneInfo("Europe/Moscow")
                ),
            ): int(r["source_success"])
            for r in csv.DictReader(f)
        }
    paths = {split: out / (split + ".csv") for split in ["train", "development", "diagnostic"]}
    fieldnames = [
        "route",
        "event_hour",
        "forecast_origin",
        "target_validation_count",
        "lag_24h",
        "lag_168h",
        "lag_336h",
        "prior_day_sum",
        "prior_day_observed_hours",
        "weekday",
        "hour",
        "stop_id",
        "stop_label_eligible",
        "availability_verified",
    ]
    handles = {s: p.with_suffix(".csv.tmp").open("w") for s, p in paths.items()}
    sizes = dict.fromkeys(paths, 0)
    try:
        writers = {s: csv.writer(f, lineterminator="\n") for s, f in handles.items()}
        for w in writers.values():
            w.writerow(fieldnames)
        for (route, t), value in sorted(counts.items()):
            origin = t.replace(hour=0)
            history = [counts.get((route, origin - timedelta(hours=h))) for h in range(1, 25)]
            observed = [v for v in history if v is not None]
            split = (
                "train"
                if t.date().isoformat() < "2025-07-01"
                else "development"
                if t.date().isoformat() < "2025-09-01"
                else "diagnostic"
            )
            writers[split].writerow(
                [
                    route,
                    t.isoformat(),
                    origin.isoformat(),
                    value,
                    *[counts.get((route, t - timedelta(hours=h))) for h in [24, 168, 336]],
                    sum(observed) if observed else None,
                    len(observed),
                    t.weekday(),
                    t.hour,
                    "",
                    False,
                    False,
                ]
            )
            sizes[split] += 1
    finally:
        for f in handles.values():
            f.close()
    for p in paths.values():
        p.with_suffix(".csv.tmp").replace(p)
    manifest: dict[str, object] = {
        "schema_version": "boarding-training.v1",
        "feature_version": "route-history.v1",
        "source_manifest_sha256": digest(run / "manifest.json"),
        "timezone": "Europe/Moscow",
        "target": "validation_count",
        "unit": "event_count",
        "date_range": json.loads((run / "manifest.json").read_text())["config"],
        "forecast_origin": "civil midnight before the target day",
        "mode": "retrospective_event_time_only",
        "availability_verified": False,
        "coverage": "only source-observed route/hour cells; missing cells are not zero-filled",
        "split_unit": "whole civil day; not random events",
        "rows": sizes,
        "leakage_policy": "every historical input event_hour < forecast_origin; fixed lags >=24h",
        "diagnostic_is_blind": False,
        "stop_labels_eligible": False,
        "files": {s: {"name": p.name, "sha256": digest(p)} for s, p in paths.items()},
    }
    write_json(out / "manifest.json", manifest)
    return manifest

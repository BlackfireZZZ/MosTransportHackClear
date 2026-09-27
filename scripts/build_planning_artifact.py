"""Build public estimated spatial shares without publishing individual events."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, predictions, catalog):
    manifest_path = root / "ml/competition_submissions/current.json"
    manifest = json.loads(manifest_path.read_text())
    identity = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    approvals = json.loads((manifest_path.parent / "approved.json").read_text())[
        "approved"
    ]
    if not any(entry["manifest_sha256"] == identity for entry in approvals):
        raise ValueError("selected manifest is not approved")
    stop_run = predictions.parent / "run.json"
    stop_audit = predictions.parent / "dataset_audit.json"
    audit = json.loads(stop_audit.read_text())
    if audit["catalog_sha256"] != digest(catalog):
        raise ValueError("catalog does not match stop model source")
    source = manifest_path.parent / manifest["csv"]
    if digest(source) != manifest["csv_sha256"]:
        raise ValueError("selected CSV checksum differs")
    with source.open() as stream:
        route_rows = list(csv.DictReader(stream, delimiter=";"))
    with catalog.open() as stream:
        stops = list(csv.DictReader(stream))
    spatial = defaultdict(lambda: defaultdict(float))
    with gzip.open(predictions, "rt") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["date"])
            if not date(2025, 11, 1) <= day <= date(2025, 12, 31):
                raise ValueError("unexpected stop forecast date")
            if row["label_origin"] != "forecast_of_inferred_or_unallocated_counts":
                raise ValueError(
                    "expected inferred prediction, not actual future labels"
                )
            value = float(row["prediction"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid stop forecast")
            key = f"{row['route']}|{day.weekday()}|{row['hour']}"
            spatial[key][(row["stop_id"], row["direction"])] += float(row["prediction"])
    shares = {}
    for key, cells in spatial.items():
        total = sum(cells.values())
        shares[key] = (
            [
                [stop, direction, value / total]
                for (stop, direction), value in sorted(cells.items())
                if value > 0
            ]
            if total
            else []
        )
    payload = {
        "route_rows": route_rows,
        "shares": shares,
        "stops": stops,
        "provenance": {
            "model_version": manifest["model_version"],
            "csv_sha256": digest(source),
            "stop_predictions_sha256": digest(predictions),
            "stop_run_sha256": digest(stop_run),
            "stop_dataset_audit_sha256": digest(stop_audit),
            "stop_producer_sha256": json.loads(stop_run.read_text())["code_sha256"],
            "spatial_cutoff": "2025-11-01T00:00:00+03:00",
            "spatial_time_status": "forecast from past inferred counts; geometry retrospective",
            "catalog_sha256": digest(catalog),
            "generator_sha256": digest(Path(__file__)),
            "data_cutoff": manifest["data_cutoff"],
            "generated_at": manifest["generated_at"],
            "spatial_status": "estimated_from_inferred_stop_targets; no spatial accuracy measured",
        },
    }
    run = json.loads(stop_run.read_text())
    if run.get("mapping_method"):
        payload["provenance"].update({
            "mapping_method": run["mapping_method"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "spatial_time_status": "pre-origin counts; retrospective GTFS and transferred calendars",
        })
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    out = root / "data/planning"
    out.mkdir(exist_ok=True, parents=True)
    with (out / "forecast.json.gz").open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as target:
            target.write(raw)
    (out / "manifest.json").write_text(
        json.dumps(
            {"sha256": digest(out / "forecast.json.gz"), **payload["provenance"]},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.predictions, args.catalog)

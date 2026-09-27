"""Serialize exact independent stop forecasts; never rescale them to a route model."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from datetime import date
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, source: Path) -> None:
    run_path = source.parent / "run.json"
    audit_path = source.parent / "dataset_audit.json"
    run = json.loads(run_path.read_text())
    audit = json.loads(audit_path.read_text())
    projection = json.loads((root / "data/planning/manifest.json").read_text())
    if audit["catalog_sha256"] != projection["catalog_sha256"]:
        raise ValueError("stop catalogue differs from planning coordinates")
    identities: dict[tuple[str, str, str], list[float | None]] = {}
    versions, generated = set(), set()
    rows = 0
    with gzip.open(source, "rt") as stream:
        for row in csv.DictReader(stream):
            day = (date.fromisoformat(row["date"]) - date(2025, 11, 1)).days
            hour = int(row["hour"])
            if not 0 <= day < 61 or not 0 <= hour < 24:
                raise ValueError("stop forecast outside November–December grid")
            if row["label_origin"] != "forecast_of_inferred_or_unallocated_counts":
                raise ValueError("expected independent inferred-count predictions")
            value = float(row["prediction"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid stop prediction")
            key = (row["route"], row["stop_id"], row["direction"])
            values = identities.setdefault(key, [None] * 1464)
            index = day * 24 + hour
            if values[index] is not None:
                raise ValueError("duplicate stop prediction")
            values[index] = value
            versions.add(row["model_version"])
            generated.add(row["generated_at"])
            rows += 1
    if not identities or any(
        any(v is None for v in row) for row in identities.values()
    ):
        raise ValueError("incomplete per-stop forecast grid")
    expected_version = run["feature_version"] + ":" + run["selected"]
    if (
        versions != {expected_version}
        or run["horizon_days"] != 61
        or run["timezone"] != "Europe/Moscow"
        or run["stop_ground_truth"] is not False
    ):
        raise ValueError("stop run metadata does not describe source predictions")
    if len(versions) != 1 or len(generated) != 1:
        raise ValueError("mixed stop forecast versions")
    provenance = {
        "model_version": next(iter(versions)),
        "generated_at": next(iter(generated)),
        "stop_predictions_sha256": digest(source),
        "stop_run_sha256": digest(run_path),
        "stop_dataset_audit_sha256": digest(audit_path),
        "catalog_sha256": audit["catalog_sha256"],
        "stop_producer_sha256": run["code_sha256"],
        "generator_sha256": digest(Path(__file__)),
        "data_cutoff": "2025-10-31T23:59:59+03:00",
        "target": "inferred_validation_count",
        "spatial_status": "independent_stop_predictions; not route-normalized; no observed stop truth",
        "spatial_time_status": "counts use pre-origin history; geometry retrospective",
    }
    if run.get("mapping_method"):
        provenance.update({
            "mapping_method": run["mapping_method"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "spatial_time_status": "pre-origin counts; retrospective GTFS and transferred calendars",
        })
    payload = {
        "schema": "independent-stop-forecast.v1",
        "start_date": "2025-11-01",
        "days": 61,
        "hours_per_day": 24,
        "rows": rows,
        "provenance": provenance,
        "identities": [
            {"route": key[0], "stop_id": key[1], "direction": key[2], "values": values}
            for key, values in sorted(identities.items())
        ],
    }
    target = root / "data/planning/stop-model.json.gz"
    with target.open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            )
    (target.parent / "stop-model-manifest.json").write_text(
        json.dumps(
            {
                "schema": payload["schema"],
                "sha256": digest(target),
                "rows": rows,
                "identities": len(identities),
                **provenance,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.source)

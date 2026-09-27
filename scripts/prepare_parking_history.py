"""Prepare cutoff-bounded parking observations, never interpreted as road speeds.

Requires an existing local pandas/pyarrow environment; not imported by the model.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow

REVISION = "283c1a33466424b48aef042923c1f8e6bdff3d98"
SOURCE = "https://huggingface.co/datasets/matrosovdani/moscow-parking-occupancy"


def prepare(source, output):
    files = sorted((source / "data").glob("occupancy_2025-*.parquet"))
    expected = {f"occupancy_2025-{month:02}.parquet" for month in range(3, 11)}
    if {p.name for p in files} != expected:
        raise ValueError("Expected exactly March–October source partitions")
    original = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if str(original.time.dt.tz) != "UTC":
        raise ValueError("Unexpected source timezone")
    start = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
    cutoff = pd.Timestamp("2025-11-01", tz="Europe/Moscow")
    frame = original.loc[(original.time >= start) & (original.time < cutoff)].copy()
    if frame.empty or frame[["time", "parking_id"]].isna().any().any():
        raise ValueError("Missing primary key or empty history")
    if frame.duplicated(["time", "parking_id"]).any():
        raise ValueError("Duplicate source primary keys")
    frame["time"] = frame.time.dt.tz_convert("Europe/Moscow")
    frame["occupancy_rate_valid"] = frame.occupancy_rate.between(0, 100)
    frame["counts_valid"] = (frame.free_spaces >= 0) & (
        frame.free_handicapped_spaces >= 0
    )
    frame = frame.sort_values(["time", "parking_id"]).reset_index(drop=True)
    spots_source = source / "data" / "parking_spots.parquet"
    spots = pd.read_parquet(spots_source)
    if not set(frame.parking_id) <= set(spots.id):
        raise ValueError("Missing lot metadata")
    # Source flags inspect 2026; using them for a 2025 cutoff would leak future data.
    locations = spots[["id", "external_id", "name_ru", "latitude", "longitude"]]
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "observations.parquet", index=False)
    locations.to_parquet(output / "locations-retrospective.parquet", index=False)
    shutil.copyfile(source / "README.md", output / "SOURCE_README.md")
    output_names = [
        "observations.parquet",
        "locations-retrospective.parquet",
        "SOURCE_README.md",
    ]
    reread = pd.read_parquet(output / "observations.parquet")
    pd.testing.assert_frame_equal(frame, reread)
    assert reread.time.max() < cutoff and reread.time.min() >= start
    summary = {
        "schema": "parking-history.v1",
        "feature_version": None,
        "status": "historical_parking_proxy_not_integrated_into_model",
        "source": SOURCE,
        "source_revision": REVISION,
        "license": "CC-BY-4.0",
        "attribution": "Danil Matrosov / ParkOut; upstream Moscow Department of Transport public parking data",
        "timezone": "Europe/Moscow",
        "requested_period": ["2025-01-01", "2025-10-31"],
        "actual_period": [frame.time.min().isoformat(), frame.time.max().isoformat()],
        "rows": len(frame),
        "source_rows": len(original),
        "cutoff_excluded_rows": len(original) - len(frame),
        "parking_lots": int(frame.parking_id.nunique()),
        "snapshot_instants": int(frame.time.nunique()),
        "days": int(frame.time.dt.date.nunique()),
        "hour_buckets": int(frame.time.dt.floor("h").nunique()),
        "invalid_occupancy_rates": int((~frame.occupancy_rate_valid).sum()),
        "invalid_counts": int((~frame.counts_valid).sum()),
        "always_zero_free_lots_in_retained_history": int(
            frame.groupby("parking_id").free_spaces.max().eq(0).sum()
        ),
        "rows_by_month": {
            str(k): int(v)
            for k, v in frame.groupby(frame.time.dt.strftime("%Y-%m")).size().items()
        },
        "runtime": {"pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "source_files": {
            p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in [*files, spots_source, source / "README.md"]
        },
        "files": {
            name: {
                "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                "bytes": (output / name).stat().st_size,
            }
            for name in output_names
        },
        "leakage_rule": "Every feature must use observations before its forecast origin. Later metadata flags/capacities excluded. Location metadata is retrospective, not a proven 2025 vintage.",
        "limitations": [
            "Parking occupancy is a mobility proxy, not road congestion or speed.",
            "No January–March Moscow-time coverage. Gaps are not zero occupancy.",
            "Reported API snapshots include stale sensors; zero free spaces is not reliable evidence of full occupancy.",
            "Invalid rates retained with quality flags; do not silently clip or use them.",
            "Historical capacity used by upstream occupancy_rate is not independently verified; latest capacity is not used here.",
            "No measured improvement to tram-flow forecasts established.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("source_files", "files")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)

"""Complete cutoff-indexed route features and provenance-preserving stop feature exports."""

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.boarding.partitions import write_frame
from tramflow_ml.competition import (
    complete_labels,
    feature_frame,
    load_labels,
    make_training_frame,
)
from tramflow_ml.route_models.features import OFF, WORK

VERSION = "complete-features.v1"
ORIGINS = tuple(date(2025, month, 1) for month in range(4, 10))
KEYS = ["route", "date", "hour"]
STOP_KEYS = ["route", "direction", "stop_id"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def calendar_features(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Civil Moscow calendar; no observation or target-derived field is read."""
    result = frame.copy()
    dates = pd.to_datetime(result[date_column], format="%Y-%m-%d", errors="raise")
    hours = pd.to_numeric(result.hour, errors="raise")
    if dates.isna().any() or not hours.between(0, 23).all() or (hours % 1 != 0).any():
        raise ValueError("nonempty civil dates and integer hours 0..23 required")
    result["weekday"] = dates.dt.dayofweek
    result["month"] = dates.dt.month
    result["day_of_year"] = dates.dt.dayofyear
    result["day_of_month"] = dates.dt.day
    result["is_summer"] = dates.dt.month.isin([6, 7, 8]).astype(int)
    days = dates.dt.date
    result["calendar_exception"] = days.isin(OFF | WORK).astype(int)
    result["calendar_day_type"] = [
        0 if day in WORK or (day.weekday() < 5 and day not in OFF)
        else 1 if day.weekday() == 5 and day not in OFF else 2 for day in days
    ]
    for name, values, period in (
        ("hour", hours, 24), ("weekday", result.weekday, 7),
        ("year", result.day_of_year - 1, 365),
    ):
        result[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        result[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return result


def route_fold(labels: pd.DataFrame, origin: date) -> pd.DataFrame:
    """Target columns are attached after all predictors have been frozen at origin."""
    features = feature_frame(labels, origin)
    features["origin"] = origin.isoformat()
    features["feature_cutoff"] = (origin - timedelta(days=1)).isoformat()
    features = calendar_features(features)
    end = origin + timedelta(days=60)
    truth = labels.loc[(labels.date >= origin) & (labels.date <= end), KEYS + ["boardings"]]
    if len(truth) != len(features):
        raise ValueError("complete 61-day labels required for evaluation")
    result = features.merge(truth, on=KEYS, how="left", validate="one_to_one")
    if result.boardings.isna().any():
        raise ValueError("evaluation label keys do not match forecast grid")
    return result


def catalog_features(catalog: pd.DataFrame) -> pd.DataFrame:
    """Unique identities only; multiple sequence candidates remain an explicit range."""
    required = {*STOP_KEYS, "lat", "lon"}
    if not required.issubset(catalog.columns):
        raise ValueError("stop catalog requires explicit identity and coordinates")
    if catalog[STOP_KEYS].isna().any().any():
        raise ValueError("stop catalog identity cannot be null")
    for column, low, high in (("lat", -90, 90), ("lon", -180, 180)):
        numbers = pd.to_numeric(catalog[column], errors="coerce")
        if (catalog[column].notna() & (~np.isfinite(numbers) | ~numbers.between(low, high))).any():
            raise ValueError("invalid stop geometry")
    if (catalog.groupby(STOP_KEYS)[["lat", "lon"]].nunique(dropna=False) > 1).any().any():
        raise ValueError("conflicting geometry for one stop identity")
    geometry = catalog.groupby(STOP_KEYS, as_index=False, dropna=False).agg(
        lat=("lat", "first"), lon=("lon", "first"), catalog_rows=("stop_id", "size")
    )
    if "sequence" in catalog:
        sequences = catalog.groupby(STOP_KEYS, as_index=False).agg(
            sequence_min=("sequence", "min"), sequence_max=("sequence", "max"),
        )
        geometry = geometry.merge(sequences, on=STOP_KEYS, validate="one_to_one")
        geometry["sequence_ambiguous"] = geometry.sequence_min.ne(geometry.sequence_max)
    return geometry


def enrich_stops(frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Preserve soft targets, exclusion flags and original payment-hour mass exactly."""
    result = frame.copy()
    values = pd.to_numeric(result.expected_count, errors="raise")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("stop targets must be finite nonnegative")
    stamps = pd.to_datetime(result.event_hour, utc=True).dt.tz_convert("Europe/Moscow")
    if (
        stamps.isna().any() or stamps.dt.strftime("%Y-%m-%d").ne(result.day).any()
        or stamps.dt.hour.ne(result.hour).any() or stamps.dt.minute.ne(0).any()
        or stamps.dt.second.ne(0).any() or stamps.dt.microsecond.ne(0).any()
        or stamps.dt.nanosecond.ne(0).any()
    ):
        raise ValueError("stop target escapes original Moscow hour")
    result = calendar_features(result, "day")
    result = result.merge(catalog_features(catalog), on=STOP_KEYS, how="left",
                          validate="many_to_one")
    result["geometry_available"] = result.lat.notna() & result.lon.notna()
    if len(result) != len(frame) or not np.isclose(
        result.expected_count.sum(), frame.expected_count.sum(), atol=1e-6, rtol=1e-12,
    ):
        raise ValueError("stop enrichment changed event mass")
    return result


def export_routes(archive: Path, proof_path: Path, out: Path) -> dict[str, Any]:
    """Require checksum-bound independent raw reconciliation before dense zero filling."""
    if out.exists():
        raise ValueError("new output directory required")
    proof = json.loads(proof_path.read_text())
    archive_hash = digest(archive)
    raw = load_labels(archive)
    raw = raw.loc[raw.date.between(date(2025, 1, 1), date(2025, 10, 31))]
    legacy_ok = (
        proof.get("union_all_labels_exact_match") is True
        and proof.get("keys") == len(raw)
        and proof.get("target_sum") == int(raw.boardings.sum())
    )
    canonical_ok = (
        proof.get("schema") == "route-hour-reconciliation.v1"
        and proof.get("missing_keys_are_zero_in_supplied_raw") is True
        and proof.get("period") == ["2025-01-01", "2025-10-31"]
        and proof.get("raw_keys") == proof.get("label_keys") == len(raw)
    )
    if proof.get("archive_sha256") != archive_hash or not (legacy_ok or canonical_ok):
        raise ValueError("independent reconciliation proof mismatch")
    labels = complete_labels(raw, date(2025, 1, 1), date(2025, 10, 31), missing_as_zero=True)
    out.mkdir(parents=True)
    files: dict[str, Any] = {}

    def save(name: str, frame: pd.DataFrame) -> None:
        path = out / name
        write_frame(path, frame)
        files[name] = {"rows": len(frame), "sha256": digest(path), "columns": list(frame.columns)}

    save("route-hour-labels.csv.gz", labels)
    save("route-hour-calendar.csv.gz", calendar_features(labels))
    for origin in ORIGINS:
        save(f"evaluation-{origin}.csv.gz", route_fold(labels, origin))
    train = make_training_frame(labels, date(2025, 10, 31))
    train["origin"] = (
        pd.to_datetime(train.date) - pd.to_timedelta(train.lead_day.astype(int) - 1, unit="D")
    ).dt.strftime("%Y-%m-%d")
    train["feature_cutoff"] = (
        pd.to_datetime(train.origin) - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    save("training-direct-61-day.csv.gz", calendar_features(train))
    future = calendar_features(feature_frame(labels, date(2025, 11, 1)))
    future["origin"] = "2025-11-01"
    future["feature_cutoff"] = "2025-10-31"
    save("submission-features.csv.gz", future)
    manifest = {
        "schema": VERSION, "complete": True, "files": files,
        "archive_sha256": archive_hash, "reconciliation_sha256": digest(proof_path),
        "date_range": ["2025-01-01", "2025-10-31"], "forecast_range": ["2025-11-01", "2025-12-31"],
        "timezone": "Europe/Moscow", "raw_timezone_assumption": "organizer raw unspecified",
        "target": "successful_validation_count_route_date_hour",
        "target_mass": int(labels.boardings.sum()),
        "feature_version": VERSION, "implementation_sha256": digest(Path(__file__)),
        "leakage_policy": "history strictly before origin; target attached afterwards",
        "weather_policy": "future actual weather excluded; independent retrospective table only",
        "diagnostic_is_blind": False,
        "evidence_level": "inspected labels and checksum-bound raw reconciliation",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_stops(source: Path, out: Path) -> dict[str, Any]:
    """Enrich an explicitly selected mapping run; never silently substitute another version."""
    if out.exists():
        raise ValueError("new output directory required")
    manifest_path = source / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if (source_manifest.get("complete") is not True
            or source_manifest.get("schema_version") != "boarding-real-training.v1"):
        raise ValueError("completed mapping source required")
    if digest(source / "stop_catalog.csv") != source_manifest.get("catalog_sha256"):
        raise ValueError("mapping catalog lacks matching source checksum")
    catalog = pd.read_csv(source / "stop_catalog.csv", dtype={"stop_id": str, "direction": str})
    catalog["route"] = catalog.route.astype(int)
    out.mkdir(parents=True)
    files = {}
    for split in ("train", "development", "diagnostic"):
        spec = source_manifest["files"][f"soft_{split}"]
        path = source / spec["name"]
        if path.name != spec["name"] or digest(path) != spec["sha256"]:
            raise ValueError("mapping source file changed or unsafe filename")
        frame = pd.read_csv(path, dtype={"stop_id": str, "direction": str})
        frame["route"] = frame.route.astype(int)
        result = enrich_stops(frame, catalog)
        output = out / f"stop-features-{split}.csv.gz"
        write_frame(output, result)
        files[split] = {
            "name": output.name, "rows": len(result), "sha256": digest(output),
            "mass": float(result.expected_count.sum()),
            "eligible_mass": float(result.loc[result.training_eligible, "expected_count"].sum()),
            "missing_geometry_rows": int((~result.geometry_available).sum()),
        }
    result_manifest = {
        "schema": VERSION + ".stops", "complete": True, "files": files,
        "source_manifest_sha256": digest(manifest_path),
        "catalog_sha256": digest(source / "stop_catalog.csv"),
        "feature_version": VERSION, "implementation_sha256": digest(Path(__file__)),
        "timezone": "Europe/Moscow", "target": "inferred_stop_direction_hour_boardings",
        "observed_stop_labels": False, "point_in_time_sources_verified": False,
        "geometry_use": "retrospective static source; no historical availability claim",
        "source_provenance": source_manifest,
    }
    (out / "manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n")
    return result_manifest



def export_stop_model_features(data: Any, out: Path) -> dict[str, Any]:
    """All eligible direct-horizon examples; sampling belongs only to model fitting."""
    from tramflow_ml.stop_models import features

    if out.exists():
        raise ValueError("new output directory required")
    out.mkdir(parents=True)
    identities = data.identities
    outer = len(data.train)
    first = date(2025, 1, 1)
    files: dict[str, Any] = {}
    for origin in [*range(59, outer, 30), outer]:
        forecast = origin == outer
        days = 61 if forecast else min(61, outer - origin)
        frame = features(data.train, identities, origin, days)
        count = len(identities)
        indices = np.tile(np.repeat(np.arange(count), 24), days)
        frame["stop_id"] = identities.stop_id.to_numpy()[indices]
        frame["origin"] = str(first + timedelta(days=origin))
        frame["feature_cutoff"] = str(first + timedelta(days=origin - 1))
        frame["target_date"] = np.repeat(
            pd.date_range(first + timedelta(days=origin), periods=days).strftime("%Y-%m-%d"),
            count * 24,
        )
        if not forecast:
            frame["expected_count_target"] = data.train[origin:origin + days].ravel()
            frame = frame.loc[frame.expected_count_target.notna()].copy()
        name = ("forecast" if forecast else "training") + f"-{origin:03d}.csv.gz"
        path = out / name
        write_frame(path, frame)
        files[name] = {"rows": len(frame), "sha256": digest(path),
                       "origin": str(first + timedelta(days=origin)), "horizon_days": days}
        print(f"exported {name}: {len(frame)} rows", flush=True)
    identity_path = out / "identities.csv"
    identities.to_csv(identity_path, index=False)
    manifest = {
        "schema": "stop-direct-feature-matrix.v1", "complete": True,
        "feature_version": VERSION, "files": files,
        "identities_sha256": digest(identity_path), "source": data.audit,
        "implementation_sha256": digest(Path(__file__)),
        "model_features_sha256": digest(Path(features.__code__.co_filename)),
        "timezone": "Europe/Moscow", "training_sampling": "none; all eligible cells exported",
        "target": "pseudo stop count or explicit unallocated route mass",
        "leakage_policy": "dynamic history strictly before origin; no target-hour route totals",
        "catalog_availability": "retrospective static source; not a point-in-time assertion",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    routes = sub.add_parser("routes")
    routes.add_argument("--archive", type=Path, required=True)
    routes.add_argument("--reconciliation", type=Path, required=True)
    routes.add_argument("--out", type=Path, required=True)
    stops = sub.add_parser("stops")
    stops.add_argument("--source", type=Path, required=True)
    stops.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = (export_routes(args.archive, args.reconciliation, args.out) if args.mode == "routes"
              else export_stops(args.source, args.out))
    print(json.dumps({k: v for k, v in result.items() if k != "source_provenance"}, indent=2))


if __name__ == "__main__":
    main()

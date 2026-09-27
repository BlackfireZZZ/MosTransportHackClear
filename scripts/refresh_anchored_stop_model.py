"""Refresh the fixed incumbent against checksum-bound retrospective stop targets."""

import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tramflow_ml import stop_models
from tramflow_ml.stop_models import IDENTITY, aggregate, calendar_stop_profile, evaluate
from tramflow_ml.stop_refinement import load_verified

FEATURE_VERSION = "direct-stop-calendar-history.anchored-v3"
SELECTED = "calendar_weekday_blend"
WINDOWS = [(origin, 61) for origin in (120, 181, 243)] + [
    (origin, 1) for origin in (120, 127, 134, 181, 188, 195, 243, 250, 257)
]
TOLERANCE = 1e-8


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_pair(old, new):
    if not old.identities[IDENTITY].equals(new.identities[IDENTITY]):
        raise ValueError("identity universes differ")
    if not np.array_equal(np.isfinite(old.train), np.isfinite(new.train)):
        raise ValueError("training eligibility differs")
    a = aggregate(old.values, old.identities, 0).prediction.to_numpy()
    b = aggregate(new.values, new.identities, 0).prediction.to_numpy()
    if not np.allclose(a, b, atol=TOLERANCE, rtol=0):
        raise ValueError("mapped route counts differ")


def forecast_checked(data, origin, days):
    prediction = calendar_stop_profile(data.train, origin, days, blend=True)
    poisoned = data.train.copy()
    poisoned[origin:] = 1e15
    check = calendar_stop_profile(poisoned, origin, days, blend=True)
    if not np.array_equal(prediction, check):
        raise ValueError("future counts influence forecast")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("invalid predictions")
    return prediction


def compare_routes(old, new, old_prediction, new_prediction, origin):
    a = aggregate(old_prediction, old.identities, origin).prediction.to_numpy()
    b = aggregate(new_prediction, new.identities, origin).prediction.to_numpy()
    delta = float(np.max(np.abs(a - b)))
    if delta > TOLERANCE:
        raise ValueError(f"route forecast changed by {delta}")
    return delta


def load(source, route_export):
    data = load_verified(source, route_export)
    manifest = json.loads((source / "manifest.json").read_text())
    data.audit.update({
        "catalog_sha256": digest(source / "stop_catalog.csv"),
        "source_file_hashes": {
            entry["name"]: entry["sha256"] for key, entry in manifest["files"].items()
            if key.startswith("soft_")
        },
        "source_dates": ["2025-01-01", "2025-10-31"],
        "timezone": "Europe/Moscow",
        "feature_version": FEATURE_VERSION,
        "mapping_policy": manifest.get("anchor_policy"),
        "spatial_status": "retrospective inference; not observed stop ground truth",
    })
    return data


def run(old_source, source, route_export, output, report):
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    plan = report / "experiment-plan.md"
    if not plan.exists():
        raise ValueError("preregistered experiment plan required")
    plan_hash = digest(plan)
    old, new = load(old_source, route_export), load(source, route_export)
    verify_pair(old, new)
    results = []
    for origin, days in WINDOWS:
        a, b = forecast_checked(old, origin, days), forecast_checked(new, origin, days)
        delta = compare_routes(old, new, a, b, origin)
        old_metrics, new_metrics = evaluate(new, a, origin), evaluate(new, b, origin)
        score_delta = abs(old_metrics["route_score"] - new_metrics["route_score"])
        old_route = aggregate(a, old.identities, origin).prediction.to_numpy()
        new_route = aggregate(b, new.identities, origin).prediction.to_numpy()
        rounded_delta = np.abs(np.rint(old_route) - np.rint(new_route))
        changed = rounded_delta > 0
        half_integer = np.floor(old_route) + .5
        near_half = ((np.abs(old_route - half_integer) <= TOLERANCE)
                     & (np.abs(new_route - half_integer) <= TOLERANCE))
        if (rounded_delta > 1).any() or (changed & ~near_half).any():
            raise ValueError("route rounding changed away from half-integer ties")
        actual_mass = float(new.values[origin:origin + days].sum())
        quantization_bound = float(changed.sum() / actual_mass)
        if score_delta > quantization_bound + 1e-12:
            raise ValueError("route score change exceeds measured quantization bound")
        item = {"origin": str(date(2025, 1, 1) + timedelta(days=origin)),
                "days": days, "old_mapping_against_new_targets": old_metrics,
                "anchored_mapping_against_new_targets": new_metrics,
                "route_prediction_max_abs_difference": delta,
                "route_score_abs_difference": score_delta,
                "half_tie_changed_cells": int(changed.sum()),
                "quantization_score_bound": quantization_bound,
                "actual_mass": actual_mass,
                "route_absolute_error_delta": float(
                    (old_metrics["route_score"] - new_metrics["route_score"]) * actual_mass),
                "future_count_poison_invariant": True}
        results.append(item)
        print(item["origin"], days, new_metrics["route_score"], delta, flush=True)
    a, prediction = forecast_checked(old, 304, 61), forecast_checked(new, 304, 61)
    final_delta = compare_routes(old, new, a, prediction, 304)
    unknown = new.identities.stop_id.astype(str).str.startswith("unallocated").to_numpy()
    share = float(prediction[:, unknown].sum() / prediction.sum())
    if prediction.size != 854976 or share >= .3:
        raise ValueError("final grid or unknown-share acceptance failed")
    generated = datetime.now(UTC).isoformat()
    count = len(new.identities)
    stop = pd.DataFrame({
        "route": np.tile(np.repeat(new.identities.route, 24), 61),
        "direction": np.tile(np.repeat(new.identities.direction, 24), 61),
        "stop_id": np.tile(np.repeat(new.identities.stop_id, 24), 61),
        "hour": np.tile(np.arange(24), 61 * count),
        "date": np.repeat(pd.date_range("2025-11-01", periods=61).date, count * 24),
        "prediction": prediction.ravel(),
        "label_origin": "forecast_of_inferred_or_unallocated_counts",
        "model_version": FEATURE_VERSION + ":" + SELECTED,
        "generated_at": generated,
        "lower_bound": np.nan, "upper_bound": np.nan,
        "uncertainty_status": "unavailable",
    })
    stop.to_csv(output / "stop_predictions.csv.gz", index=False,
                compression={"method": "gzip", "mtime": 0})
    route = aggregate(prediction, new.identities, 304)
    route["prediction"] = np.rint(route.prediction).astype(int)
    route.to_csv(output / "submission.csv", sep=";", index=False)
    write_json(output / "dataset_audit.json", new.audit)
    metadata = {
        "code_sha256": digest(Path(stop_models.__file__)),
        "refresh_script_sha256": digest(Path(__file__)),
        "config_sha256": hashlib.sha256(json.dumps({"selected": SELECTED,
            "windows": WINDOWS, "blend": True}, sort_keys=True).encode()).hexdigest(),
        "experiment_plan_sha256": plan_hash,
        "source_manifest_sha256": new.audit["source_manifest_sha256"],
        "baseline_source_manifest_sha256": old.audit["source_manifest_sha256"],
        "submission_sha256": digest(output / "submission.csv"),
        "stop_predictions_sha256": digest(output / "stop_predictions.csv.gz"),
        "selected": SELECTED, "selection": "fixed incumbent; no model search",
        "diagnostic_blind": False, "timezone": "Europe/Moscow", "seed": 20260927,
        "feature_version": FEATURE_VERSION, "horizon_days": 61,
        "mapping_method": "duty-calendar-clock.first-stop-local-anchors-transfer.v3",
        "mapping_temporal_caveat": "retrospective GTFS geometry and nearest same-duty/day-type calendar transfer; not historically available observations",
        "generated_at": generated, "data_cutoff": "2025-10-31T23:59:59+03:00",
        "uncertainty": "unavailable; point forecasts only", "stop_ground_truth": False,
        "aggregate_constraint": "none; raw independent stop sums",
        "leakage": "count history strictly before each origin; future poison invariant; geometry retrospective",
    }
    write_json(output / "run.json", metadata)
    summary = {
        "acceptance_passed": True, "stop_rows": prediction.size,
        "identities": count, "final_unknown_share": share,
        "baseline_unknown_share": float(a[:, unknown].sum() / a.sum()),
        "final_total": float(prediction.sum()),
        "final_route_max_abs_difference": final_delta,
        "all_finite_nonnegative": True, "future_count_poison_invariant": True,
        "stop_ground_truth": False, "timezone": "Europe/Moscow",
        "run": metadata, "new_dataset": new.audit, "old_dataset": old.audit,
        "windows": results,
    }
    write_json(output / "scores.json", results)
    write_json(report / "results.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      {"run", "new_dataset", "old_dataset", "windows"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-source", "source", "route-export", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.old_source, args.source, args.route_export, args.output, args.report)

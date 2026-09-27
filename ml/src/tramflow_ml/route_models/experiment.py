"""CLI for reproducible offline comparison, training and route artifact generation."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from tramflow_ml.route_models.artifact import MOSCOW, RouteForecast, RoutePoint
from tramflow_ml.route_models.data import (
    COVERAGE_POLICY,
    FEATURE_VERSION,
    ROUTES,
    FloatArray,
    History,
    civil_date,
    day_index,
    read_labels,
)
from tramflow_ml.route_models.features import CALENDAR_ISSUED, CALENDAR_SOURCE
from tramflow_ml.route_models.models import CANDIDATES, Candidate, Trained, feature_names, fit

DEVELOPMENT_ORIGINS = ("2025-04-01", "2025-06-01", "2025-08-01")
CONFIRMATION_ORIGIN = "2025-09-01"


def metric(actual: FloatArray, predicted: FloatArray) -> dict[str, Any]:
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("metric requires matching finite arrays")
    if (actual < 0).any() or (predicted < 0).any() or not np.isfinite(actual).all():
        raise ValueError("metric requires nonnegative finite counts")
    total, error = float(actual.sum()), float(np.abs(actual - predicted).sum())
    wape = error / total if total else None
    return {
        "actual": total,
        "absolute_error": error,
        "rows": actual.size,
        "mae": error / actual.size if actual.size else None,
        "wape": wape,
        "score": max(0, 1 - wape) if wape is not None else None,
    }


def slices(actual: FloatArray, prediction: FloatArray) -> dict[str, Any]:
    return {
        "overall": metric(actual, prediction),
        "routes": {
            str(route): metric(actual[:, i], prediction[:, i]) for i, route in enumerate(ROUTES)
        },
        "lead": {
            name: metric(actual[a:b], prediction[a:b])
            for name, a, b in (
                ("day1", 0, 1),
                ("days1_31", 0, 31),
                ("days32_61", 31, 61),
            )
        },
        "peak_hours": metric(
            actual[:, :, [7, 8, 9, 16, 17, 18, 19]], prediction[:, :, [7, 8, 9, 16, 17, 18, 19]]
        ),
    }


def environment() -> dict[str, str]:
    packages = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "catboost"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return packages


def compare(
    history: History, output: Path, candidates: list[Candidate], origins: tuple[str, ...]
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for label in origins:
        origin = day_index(datetime.fromisoformat(label).date())
        if origin + 61 > len(history.values):
            raise ValueError("evaluation needs a complete observed 61-day horizon")
        truth = history.values[origin : origin + 61]
        for candidate in candidates:
            started = time.monotonic()
            model = fit(history, origin, candidate)
            prediction = model.predict()
            entry: dict[str, Any] = {
                "origin": label,
                "candidate": candidate.name,
                "metrics": slices(truth, prediction),
                "seconds": round(time.monotonic() - started, 3),
            }
            if label == CONFIRMATION_ORIGIN:
                entry["october_confirmation"] = metric(truth[30:61], prediction[30:61])
            results.append(entry)
            print(
                json.dumps(
                    {
                        "origin": label,
                        "candidate": candidate.name,
                        "score": entry["metrics"]["overall"]["score"],
                        "seconds": entry["seconds"],
                    }
                ),
                flush=True,
            )
    ranking = []
    for candidate in candidates:
        entries = [r for r in results if r["candidate"] == candidate.name]
        error = sum(r["metrics"]["overall"]["absolute_error"] for r in entries)
        total = sum(r["metrics"]["overall"]["actual"] for r in entries)
        gains = []
        for entry in entries:
            baseline = next(
                (
                    r
                    for r in results
                    if r["candidate"] == "week1" and r["origin"] == entry["origin"]
                ),
                None,
            )
            if baseline is not None:
                gains.append(
                    entry["metrics"]["overall"]["score"] - baseline["metrics"]["overall"]["score"]
                )
        ranking.append(
            {
                "candidate": candidate.name,
                "pooled_score": max(0, 1 - error / total),
                "gains_vs_week1": gains,
                "passes_declared_gate": len(gains) == len(origins)
                and all(gain >= 0.01 for gain in gains),
            }
        )
    ranking.sort(key=lambda r: r["pooled_score"], reverse=True)
    eligible = [r for r in ranking if r["passes_declared_gate"]]
    chosen = eligible[0] if eligible else ranking[0]
    result: dict[str, Any] = {
        "schema_version": "route-experiments.v1",
        "source_hash": history.source_hash,
        "feature_version": FEATURE_VERSION,
        "coverage_policy": COVERAGE_POLICY,
        "availability_policy": "retrospective-perfect-delivery-assumption",
        "timezone": "Europe/Moscow",
        "environment": environment(),
        "seed": 42,
        "threads": 4,
        "origins": origins,
        "horizon_days": 61,
        "calendar_source": CALENDAR_SOURCE,
        "calendar_issued": str(CALENDAR_ISSUED),
        "configs": [asdict(c) for c in candidates],
        "results": results,
        "ranking": ranking,
        "selected": chosen["candidate"],
        "quality_gate": chosen["passes_declared_gate"],
        "limits": [
            "folds may overlap",
            "confirmation was previously inspected in EDA",
            "August development overlaps September; only October is post-selection confirmation",
            "route5 zero prior has no positive support",
            "no live availability data",
            "selection is experimental, not production promotion",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def build_artifact(model: Trained, generated_at: datetime) -> RouteForecast:
    values = model.predict()
    origin = datetime.combine(civil_date(model.origin), datetime.min.time(), tzinfo=MOSCOW)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    return RouteForecast(
        run_id="route-" + model.candidate.name + "-" + digest[:12],
        dataset_id="organizer-2025-jan-oct",
        source_hash=model.history.source_hash,
        source_version="organizer-labels-2025.v1",
        feature_version=FEATURE_VERSION,
        model_version=model.candidate.name + "-seed42-" + digest[:12],
        forecast_origin=origin,
        data_cutoff=origin,
        generated_at=generated_at,
        points=tuple(
            RoutePoint(
                route=route,
                bucket_start=origin + timedelta(days=d, hours=h),
                predicted=float(values[d, i, h]),
            )
            for d in range(61)
            for i, route in enumerate(ROUTES)
            for h in range(24)
        ),
    )


def train(history: History, candidate: Candidate, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    model = fit(history, len(history.values), candidate)
    model.save(output)
    restored = Trained.load(output)
    if not np.array_equal(model.predict(), restored.predict()):
        raise ValueError("saved model changes predictions")
    artifact = build_artifact(restored, datetime.now(tz=MOSCOW))
    (output / "forecast.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    artifact.write_submission(output / "submission.csv")
    importance = None
    if candidate.engine == "cat":
        importance = dict(
            zip(
                feature_names(candidate),
                model.estimator.get_feature_importance().tolist(),
                strict=True,
            )
        )
    result: dict[str, Any] = {
        "candidate": asdict(candidate),
        "source_hash": history.source_hash,
        "feature_version": FEATURE_VERSION,
        "environment": environment(),
        "origin": artifact.forecast_origin.isoformat(),
        "generated_at": artifact.generated_at.isoformat(),
        "rows": len(artifact.points),
        "reload_predictions_equal": True,
        "feature_importance": importance,
        "seconds": round(time.monotonic() - started, 3),
        "status": "experimental-not-published",
        "hashes": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file():
            result["hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="tramflow-route-ml")
    parser.add_argument("command", choices=("compare", "train", "predict"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidates", default=",".join(c.name for c in CANDIDATES))
    parser.add_argument("--origins", default=",".join(DEVELOPMENT_ORIGINS))
    parser.add_argument("--candidate", choices=[c.name for c in CANDIDATES], default="cat_native")
    args = parser.parse_args()
    if args.command == "predict":
        if args.model is None:
            parser.error("predict requires --model")
        artifact = build_artifact(Trained.load(args.model), datetime.now(tz=MOSCOW))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return
    if args.archive is None:
        parser.error("compare/train requires --archive")
    history = read_labels(args.archive)
    if args.command == "compare":
        names = args.candidates.split(",")
        known = {c.name: c for c in CANDIDATES}
        if len(names) != len(set(names)) or set(names) - known.keys():
            parser.error("candidates must be unique known names")
        result = compare(
            history, args.output, [known[name] for name in names], tuple(args.origins.split(","))
        )
        print(json.dumps({"selected": result["selected"], "quality_gate": result["quality_gate"]}))
    else:
        candidate = next(c for c in CANDIDATES if c.name == args.candidate)
        print(json.dumps(train(history, candidate, args.output), indent=2))


if __name__ == "__main__":
    main()

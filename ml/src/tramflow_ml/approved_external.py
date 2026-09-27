"""Cutoff-audited, offline source-family challengers to the approved route-hour model."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from datetime import date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.competition import (
    complete_labels,
    feature_frame,
    load_labels,
    make_training_frame,
    rounded,
    score,
    validate_submission,
    write_submission,
)
from tramflow_ml.competition_cli import _verified_proof
from tramflow_ml.competition_hybrid import (
    CORRECTION_CAP,
    HYBRID_CONFIG,
    _daily,
    _origin_dates,
    hybrid_predict,
)
from tramflow_ml.competition_profile import robust_hour_shape_predict
from tramflow_ml.issued_weather import load_issued_weather
from tramflow_ml.route_models.features import OFF, WORK
from tramflow_ml.weather import load_weather

SOURCES = ("calendar", "weather", "traffic", "events")
ORIGINS = (date(2025, 5, 1), date(2025, 7, 1), date(2025, 9, 1))
CALIBRATION_ORIGIN = ORIGINS[0]
HOLDOUT_ORIGINS = ORIGINS[1:]
WEIGHT_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)
MIN_GAIN = 0.0005
MAX_SEGMENT_LOSS = 0.005
SOURCE_FIELDS = {
    "calendar": ("days_to_off", "days_since_off", "adjacent_off"),
    "weather": ("temperature_28", "precipitation_28", "weather_support"),
    "traffic": ("congestion_56", "traffic_support"),
    "events": ("incident_28", "closure_28", "event_support"),
}
INELIGIBLE = {
    "weather": "Open-Meteo archive is retrospectively revised; no 2025 vintage",
    "traffic": "2026 retrieval does not prove 2025 content version",
    "events": "2026 retrieval does not prove 2025 content version; KudaGo unavailable",
}
SWITCH_WEIGHT = 0.05
MAX_SWITCH_LOSS = 0.003


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _posts(path: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    frame = pd.DataFrame.from_records(records)
    if frame.empty or not {"published_at", "edited", "congestion_score", "incident_notice",
                           "closure_notice"} <= set(frame.columns):
        raise ValueError("incomplete traffic post contract")
    stamps = pd.to_datetime(frame.published_at, utc=True).dt.tz_convert("Europe/Moscow")
    frame["date"] = stamps.dt.date
    if frame.edited.isna().any():
        raise ValueError("unknown post edit state")
    return frame.loc[~frame.edited.astype(bool)].copy()


def source_inputs(
    daily: pd.DataFrame, origin: date, family: str,
    weather: pd.DataFrame, posts: pd.DataFrame, *, weather_embargo_days: int = 2,
) -> pd.DataFrame:
    """Every dynamic observation is strictly earlier than its simulated origin."""
    if (family not in SOURCES or daily.empty or (daily.date < origin).any()
            or weather_embargo_days < 1):
        raise ValueError("invalid branch or target-origin boundary")
    result = daily[["route", "anchor_day", "lead_day"]].copy().reset_index(drop=True)
    result["route"] = result.route.astype(int).astype(str)
    if family == "calendar":
        exceptions = sorted(OFF | WORK)
        days = daily.date.to_list()
        result["days_to_off"] = [min(((special - day).days for special in exceptions
                                       if special >= day), default=90) for day in days]
        result["days_since_off"] = [min(((day - special).days for special in exceptions
                                          if special <= day), default=90) for day in days]
        result["adjacent_off"] = [int(day - timedelta(days=1) in OFF)
                                   + int(day + timedelta(days=1) in OFF) for day in days]
    elif family == "weather":
        past = weather.loc[(weather.date <= origin - timedelta(days=weather_embargo_days))
                           & (weather.date >= origin - timedelta(days=28))]
        grouped = past.groupby("route", as_index=False).agg(
            temperature_28=("temperature_2m", "mean"),
            precipitation_28=("precipitation", "mean"),
            weather_support=("date", "nunique"),
        )
        joined = daily[["route"]].reset_index(drop=True).merge(
            grouped, on="route", how="left", validate="many_to_one",
        )
        for field in SOURCE_FIELDS[family]:
            result[field] = joined[field].fillna(-999 if field != "weather_support" else 0)
    else:
        days = 56 if family == "traffic" else 28
        past = posts.loc[(posts.date < origin)
                         & (posts.date >= origin - timedelta(days=days))]
        if family == "traffic":
            observed = pd.to_numeric(past.congestion_score, errors="coerce").dropna()
            result["congestion_56"] = float(observed.mean()) if len(observed) else -1.0
            result["traffic_support"] = len(observed)
        else:
            result["incident_28"] = int(past.incident_notice.sum())
            result["closure_28"] = int(past.closure_notice.sum())
            result["event_support"] = len(past)
    return result[["route", "anchor_day", "lead_day", *SOURCE_FIELDS[family]]]


def _fit_branch(
    history: pd.DataFrame, future: pd.DataFrame, family: str,
    weather: pd.DataFrame, posts: pd.DataFrame, *, weather_embargo_days: int = 2,
) -> np.ndarray:
    from catboost import CatBoostRegressor  # type: ignore[import-not-found,import-untyped]

    origin = min(future.date)
    if (history.date >= origin).any():
        raise ValueError("branch history reaches forecast origin")
    train = make_training_frame(history, origin - timedelta(days=1))
    train_origins = _origin_dates(train)
    anchors = np.empty(len(train), dtype=float)
    for simulated_origin in sorted(train_origins.unique()):
        mask = (train_origins == simulated_origin).to_numpy()
        anchors[mask] = robust_hour_shape_predict(
            history.loc[history.date < simulated_origin], train.loc[mask],
            "calendar_robust_28", group="route",
        )
    train_daily = _daily(train, anchors, observed=True)
    daily_origins = _origin_dates(train_daily)
    train_inputs: list[pd.DataFrame] = []
    for simulated_origin in sorted(daily_origins.unique()):
        mask = (daily_origins == simulated_origin).to_numpy()
        train_inputs.append(source_inputs(train_daily.loc[mask], simulated_origin,
                                          family, weather, posts,
                                          weather_embargo_days=weather_embargo_days))
    x_train = pd.concat(train_inputs, ignore_index=True)
    anchor_day = train_daily.anchor_day.to_numpy(dtype=float)
    target = np.clip((train_daily.boardings.to_numpy(dtype=float) - anchor_day)
                     / (anchor_day + 300), -1, 2)
    model = CatBoostRegressor(**HYBRID_CONFIG)
    model.fit(x_train, target)
    anchor = robust_hour_shape_predict(history, future, "calendar_robust_28", group="route")
    future_daily = _daily(future, anchor, observed=False)
    predicted = np.asarray(model.predict(source_inputs(future_daily, origin, family,
                                                       weather, posts,
                                                       weather_embargo_days=weather_embargo_days)),
                           dtype=float)
    relative = np.divide(predicted * (future_daily.anchor_day.to_numpy(dtype=float) + 300),
                         future_daily.anchor_day.to_numpy(dtype=float),
                         out=np.zeros(len(future_daily)),
                         where=future_daily.anchor_day.to_numpy(dtype=float) > 0)
    keyed = future_daily[["route", "date", "lead_day"]].copy()
    keyed["relative"] = relative
    mapped = future[["route", "date", "lead_day"]].merge(
        keyed, on=["route", "date", "lead_day"], how="left", validate="many_to_one",
    )
    if not np.isfinite(mapped.relative.to_numpy(dtype=float)).all():
        raise ValueError("branch correction does not cover forecast grid")
    return np.maximum(0, anchor * (1 + np.clip(mapped.relative.to_numpy(dtype=float),
                                               -CORRECTION_CAP, CORRECTION_CAP)))


def blend(incumbent: np.ndarray, branches: dict[str, np.ndarray],
          weights: dict[str, float]) -> np.ndarray:
    """Convex mixture with explicit grid, finite-value and simplex checks."""
    base = np.asarray(incumbent, dtype=float)
    if (not set(weights) <= set(branches) or any(not np.isfinite(weight) or weight < 0
                                                 for weight in weights.values())
            or sum(weights.values()) > 1 + 1e-12
            or not np.isfinite(base).all() or (base < 0).any()):
        raise ValueError("invalid convex mixture")
    result = (1 - sum(weights.values())) * base
    for name, weight in weights.items():
        branch = np.asarray(branches[name], dtype=float)
        if branch.shape != base.shape or not np.isfinite(branch).all() or (branch < 0).any():
            raise ValueError("misaligned branch prediction")
        result = result + weight * branch
    return result


def _segments(
    truth: np.ndarray, prediction: np.ndarray, keys: pd.DataFrame
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for route in sorted(keys.route.unique()):
        mask = (keys.route == route).to_numpy()
        output[f"route:{int(route)}"] = score(truth[mask], prediction[mask])
    for start in (0, 6, 12, 18):
        mask = keys.hour.between(start, start + 5).to_numpy()
        output[f"hours:{start:02d}-{start + 5:02d}"] = score(truth[mask], prediction[mask])
    return output


def _evaluate(truth: np.ndarray, prediction: np.ndarray,
              keys: pd.DataFrame) -> dict[str, Any]:
    return {"score": score(truth, prediction), "segments": _segments(truth, prediction, keys)}


def _choose(calibration: dict[str, Any]) -> dict[str, float]:
    eligible = [source for source in SOURCES if source not in INELIGIBLE]
    incumbent = calibration["incumbent"]
    truth = calibration["truth"]
    best_score = score(truth, incumbent)
    if best_score is None:
        raise ValueError("calibration target has no observed boardings")
    best: dict[str, float] = {}
    for combination in product(WEIGHT_GRID, repeat=len(eligible)):
        candidate = {name: weight for name, weight in zip(eligible, combination,
                                                         strict=True) if weight > 0}
        if sum(candidate.values()) > 0.30 + 1e-12:
            continue
        value = score(truth, blend(incumbent, calibration["branches"], candidate))
        if value is None:
            continue
        rank = (len(candidate), sum(candidate.values()), tuple(candidate.get(name, 0)
                                                           for name in eligible))
        best_rank = (len(best), sum(best.values()), tuple(best.get(name, 0)
                                                        for name in eligible))
        if value > best_score + 1e-12 or (abs(value - best_score) <= 1e-12
                                          and rank < best_rank):
            best_score, best = value, candidate
    return best


def _passes_windows(weights: dict[str, float], folds: dict[str, dict[str, Any]]) -> bool:
    if not weights:
        return True
    for origin in HOLDOUT_ORIGINS:
        fold = folds[origin.isoformat()]
        before = _evaluate(fold["truth"], fold["incumbent"], fold["keys"])
        after = _evaluate(fold["truth"], blend(fold["incumbent"], fold["branches"],
                                               weights), fold["keys"])
        if after["score"] - before["score"] < MIN_GAIN:
            return False
        if any(after["segments"][name] < baseline - MAX_SEGMENT_LOSS
               for name, baseline in before["segments"].items()
               if baseline is not None and after["segments"][name] is not None):
            return False
    return True


def _accepted(weights: dict[str, float], folds: dict[str, dict[str, Any]]) -> bool:
    return (_passes_windows(weights, folds)
            and all(_passes_windows({name: weight}, folds)
                    for name, weight in weights.items() if weight > 0))


def run(archive: Path, proof: Path, weather_path: Path, posts_path: Path,
        output_dir: Path) -> dict[str, Any]:
    archive_sha = _verified_proof(archive, proof)
    labels = complete_labels(load_labels(archive), date(2025, 1, 1),
                             date(2025, 10, 31), missing_as_zero=True)
    weather = load_weather(weather_path)
    posts = _posts(posts_path)
    folds: dict[str, dict[str, Any]] = {}
    for origin in ORIGINS:
        history = labels.loc[labels.date < origin]
        future = feature_frame(history, origin)
        keys = future[["route", "date", "hour"]].copy()
        truth = keys.merge(labels[["route", "date", "hour", "boardings"]],
                           on=["route", "date", "hour"], how="left",
                           validate="one_to_one").boardings.to_numpy(dtype=float)
        incumbent = hybrid_predict(history, future, "catboost_daily_recency_ensemble_50")
        branches = {}
        for family in SOURCES:
            branches[family] = _fit_branch(history, future, family, weather, posts,
                                           weather_embargo_days=1)
        folds[origin.isoformat()] = {"keys": keys, "truth": truth,
                                     "incumbent": incumbent, "branches": branches}
    proposed = _choose(folds[CALIBRATION_ORIGIN.isoformat()])
    selected = proposed if _accepted(proposed, folds) else {}
    metrics = {}
    for origin_key, fold in folds.items():
        truth, keys = fold["truth"], fold["keys"]
        metrics[origin_key] = {
            "incumbent": _evaluate(truth, fold["incumbent"], keys),
            "branches": {name: _evaluate(truth, value, keys)
                         for name, value in fold["branches"].items()},
            "branch_mix_005": {
                name: _evaluate(truth, blend(fold["incumbent"], fold["branches"],
                                             {name: 0.05}), keys)
                for name in SOURCES
            },
            "proposed_mix": _evaluate(truth, blend(fold["incumbent"], fold["branches"],
                                                   proposed), keys),
            "selected_mix": _evaluate(truth, blend(fold["incumbent"], fold["branches"],
                                                   selected), keys),
        }
    origin = date(2025, 11, 1)
    future = feature_frame(labels, origin)
    final_incumbent = hybrid_predict(labels, future, "catboost_daily_recency_ensemble_50")
    final_branches = {name: _fit_branch(labels, future, name, weather, posts,
                                       weather_embargo_days=1)
                      for name in SOURCES if selected.get(name, 0) > 0}
    prediction = blend(final_incumbent, final_branches, selected)
    output = future[["route", "date", "hour"]].copy()
    output["route"] = output.route.astype(int)
    output["hour"] = output.hour.astype(int)
    output["prediction"] = rounded(prediction)
    validate_submission(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv = output_dir / "approved-external-candidate.csv"
    write_submission(output, csv)
    report = {
        "schema": "approved-external-ensemble-evaluation.v1",
        "archive_sha256": archive_sha,
        "source_sha256": {"weather": _digest(weather_path), "traffic_events": _digest(posts_path)},
        "source_manifest_sha256": {
            "weather": _digest(weather_path.parent / "manifest.json"),
            "traffic_events": _digest(posts_path.parent / "manifest.json"),
        },
        "source_vintage": {
            "calendar": "official 2025 calendar issued 2024-10-04",
            "weather": "retrospective Open-Meteo best_match archive; 2025 vintage unproved",
            "traffic": "2026 retrieval of historical posts; 2025 text version unproved",
            "events": "prior road incident and closure notices only; operational-news proxy",
        },
        "origins": [str(origin) for origin in ORIGINS],
        "calibration_origin": str(CALIBRATION_ORIGIN),
        "heldout_origins": [str(origin) for origin in HOLDOUT_ORIGINS],
        "candidate_weights": list(WEIGHT_GRID),
        "availability_exclusions": INELIGIBLE,
        "proposed_weights": {"incumbent": 1 - sum(proposed.values()),
                             **{name: proposed.get(name, 0.0) for name in SOURCES}},
        "selected_weights": {"incumbent": 1 - sum(selected.values()),
                             **{name: selected.get(name, 0.0) for name in SOURCES}},
        "metrics": metrics,
        "csv": str(csv), "csv_sha256": _digest(csv), "csv_rows": len(output),
        "config": HYBRID_CONFIG,
        "source_fields": SOURCE_FIELDS,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("catboost", "numpy", "pandas", "tramflow-ml")},
    }
    (output_dir / "approved-external-evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report


def run_switches(
    archive: Path, proof: Path, weather_path: Path, posts_path: Path,
    output_dir: Path, artifact_dir: Path, generated_at: str,
) -> dict[str, Any]:
    """Build optional source variants without replacing the approved default."""
    timestamp = datetime.fromisoformat(generated_at)
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(hours=3):
        raise ValueError("generated_at must be a Moscow timestamp")
    archive_sha = _verified_proof(archive, proof)
    labels = complete_labels(load_labels(archive), date(2025, 1, 1),
                             date(2025, 10, 31), missing_as_zero=True)
    weather = load_issued_weather(weather_path)
    posts = _posts(posts_path)
    weights = {name: SWITCH_WEIGHT for name in SOURCES}
    metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        history = labels.loc[labels.date < origin]
        future = feature_frame(history, origin)
        keys = future[["route", "date", "hour"]].copy()
        truth = keys.merge(labels[["route", "date", "hour", "boardings"]],
                           on=["route", "date", "hour"], how="left",
                           validate="one_to_one").boardings.to_numpy(dtype=float)
        incumbent = rounded(hybrid_predict(history, future,
                                            "catboost_daily_recency_ensemble_50")).astype(float)
        branches = {name: _fit_branch(history, future, name, weather, posts)
                    for name in SOURCES}
        baseline = _evaluate(truth, incumbent, keys)
        all_enabled = _evaluate(truth, blend(incumbent, branches, weights), keys)
        metrics[origin.isoformat()] = {
            "incumbent": baseline,
            "standalone": {name: _evaluate(truth, values, keys)
                           for name, values in branches.items()},
            "single_source": {
                name: _evaluate(truth, blend(incumbent, branches, {name: SWITCH_WEIGHT}),
                                keys) for name in SOURCES
            },
            "all_enabled": all_enabled,
        }
        if origin in HOLDOUT_ORIGINS and baseline["score"] - all_enabled["score"] > MAX_SWITCH_LOSS:
            raise ValueError("predeclared source-switch loss bound exceeded")
    origin = date(2025, 11, 1)
    future = feature_frame(labels, origin)
    incumbent = rounded(hybrid_predict(labels, future,
                                        "catboost_daily_recency_ensemble_50")).astype(float)
    branches = {name: _fit_branch(labels, future, name, weather, posts)
                for name in SOURCES}
    keys = future[["route", "date", "hour"]].copy()
    keys["route"] = keys.route.astype(int)
    keys["hour"] = keys.hour.astype(int)
    all_enabled = keys.copy()
    all_enabled["prediction"] = rounded(blend(incumbent, branches, weights))
    validate_submission(all_enabled)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "approved-sources-all.csv"
    write_submission(all_enabled, csv_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "approved-source-variants.json.gz"
    artifact = {
        "schema": "approved-source-variants.v1", "cutoff": "2025-10-31",
        "generated_at": generated_at,
        "base_csv_sha256": "edc07cca26ff21dd5228f744dea79940eb81a65c3ce5368aa743e2fc2fd0f11f",
        "keys": keys.astype({"date": str}).values.tolist(),
        "branches": {name: np.round(branches[name], 6).tolist() for name in SOURCES},
        "weights": weights,
        "availability": {
            "weather": "issued forecast for past dates, two-day embargo; city-centre proxy",
            "traffic": "unedited public posts before origin; 2025 content vintage unarchived",
            "events": "unedited past operational incident/closure notices, not city event listings",
            "calendar": "official 2025 calendar issued 2024-10-04",
        },
    }
    raw = json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()
    with artifact_path.open("wb") as stream, gzip.GzipFile(fileobj=stream,
                                                            mode="wb", mtime=0,
                                                            filename="") as zipped:
        zipped.write(raw)
    manifest = {
        "schema": "approved-source-variants-manifest.v1",
        "sha256": _digest(artifact_path), "rows": len(keys),
        "archive_sha256": archive_sha,
        "source_sha256": {"weather": _digest(weather_path), "posts": _digest(posts_path)},
        "source_manifest_sha256": {
            "weather": _digest(weather_path.parent / "manifest.json"),
            "posts": _digest(posts_path.parent / "manifest.json"),
        },
        "csv_sha256": _digest(csv_path),
        "config": HYBRID_CONFIG,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("catboost", "numpy", "pandas", "tramflow-ml")},
        "metrics": metrics,
        "origin": "2025-11-01", "weights": {"incumbent": 0.8, **weights},
    }
    (artifact_dir / "approved-source-variants-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {"csv": str(csv_path), "csv_sha256": _digest(csv_path),
            "artifact_sha256": _digest(artifact_path), "metrics": metrics,
            "weights": manifest["weights"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    if args.artifact_dir is not None:
        if args.generated_at is None:
            parser.error("--generated-at is required with --artifact-dir")
        report = run_switches(args.archive, args.reconciliation, args.weather,
                              args.posts, args.output_dir, args.artifact_dir,
                              args.generated_at)
        print(json.dumps({key: report[key] for key in ("csv_sha256", "artifact_sha256",
                                                       "weights")}, sort_keys=True))
        return
    report = run(args.archive, args.reconciliation, args.weather,
                 args.posts, args.output_dir)
    print(json.dumps({key: report[key] for key in ("selected_weights", "csv_sha256",
                                                    "csv_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Experimental independent stop residuals; inferred targets remain explicitly unverified."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.stop_models import (
    FloatArray,
    StopData,
    calendar_stop_profile,
    features,
    prepare,
    profile,
)

VERSION = "stop-calendar-bounded.v1"
CONFIG = {"iterations": 200, "depth": 5, "learning_rate": .05,
          "loss_function": "RMSE", "random_seed": 20260927, "thread_count": 4,
          "verbose": False, "allow_writing_files": False, "allow_const_label": True}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_verified(source: Path, route_export: Path) -> StopData:
    """Accept only complete hash-bound route labels and mass-conserving mapped targets."""
    manifest = json.loads((route_export / "manifest.json").read_text())
    filename = "route-hour-labels.csv.gz"
    if (manifest.get("complete") is not True
            or manifest.get("timezone") != "Europe/Moscow"
            or manifest.get("target") != "successful_validation_count_route_date_hour"
            or not manifest.get("reconciliation_sha256")):
        raise ValueError("verified route export required")
    if _digest(route_export / filename) != manifest["files"][filename]["sha256"]:
        raise ValueError("route label checksum mismatch")
    labels = pd.read_csv(route_export / filename)
    labels["date"] = pd.to_datetime(labels.date).dt.date
    if (len(labels) != manifest["files"][filename]["rows"]
            or labels.boardings.sum() != manifest["target_mass"]):
        raise ValueError("route label mass or count mismatch")
    mapped = json.loads((source / "manifest.json").read_text())
    frames = []
    for split in ("train", "development", "diagnostic"):
        info = mapped["files"][f"soft_{split}"]
        if _digest(source / info["name"]) != info["sha256"]:
            raise ValueError("stop target checksum mismatch")
        frame = pd.read_csv(source / info["name"], dtype={"stop_id": str})
        if len(frame) != info["rows"]:
            raise ValueError("stop target count mismatch")
        frames.append(frame)
    if _digest(source / "stop_catalog.csv") != mapped["catalog_sha256"]:
        raise ValueError("stop catalog checksum mismatch")
    result = prepare(pd.concat(frames, ignore_index=True),
                     pd.read_csv(source / "stop_catalog.csv", dtype={"stop_id": str}), labels,
                     complete_decode=mapped.get("complete") is True,
                     raw_reconciled=True, source_version=mapped["schema_version"])
    result.audit.update({"route_manifest_sha256": _digest(route_export / "manifest.json"),
                         "source_manifest_sha256": _digest(source / "manifest.json"),
                         "archive_sha256": manifest["archive_sha256"]})
    return result


def refinement_features(data: StopData, origin: int, days: int) -> Any:
    """All count, share and coverage predictors end strictly before origin."""
    x = features(data.train, data.identities, origin, days)
    anchor = calendar_stop_profile(data.train, origin, days, blend=True)
    x["calendar_anchor"] = anchor.ravel()
    routes = data.identities.route.to_numpy()
    known = ~data.identities.stop_id.astype(str).str.startswith("unallocated").to_numpy()
    for window in (7, 14, 28):
        recent = data.train[max(0, origin - window):origin]
        mass = np.nansum(recent, axis=(0, 2))
        total = np.zeros(len(routes))
        coverage = np.zeros(len(routes))
        for route in np.unique(routes):
            mask = routes == route
            denominator = mass[mask].sum()
            total[mask] = denominator
            coverage[mask] = mass[mask & known].sum() / denominator if denominator else 0
        shares = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
        x[f"recent_profile_{window}"] = profile(data.train, origin, days, window).ravel()
        x[f"recent_share_{window}"] = np.tile(np.repeat(shares, 24), days)
        x[f"mapping_coverage_{window}"] = np.tile(np.repeat(coverage, 24), days)
    return x


def bounded_prediction(anchor: FloatArray, correction: FloatArray, cap: float = .05) -> FloatArray:
    if (anchor.shape != correction.shape or not np.isfinite(anchor).all()
            or not np.isfinite(correction).all() or (anchor < 0).any()
            or not np.isfinite(cap) or not 0 <= cap <= 1):
        raise ValueError("invalid bounded correction")
    return np.asarray(anchor * (1 + np.clip(correction, -cap, cap)), dtype=float)


def predict_refined(data: StopData, origin: int, days: int) -> FloatArray:
    """Fit independent stop corrections; never constrain them to observed route totals."""
    if origin < 90 or origin > len(data.train) or days < 1 or origin + days > 365:
        raise ValueError("insufficient history or unsupported horizon")
    from catboost import CatBoostRegressor  # type: ignore[import-not-found,import-untyped]

    rng = np.random.default_rng(20260927)
    frames, targets = [], []
    for past_origin in range(59, origin, 28):
        width = min(61, origin - past_origin)
        x = refinement_features(data, past_origin, width)
        y = data.train[past_origin:past_origin + width].ravel()
        eligible = np.flatnonzero(np.isfinite(y))
        chosen = rng.choice(eligible, min(20000, len(eligible)), replace=False)
        selected = x.iloc[chosen]
        base = selected.calendar_anchor.to_numpy()
        frames.append(selected)
        targets.append(np.clip((y[chosen] - base) / (base + 1), -1, 2))
    model = CatBoostRegressor(**CONFIG)
    model.fit(pd.concat(frames, ignore_index=True), np.concatenate(targets),
              cat_features=["route", "stop", "direction"])
    future = refinement_features(data, origin, days)
    prediction = bounded_prediction(future.calendar_anchor.to_numpy(),
                                    np.asarray(model.predict(future), dtype=float))
    return prediction.reshape(days, len(data.identities), 24)

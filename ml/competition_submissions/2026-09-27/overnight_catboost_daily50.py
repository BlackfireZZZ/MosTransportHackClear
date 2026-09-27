"""Daily-level CatBoost correction of a cutoff-safe route-hour anchor."""

import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from tramflow_ml.competition import (
    SUBMISSION_END,
    SUBMISSION_START,
    complete_labels,
    feature_frame,
    load_labels,
    make_training_frame,
    rounded,
    score,
    write_submission,
)
from tramflow_ml.competition_cli import _verified_proof
from tramflow_ml.competition_profile import robust_hour_shape_predict

ROOT = Path(os.environ["TRAMFLOW_DATA_DIR"]).expanduser().resolve()
ARCHIVE = ROOT / "dataset.zip"
CAPS = (0.05, 0.10, 0.20)
WEIGHT = 0.5
FIELDS = [
    "route",
    "day_of_week",
    "month",
    "day_of_year",
    "is_summer",
    "lead_day",
    "route_day_28",
    "route_day_56",
    "history_days",
    "calendar_day_type",
    "anchor_day",
]
CONFIG = {
    "loss_function": "MAE",
    "iterations": 400,
    "depth": 4,
    "learning_rate": 0.04,
    "l2_leaf_reg": 10,
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
    "cat_features": ["route"],
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def daily(rows: pd.DataFrame, anchor: np.ndarray, observed: bool) -> pd.DataFrame:
    frame = rows[[*FIELDS[:-1], "date", *(["boardings"] if observed else [])]].copy()
    frame["anchor_day"] = anchor
    reductions = {field: "first" for field in FIELDS if field not in ("route", "lead_day")}
    reductions["anchor_day"] = "sum"
    if observed:
        reductions["boardings"] = "sum"
    return frame.groupby(["route", "date", "lead_day"], sort=False, as_index=False).agg(reductions)


def inputs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FIELDS].copy()
    result["route"] = result.route.astype(int).astype(str)
    return result


def forecast(labels: pd.DataFrame, origin: date):
    past = labels.loc[labels.date < origin]
    train = make_training_frame(past, origin - timedelta(days=1))
    train_origins = (
        pd.to_datetime(train.date) - pd.to_timedelta(train.lead_day.astype(int) - 1, unit="D")
    ).dt.date
    anchor_train = np.empty(len(train), dtype=float)
    for historical_origin in sorted(train_origins.unique()):
        mask = (train_origins == historical_origin).to_numpy()
        anchor_train[mask] = robust_hour_shape_predict(
            past.loc[past.date < historical_origin],
            train.loc[mask],
            "calendar_robust_28",
            group="route",
        )
    train_daily = daily(train, anchor_train, True)
    target = np.clip(
        (train_daily.boardings.to_numpy() - train_daily.anchor_day.to_numpy())
        / (train_daily.anchor_day.to_numpy() + 300),
        -1,
        2,
    )
    model = CatBoostRegressor(**CONFIG)
    model.fit(inputs(train_daily), target)
    future = feature_frame(labels, origin)
    base = robust_hour_shape_predict(past, future, "calendar_robust_28", group="route")
    future_daily = daily(future, base, False)
    daily_residual = model.predict(inputs(future_daily))
    daily_relative = np.divide(
        daily_residual * (future_daily.anchor_day.to_numpy() + 300),
        future_daily.anchor_day.to_numpy(),
        out=np.zeros(len(future_daily)),
        where=future_daily.anchor_day.to_numpy() > 0,
    )
    future_daily["relative_correction"] = daily_relative
    mapped = future[["route", "date", "lead_day"]].merge(
        future_daily[["route", "date", "lead_day", "relative_correction"]],
        on=["route", "date", "lead_day"],
        how="left",
        validate="many_to_one",
    )
    if mapped.relative_correction.isna().any():
        raise ValueError("daily correction does not cover the forecast grid")
    return (
        future,
        base,
        mapped.relative_correction.to_numpy(),
        {
            "train_rows": len(train_daily),
            "train_origins": len(train_origins.unique()),
            "train_cutoff": (origin - timedelta(days=1)).isoformat(),
        },
    )


def candidate(base: np.ndarray, correction: np.ndarray, cap: float) -> np.ndarray:
    ml = np.maximum(0, base * (1 + np.clip(correction, -cap, cap)))
    return (1 - WEIGHT) * base + WEIGHT * ml


def main():
    sha = _verified_proof(ARCHIVE, ROOT / "reconciliation.json")
    labels = complete_labels(
        load_labels(ARCHIVE), date(2025, 1, 1), date(2025, 10, 31), missing_as_zero=True
    )
    report = {
        "archive_sha256": sha,
        "model": "catboost_daily_bounded_50",
        "ml_weight": WEIGHT,
        "config": CONFIG,
        "caps": CAPS,
        "folds": {},
    }
    for origin in (date(2025, 5, 1), date(2025, 7, 1), date(2025, 9, 1)):
        future, base, correction, meta = forecast(labels, origin)
        truth = (
            future[["route", "date", "hour"]]
            .merge(
                labels.loc[
                    (labels.date >= origin) & (labels.date < origin + timedelta(days=61)),
                    ["route", "date", "hour", "boardings"],
                ],
                on=["route", "date", "hour"],
                validate="one_to_one",
            )
            .boardings.to_numpy()
        )
        row = {"base": score(truth, base), "truth_sum": int(truth.sum()), **meta}
        for cap in CAPS:
            pred = candidate(base, correction, cap)
            row[f"cap_{cap}"] = score(truth, pred)
            row[f"changed_rows_{cap}"] = int(np.count_nonzero(rounded(pred) != rounded(base)))
        report["folds"][origin.isoformat()] = row
        print(origin, row, flush=True)
    future, base, correction, meta = forecast(labels, SUBMISSION_START)
    cap = 0.05
    output = future[["route", "date", "hour"]].copy()
    output["route"] = output.route.astype(int)
    output["hour"] = output.hour.astype(int)
    output["prediction"] = rounded(candidate(base, correction, cap))
    if max(output.date) != SUBMISSION_END:
        raise ValueError("wrong submission horizon")
    path = ROOT / "submission-catboost50-daily-bounded05-routeshape.csv"
    write_submission(output, path)
    report["submission"] = {
        "path": str(path),
        "sha256": digest(path),
        "rows": len(output),
        "cap": cap,
        "training": meta,
        "changed_rows": int(np.count_nonzero(output.prediction.to_numpy() != rounded(base))),
    }
    (ROOT / "overnight_catboost_daily50.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report["submission"], flush=True)


if __name__ == "__main__":
    main()

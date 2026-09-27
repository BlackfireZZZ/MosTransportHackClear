"""Test recency weighting for the 50% daily CatBoost route-hour candidate."""

import json
from datetime import date, timedelta

import numpy as np
import overnight_catboost_daily50 as base_model
import pandas as pd
from catboost import CatBoostRegressor

from tramflow_ml.competition import (
    complete_labels,
    feature_frame,
    load_labels,
    make_training_frame,
    score,
)
from tramflow_ml.competition_cli import _verified_proof
from tramflow_ml.competition_profile import robust_hour_shape_predict


def forecast(labels: pd.DataFrame, origin: date, half_life: int):
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
    train_daily = base_model.daily(train, anchor_train, True)
    target = np.clip(
        (train_daily.boardings.to_numpy() - train_daily.anchor_day.to_numpy())
        / (train_daily.anchor_day.to_numpy() + 300),
        -1,
        2,
    )
    daily_origins = (
        pd.to_datetime(train_daily.date)
        - pd.to_timedelta(train_daily.lead_day.astype(int) - 1, unit="D")
    ).dt.date
    age = np.asarray([(origin - x).days for x in daily_origins], dtype=float)
    weights = np.exp(-np.log(2) * age / half_life)
    model = CatBoostRegressor(**base_model.CONFIG)
    model.fit(base_model.inputs(train_daily), target, sample_weight=weights)
    future = feature_frame(labels, origin)
    anchor = robust_hour_shape_predict(past, future, "calendar_robust_28", group="route")
    future_daily = base_model.daily(future, anchor, False)
    daily_residual = model.predict(base_model.inputs(future_daily))
    relative = np.divide(
        daily_residual * (future_daily.anchor_day.to_numpy() + 300),
        future_daily.anchor_day.to_numpy(),
        out=np.zeros(len(future_daily)),
        where=future_daily.anchor_day.to_numpy() > 0,
    )
    future_daily["relative"] = relative
    mapped = future[["route", "date", "lead_day"]].merge(
        future_daily[["route", "date", "lead_day", "relative"]],
        on=["route", "date", "lead_day"],
        how="left",
        validate="many_to_one",
    )
    if mapped.relative.isna().any():
        raise ValueError("daily correction does not cover the forecast grid")
    return future, anchor, mapped.relative.to_numpy(), len(train_origins.unique())


def main():
    _verified_proof(base_model.ARCHIVE, base_model.ROOT / "reconciliation.json")
    labels = complete_labels(
        load_labels(base_model.ARCHIVE), date(2025, 1, 1), date(2025, 10, 31), missing_as_zero=True
    )
    report = {"half_lives": [60, 120], "cap": 0.05, "ml_weight": 0.5, "folds": {}}
    for origin in (
        date(2025, 5, 1),
        date(2025, 7, 1),
        date(2025, 8, 1),
        date(2025, 8, 15),
        date(2025, 9, 1),
        date(2025, 9, 15),
        date(2025, 10, 1),
    ):
        row = {}
        for half_life in report["half_lives"]:
            future, anchor, correction, origins = forecast(labels, origin, half_life)
            end = min(origin + timedelta(days=60), date(2025, 10, 31))
            mask = (future.date <= end).to_numpy()
            truth = (
                future.loc[mask, ["route", "date", "hour"]]
                .merge(
                    labels.loc[
                        (labels.date >= origin) & (labels.date <= end),
                        ["route", "date", "hour", "boardings"],
                    ],
                    on=["route", "date", "hour"],
                    validate="one_to_one",
                )
                .boardings.to_numpy()
            )
            row["base"] = score(truth, anchor[mask])
            row["truth_sum"] = int(truth.sum())
            row["days"] = (end - origin).days + 1
            row["train_origins"] = origins
            row[str(half_life)] = score(truth, base_model.candidate(anchor, correction, 0.05)[mask])
        report["folds"][origin.isoformat()] = row
        print(origin, row, flush=True)
    (base_model.ROOT / "overnight_catboost_recency.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

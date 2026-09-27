"""Bounded offline residual ablation; frozen plan controls the complete experiment."""

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tramflow_ml.competition import feature_frame, make_training_frame
from tramflow_ml.competition_hybrid import hybrid_predict
from tramflow_ml.competition_profile import robust_hour_shape_predict
from tramflow_ml.external_factors import SOURCES, load_labels, source_features


def bounded(anchor, residual, bound):
    """Relative bounds preserve zero anchors and cannot create negative counts."""
    anchor, residual = np.asarray(anchor), np.asarray(residual)
    if anchor.shape != residual.shape or not 0 <= bound <= 1:
        raise ValueError("incompatible correction contract")
    if not np.isfinite(anchor).all() or not np.isfinite(residual).all() or (anchor < 0).any():
        raise ValueError("finite nonnegative anchors and finite residuals required")
    return anchor + np.clip(residual, -bound * anchor, bound * anchor)


def metrics(actual, prediction):
    error = float(np.abs(actual - np.rint(prediction)).sum())
    mass = float(actual.sum())
    return {"actual": mass, "absolute_error": error,
            "score": max(0., 1 - error / mass) if mass else None}


def run(labels, weather, posts, plan, output):
    from catboost import CatBoostRegressor

    results = []
    for origin_text in plan["origins"]:
        origin = date.fromisoformat(origin_text)
        history = labels.loc[labels.date < origin]
        train = make_training_frame(history, origin - timedelta(days=1))
        inner_dates = (pd.to_datetime(train.date)
                       - pd.to_timedelta(train.lead_day - 1, unit="D")).dt.date
        future = feature_frame(history, origin)
        target = future[["route", "date", "hour"]].merge(
            labels, validate="one_to_one", how="left").boardings.to_numpy(dtype=float)
        anchor = robust_hour_shape_predict(history, future, "calendar_robust_28", group="route")
        train_anchor = np.empty(len(train))
        for inner in sorted(inner_dates.unique()):
            mask = inner_dates == inner
            train_anchor[mask] = robust_hour_shape_predict(
                history.loc[history.date < inner], train.loc[mask],
                "calendar_robust_28", group="route")
        assert train.date.max() < origin
        predictions = {"robust28": anchor, "published_hybrid": hybrid_predict(
            history, future, "catboost_daily_recency_ensemble_50")}
        for variant in plan["variants"]:
            sources = SOURCES if variant == "all" else (
                frozenset() if variant == "base" else frozenset({variant}))
            pieces = []
            for inner in sorted(inner_dates.unique()):
                part = train.loc[inner_dates == inner]
                frame = source_features(part, inner, enabled=sources,
                                        weather=weather, posts=posts)
                frame.index = part.index
                pieces.append(frame)
            xtrain = pd.concat(pieces).sort_index()
            xtrain["anchor"] = train_anchor
            xfuture = source_features(future, origin, enabled=sources,
                                      weather=weather, posts=posts)
            xfuture["anchor"] = anchor
            model = CatBoostRegressor(**plan["config"], cat_features=["route"])
            model.fit(xtrain, train.boardings.to_numpy(dtype=float) - train_anchor)
            predictions[variant] = bounded(anchor, model.predict(xfuture),
                                           plan["correction_bound"])
        for name, prediction in predictions.items():
            row = {"origin": origin_text, "variant": name, **metrics(target, prediction)}
            row["route_slices"] = {
                str(route): metrics(target[future.route == route], prediction[future.route == route])
                for route in sorted(future.route.unique())}
            row["month_slices"] = {
                str(month): metrics(target[pd.to_datetime(future.date).dt.month == month],
                                    prediction[pd.to_datetime(future.date).dt.month == month])
                for month in sorted(pd.to_datetime(future.date).dt.month.unique())}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("slices")}), flush=True)
        (output / "scores.json").write_text(json.dumps(results, indent=2) + "\n")
    scores = {(r["origin"], r["variant"]): r["score"] for r in results}
    development = plan["origins"][:2]
    selected = min(plan["variants"], key=lambda v: (-sum(scores[o, v] for o in development), v))
    accepted = all(scores[o, selected] - scores[o, baseline] >= (0.0001 if o in development else 0)
                   for o in plan["origins"] for baseline in ("robust28", "published_hybrid"))
    (output / "decision.json").write_text(json.dumps({"selected": selected, "accepted": accepted,
        "status": "candidate_requires_independent_review" if accepted else "rejected",
        "source_effect": {v: {o: scores[o, v] - scores[o, "base"] for o in plan["origins"]}
                          for v in plan["variants"] if v != "base"}}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "weather", "posts", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    weather = pd.read_csv(args.weather, sep=";")
    weather["date"] = pd.to_datetime(weather.date).dt.date
    inputs = {"plan": args.plan, "weather": args.weather, "posts": args.posts,
              "labels": args.labels / "route-hour-labels.csv.gz", "script": Path(__file__)}
    (args.output / "manifest.json").write_text(json.dumps({
        "version": plan["version"], "timezone": plan["timezone"],
        "input_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in inputs.items()},
        "catboost_version": __import__("catboost").__version__}, indent=2) + "\n")
    run(load_labels(args.labels), weather, pd.read_json(args.posts, lines=True), plan, args.output)


if __name__ == "__main__":
    main()

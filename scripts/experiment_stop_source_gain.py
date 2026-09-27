"""Preregistered bounded residual ablation against the anchored stop baseline."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from stop_source_gain_features import event_tensor, feature_frame
from tramflow_ml.route_models.features import DOW
from tramflow_ml.stop_models import aggregate, calendar_stop_profile, evaluate
from tramflow_ml.stop_refinement import load_verified

CONFIG = {"iterations": 200, "depth": 5, "learning_rate": .05,
          "loss_function": "MAE", "l2_leaf_reg": 10, "random_seed": 20260927,
          "thread_count": 4, "verbose": False, "allow_writing_files": False,
          "allow_const_label": True}
CAPS = (0., .025, .10, .25)
ORIGINS = (120, 181, 243)
MASKS = (0, 1, 15)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchor(history, origin, days, calendar):
    """Calendar-off retains ordinary weekdays only, including civil weekends."""
    if calendar:
        return calendar_stop_profile(history, origin, days, blend=True)
    past = history[:origin]
    output = np.zeros((days, history.shape[1], 24))
    ordinary = np.where(DOW < 5, 0, DOW - 4)
    for lead in range(days):
        target = origin + lead
        means = []
        for grouping in (DOW, ordinary):
            selected = past[grouping[:origin] == grouping[target]]
            counts = np.isfinite(selected).sum(axis=0)
            means.append(np.divide(np.nansum(selected, axis=0), counts,
                                   out=np.zeros(counts.shape), where=counts > 0))
        output[lead] = .5 * means[0] + .5 * means[1]
    return output


def features(data, origin, days, mask, weather, posts, events):
    frame = feature_frame(data, origin, days, mask, weather, posts, events)
    baseline = anchor(data.train, origin, days, bool(mask & 1))
    frame["anchor"] = baseline.ravel()
    return frame, baseline


def fit(data, origin, mask, weather, posts, events):
    rng = np.random.default_rng(20260927)
    frames, targets = [], []
    for inner in range(59, origin, 30):
        width = min(61, origin-inner)
        frame, baseline = features(data, inner, width, mask, weather, posts, events)
        labels = data.train[inner:inner+width].ravel()
        eligible = np.flatnonzero(np.isfinite(labels))
        chosen = rng.choice(eligible, min(12000, len(eligible)), replace=False)
        frames.append(frame.iloc[chosen])
        targets.append(labels[chosen] - baseline.ravel()[chosen])
    model = CatBoostRegressor(**CONFIG)
    model.fit(pd.concat(frames, ignore_index=True), np.concatenate(targets),
              cat_features=["route", "stop", "direction"])
    frame, baseline = features(data, origin, 61, mask, weather, posts, events)
    residual = np.asarray(model.predict(frame)).reshape(baseline.shape)
    return baseline, residual, model


def bounded(baseline, residual, cap):
    return np.maximum(0, baseline + np.clip(residual, -cap*baseline, cap*baseline))


def measure(data, prediction, origin):
    actual = data.values[origin:origin+61]
    known = ~data.identities.stop_id.astype(str).str.startswith("unallocated").to_numpy()
    route_y = aggregate(actual, data.identities, origin)
    route_p = aggregate(prediction, data.identities, origin)
    route_y["absolute_error"] = np.abs(route_y.prediction - np.rint(route_p.prediction))
    daily = route_y.groupby("date")[["prediction", "absolute_error"]].sum()
    return {**evaluate(data, prediction, origin),
            "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
            "daily": {"dates": [str(d) for d in daily.index],
                      "known_absolute_error": np.abs(prediction[:, known]-actual[:, known]).sum(axis=(1, 2)).tolist(),
                      "known_actual": actual[:, known].sum(axis=(1, 2)).tolist(),
                      "route_absolute_error": daily.absolute_error.tolist(),
                      "route_actual": daily.prediction.tolist()}}


def assess(rows, cap, origins):
    by_key = {(r["origin"], r["mask"], r["cap"]): r for r in rows}
    comparisons = []
    for origin in origins:
        candidate = by_key[origin, 15, cap]
        for label, mask, control_cap in (("matched_no_sources", 0, cap), ("strong_default", 1, 0.)):
            control = by_key[origin, mask, control_cap]
            gain = 1-candidate["pseudo_wape_known"]/control["pseudo_wape_known"]
            route_delta = candidate["route_score"]-control["route_score"]
            comparisons.append({"origin": origin, "control": label, "known_wape_relative_gain": gain,
                                "route_score_delta": route_delta,
                                "pass": gain >= (.01 if origin in (120, 181) else 0.) and route_delta >= -.002})
    return {"cap": cap, "pass": all(r["pass"] for r in comparisons), "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "route-export", "weather", "posts", "events", "report", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"version": "stop-source-gain.v1", "objective": "matched external-source gain on anchored-v3 inferred stop labels",
            "config": CONFIG, "masks": MASKS, "caps": CAPS, "origins": ORIGINS, "horizon_days": 61,
            "sample_per_inner_origin": 12000, "inner_origins": "range(59, outer, 30); width=min(61, outer-inner)",
            "anchor_off": "0.5 all-history ordinary weekday mean + 0.5 ordinary Mon-Fri/Sat/Sun mean; no holiday/month",
            "anchor_on": "calendar_stop_profile(blend=True), same as strong live baseline",
            "acceptance": "All-on known-stop pseudoWAPE >=1% relatively lower in BOTH May/July and nonworse September versus BOTH matched mask0 and strong calendar baseline; route score delta >=-0.002 in every comparison",
            "selection": "Only May/July: passing cap with lowest mean known-stop pseudoWAPE, smaller cap breaks ties; no replacement on diagnostic failure; no additional tuning",
            "diagnostic_blind": False, "labels": "inferred stop counts, not observed stop truth; route aggregates reconciled to observed labels",
            "timezone": "Europe/Moscow", "leakage_policy": "all dynamic features and anchors strictly before their inner or outer origin; events/geometry retrospective; no future weather forecast",
            "research": ["https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit", "https://otexts.com/fpp3/tscv.html"],
            "versions": {p: importlib.metadata.version(p) for p in ("catboost", "numpy", "pandas")},
            "source_paths": {k: str(v) for k, v in vars(args).items()},
            "source_sha256": {k: digest(v) for k, v in {
                "builder": Path(__file__), "stop_source_gain_features": Path("scripts/stop_source_gain_features.py"),
                "stop_models": Path("ml/src/tramflow_ml/stop_models.py"), "stop_refinement": Path("ml/src/tramflow_ml/stop_refinement.py"),
                "source_manifest": args.source/"manifest.json", "catalog": args.source/"stop_catalog.csv",
                "route_manifest": args.route_export/"manifest.json", "weather": args.weather,
                "posts": args.posts, "events": args.events}.items()}}
    plan_path = args.report/"plan.json"
    encoded = json.dumps(plan, indent=2)+"\n"
    if plan_path.exists() and plan_path.read_text() != encoded:
        raise ValueError("frozen plan differs; cannot overwrite")
    plan_path.write_text(encoded)
    print("FROZEN PLAN", digest(plan_path), flush=True)
    data = load_verified(args.source, args.route_export)
    weather = pd.read_csv(args.weather, sep=";")
    weather["date"] = pd.to_datetime(weather.date).dt.date
    posts = pd.read_json(args.posts, lines=True)
    events, event_audit = event_tensor(args.events, data.identities)
    result = {"plan_sha256": digest(plan_path), "dataset_audit": data.audit, "events": event_audit, "rows": []}
    result_path = args.report/"results.json"
    def persist():
        result_path.write_text(json.dumps(result, indent=2)+"\n")
    for origin in ORIGINS:
        for mask in MASKS:
            baseline, residual, model = fit(data, origin, mask, weather, posts, events)
            for cap in CAPS:
                prediction = bounded(baseline, residual, cap)
                row = {"origin": origin, "mask": mask, "cap": cap, **measure(data, prediction, origin)}
                result["rows"].append(row)
                print(json.dumps({k: row[k] for k in ("origin", "mask", "cap", "pseudo_wape_known", "route_score")}), flush=True)
                persist()
        if origin == 181:
            result["development_assessments"] = [assess(result["rows"], cap, (120, 181)) for cap in CAPS]
            passing = [x["cap"] for x in result["development_assessments"] if x["pass"]]
            result["selected_cap_before_diagnostic"] = min(passing, key=lambda c: (np.mean([r["pseudo_wape_known"] for r in result["rows"] if r["mask"] == 15 and r["cap"] == c]), c)) if passing else None
            persist()
    result["final_assessments"] = [assess(result["rows"], cap, ORIGINS) for cap in CAPS]
    selected = result["selected_cap_before_diagnostic"]
    accepted = selected is not None and assess(result["rows"], selected, ORIGINS)["pass"]
    result["status"] = "accepted_retro_experiment" if accepted else "rejected_no_promotion"
    persist()
    if accepted:
        baseline, residual, model = fit(data, 304, 15, weather, posts, events)
        prediction = bounded(baseline, residual, selected)
        model.save_model(str(args.output/"model.cbm"))
        count = len(data.identities)
        stop = pd.DataFrame({"route": np.tile(np.repeat(data.identities.route, 24), 61),
                             "direction": np.tile(np.repeat(data.identities.direction, 24), 61),
                             "stop_id": np.tile(np.repeat(data.identities.stop_id, 24), 61),
                             "date": np.repeat(pd.date_range("2025-11-01", periods=61).date, count*24),
                             "hour": np.tile(np.arange(24), 61*count), "prediction": prediction.ravel()})
        stop.to_csv(args.output/"stop_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        aggregate(prediction, data.identities, 304).to_csv(args.output/"route_sums.csv", index=False)
        result["final_artifacts"] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
        persist()
    print("FINAL STATUS", result["status"], "SELECTED", selected, flush=True)


if __name__ == "__main__":
    main()

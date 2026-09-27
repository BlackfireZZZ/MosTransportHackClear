"""Independent stop-hour regression on explicitly inferred, uncalibrated targets."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from tramflow_ml.competition import LABEL_ROUTES, hour_grid
from tramflow_ml.route_models.features import DAY_TYPE, DOW, HOLIDAY, MONTH, OFF, WORK

FloatArray = NDArray[np.float64]
IDENTITY = ["route", "stop_id", "direction"]
FEATURE_VERSION = "direct-stop-calendar-history.v1"


@dataclass
class StopData:
    """Decoded pseudo-count tensor; excluded blocks remain missing during training."""

    values: FloatArray
    train: FloatArray
    identities: Any
    audit: dict[str, Any]


def aggregate(values: FloatArray, identities: Any, start: int) -> Any:
    """Sum independent continuous predictions before any route-level rounding."""
    if values.ndim != 3 or values.shape[1:] != (len(identities), 24):
        raise ValueError("expected day x identity x 24 tensor")
    first = date(2025, 1, 1) + timedelta(days=start)
    result = hour_grid(first, first + timedelta(days=len(values) - 1))
    totals = np.zeros((len(LABEL_ROUTES), len(values), 24))
    for index, route in enumerate(LABEL_ROUTES):
        totals[index] = values[:, identities.route.to_numpy() == route].sum(axis=1)
    result["prediction"] = totals.ravel()
    return result


def prepare(
    rows: Any, coordinates: Any, labels: Any, *, complete_decode: bool,
    raw_reconciled: bool, source_version: str, days: int = 304,
) -> StopData:
    """Zero-fill inferred cells only after full route-hour mass reconciliation.

    No absent stop cell becomes an observed boarding label. A single ineligible
    source row masks its whole route-hour block, including absent stop cells.
    """
    if not complete_decode or not raw_reconciled or not source_version:
        raise ValueError("complete decoding, verified raw reconciliation and version required")
    rows = rows.copy()
    if rows.stop_id.isna().any() or rows.direction.isna().any():
        raise ValueError("unallocated rows require a separate explicit bucket")
    coordinates = coordinates.copy()
    for column, bound in (("lat", 90), ("lon", 180)):
        values = pd.to_numeric(coordinates[column], errors="raise")
        present = values.notna()
        if not np.isfinite(values[present]).all() or (values[present].abs() > bound).any():
            raise ValueError("invalid coordinate range")
        coordinates[column] = values
    if (coordinates.lat.isna() != coordinates.lon.isna()).any():
        raise ValueError("partial coordinate pair")
    if not rows.day.astype(str).str.fullmatch(r"2025-\d{2}-\d{2}").all():
        raise ValueError("day must be a 2025 civil date")
    hours = pd.to_numeric(rows.hour, errors="raise")
    if not np.isfinite(hours).all() or (hours % 1 != 0).any():
        raise ValueError("hour must be an integer")
    rows["hour"] = hours.astype(int)
    rows["stop_id"] = rows.stop_id.astype(str)
    coordinates["stop_id"] = coordinates.stop_id.astype(str)
    if not np.isfinite(rows.expected_count).all() or (rows.expected_count < 0).any():
        raise ValueError("invalid pseudo-count")
    if not rows.route.isin(LABEL_ROUTES).all():
        raise ValueError("route outside scored grid")
    if not rows.training_eligible.isin([True, False]).all():
        raise ValueError("eligibility must be explicit boolean")
    if rows.stop_id.isna().any() or rows.direction.isna().any():
        raise ValueError("unallocated rows require a separate explicit bucket")
    buckets = pd.DataFrame({"route": LABEL_ROUTES,
                            "stop_id": [f"unallocated:{r}" for r in LABEL_ROUTES],
                            "direction": -1})
    identities = pd.concat([coordinates[IDENTITY], buckets]).drop_duplicates()
    identities = identities.sort_values(IDENTITY).reset_index(drop=True)
    identities["identity"] = np.arange(len(identities))
    for column in ("lat", "lon"):
        if coordinates.groupby("stop_id")[column].nunique().max() > 1:
            raise ValueError("ambiguous stop coordinates")
    coords = coordinates[["stop_id", "lat", "lon"]].drop_duplicates("stop_id")
    identities = identities.merge(coords, on="stop_id", how="left", validate="many_to_one")
    rows = rows.merge(identities[IDENTITY + ["identity"]], on=IDENTITY, how="left",
                      validate="many_to_one")
    if rows.identity.isna().any():
        raise ValueError("mapped identity missing from declared catalog")
    rows["identity"] = rows.identity.astype(int)
    rows["day_index"] = (pd.to_datetime(rows.day) - pd.Timestamp("2025-01-01")).dt.days
    if not rows.day_index.between(0, days - 1).all() or not rows.hour.between(0, 23).all():
        raise ValueError("source dates or hours outside tensor")
    values = np.zeros((days, len(identities), 24), dtype=np.float64)
    np.add.at(values, (rows.day_index, rows.identity, rows.hour), rows.expected_count)
    summed = aggregate(values, identities, 0)
    checked = summed.merge(labels, on=["route", "date", "hour"], how="outer",
                           validate="one_to_one")
    if len(checked) != len(summed) or checked.boardings.isna().any():
        raise ValueError("full reconciled label grid required")
    if not np.allclose(checked.boardings, checked.prediction, atol=1e-6, rtol=0):
        raise ValueError("pseudo-counts do not reconcile to actual route-hour labels")
    train = values.copy()
    invalid = rows.loc[~rows.training_eligible, ["day_index", "route", "hour"]].drop_duplicates()
    for row in invalid.itertuples(index=False):
        train[row.day_index, identities.route.to_numpy() == row.route, row.hour] = np.nan
    return StopData(values, train, identities, {
        "source_version": source_version, "source_rows": len(rows), "mass": float(values.sum()),
        "row_eligible_mass": float(rows.loc[rows.training_eligible, "expected_count"].sum()),
        "block_eligible_mass": float(np.nansum(train)), "blocked_route_hours": len(invalid),
        "zero_policy": "inferred zeros only after complete route-hour mass reconciliation",
        "target_kind": "uncalibrated inferred pseudo-count; never observed stop labels",
        "geometry": "retrospective declared catalog; ambiguous sequence omitted",
        "identity_universe": "declared catalog plus ten unknown buckets; not target-row support",
    })


def profile(history: FloatArray, origin: int, days: int, window: int) -> FloatArray:
    if origin < 1 or origin > len(history) or days < 1 or origin + days > 365 or window < 1:
        raise ValueError("invalid profile dates or window")
    past = history[max(0, origin - window):origin]
    types = DAY_TYPE[max(0, origin - window):origin]
    predictions = np.zeros((days, history.shape[1], 24))
    for kind in range(3):
        selected = past[types == kind]
        counts = np.isfinite(selected).sum(axis=0)
        means = np.divide(np.nansum(selected, axis=0), counts,
                          out=np.zeros(counts.shape), where=counts > 0)
        predictions[DAY_TYPE[origin:origin + days] == kind] = means
    return predictions


def features(history: FloatArray, identities: Any, origin: int, days: int) -> Any:
    """Dynamic covariates are frozen strictly before the forecast origin."""
    if origin < 28 or origin + days > 365:
        raise ValueError("features require 28 past days and 2025 target dates")
    count = len(identities)
    d = np.repeat(np.arange(origin, origin + days), count * 24)
    s = np.tile(np.repeat(np.arange(count), 24), days)
    h = np.tile(np.arange(24), days * count)
    out = pd.DataFrame({"route": identities.route.to_numpy()[s], "stop": s,
                        "direction": identities.direction.to_numpy()[s], "hour": h,
                        "weekday": DOW[d], "day_type": DAY_TYPE[d], "holiday": HOLIDAY[d],
                        "month": MONTH[d], "lead": d - origin + 1,
                        "lat": identities.lat.fillna(0).to_numpy()[s],
                        "lon": identities.lon.fillna(0).to_numpy()[s]})
    for window in (28, 56, 84):
        out[f"profile_{window}"] = profile(history, origin, days, window).ravel()
    out["geometry_missing"] = identities.lat.isna().to_numpy()[s]
    out["unallocated"] = identities.stop_id.astype(str).str.startswith("unallocated").to_numpy()[s]
    out["seen"] = (np.nansum(history[:origin], axis=(0, 2)) > 0)[s]
    route = aggregate(profile(history, origin, days, 56), identities, origin)
    route["day_index"] = (pd.to_datetime(route.date) - pd.Timestamp("2025-01-01")).dt.days
    out["route_profile"] = out[["route", "hour"]].assign(day_index=d).merge(
        route, on=["route", "hour", "day_index"], how="left", validate="many_to_one"
    ).prediction.to_numpy()
    return out


def training_frame(
    data: StopData, outer: int, sample_per_origin: int = 60000,
) -> tuple[Any, FloatArray]:
    if outer < 60 or outer > len(data.train):
        raise ValueError("training origin outside available history")
    xs, ys = [], []
    rng = np.random.default_rng(20260927)
    for origin in range(59, outer, 30):
        days = min(61, outer - origin)
        x = features(data.train, data.identities, origin, days)
        y = data.train[origin:origin + days].ravel()
        valid = np.flatnonzero(np.isfinite(y))
        chosen = rng.choice(valid, min(sample_per_origin, len(valid)), replace=False)
        xs.append(x.iloc[chosen])
        ys.append(y[chosen])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def calendar_stop_profile(
    history: FloatArray, origin: int, days: int, *, blend: bool,
) -> FloatArray:
    """Linear per-stop counterpart of the existing strong route calendar baseline."""
    if origin < 1 or origin > len(history) or origin + days > 365 or days < 1:
        raise ValueError("invalid calendar profile dates")
    past = history[:origin]
    summer = np.isin(MONTH, (6, 7, 8))
    output = np.zeros((days, history.shape[1], 24))

    def means(mask: Any) -> tuple[FloatArray, Any]:
        selected = past[mask]
        count = np.isfinite(selected).sum(axis=0)
        mean = np.divide(np.nansum(selected, axis=0), count,
                         out=np.zeros(count.shape), where=count > 0)
        return mean, count > 0

    for lead in range(days):
        target = origin + lead
        same_season = summer[:origin] == summer[target]
        weekday = DOW[:origin] == DOW[target]
        fallback, _ = means(weekday)
        seasonal, supported = means(same_season & weekday)
        baseline = np.where(supported, seasonal, fallback)
        daytype, supported_type = means(same_season & (DAY_TYPE[:origin] == DAY_TYPE[target]))
        civil = date(2025, 1, 1) + timedelta(days=target)
        if civil in OFF | WORK:
            baseline = np.where(supported_type, daytype, baseline)
        elif blend:
            baseline = np.where(supported_type, .5 * baseline + .5 * daytype, baseline)
        output[lead] = baseline
    return output


def predict(data: StopData, origin: int, days: int, model_name: str) -> FloatArray:
    """Fit one independent count regressor; route totals are never imposed."""
    if model_name in {"calendar_profile", "calendar_weekday_blend"}:
        return calendar_stop_profile(data.train, origin, days,
                                     blend=model_name == "calendar_weekday_blend")
    if model_name in {"profile28", "profile56", "profile84"}:
        return profile(data.train, origin, days, int(model_name[7:]))
    if model_name not in {"cat_mae", "cat_rmse", "cat_residual",
                          "cat_relative84", "cat_relative84_shrunk"}:
        raise ValueError("unknown stop model")
    from catboost import CatBoostRegressor  # type: ignore[import-untyped]

    xtrain, ytrain = training_frame(data, origin)
    xfuture = features(data.train, data.identities, origin, days)
    residual = model_name == "cat_residual"
    model = CatBoostRegressor(iterations=250, depth=6, learning_rate=.08,
                              loss_function="MAE" if model_name == "cat_mae" else "RMSE",
                              random_seed=20260927, thread_count=4, verbose=False,
                              allow_writing_files=False, allow_const_label=True)
    target = ytrain - xtrain.profile_56.to_numpy() if residual else ytrain
    relative = model_name.startswith("cat_relative84")
    if relative:
        target = np.clip(ytrain / (xtrain.profile_84.to_numpy() + 1), 0, 5)
    model.fit(xtrain, target, cat_features=["route", "stop", "direction"])
    pred = np.asarray(model.predict(xfuture), dtype=float)
    if residual:
        pred += xfuture.profile_56.to_numpy()
    if relative:
        base = xfuture.profile_84.to_numpy()
        pred = (base + 1) * pred if model_name == "cat_relative84" else (
            base * (1 + .25 * (np.clip(pred, .8, 1.2) - 1))
        )
    return np.maximum(0, pred).reshape(days, len(data.identities), 24)


def evaluate(data: StopData, prediction: FloatArray, origin: int) -> dict[str, Any]:
    """Report actual route score separately from mapped pseudo-label error."""
    actual = data.values[origin:origin + len(prediction)]
    if actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid evaluation window or predictions")
    route_y = aggregate(actual, data.identities, origin)
    route_p = aggregate(prediction, data.identities, origin)
    error = np.abs(route_y.prediction.to_numpy() - np.rint(route_p.prediction.to_numpy()))
    total = float(actual.sum())
    eligible = np.isfinite(data.train[origin:origin + len(prediction)])
    known = ~data.identities.stop_id.astype(str).str.startswith("unallocated").to_numpy()
    unseen = data.values[:origin].sum(axis=(0, 2)) == 0
    per_route = {}
    for route in LABEL_ROUTES:
        mask = route_y.route.to_numpy() == route
        denominator = float(route_y.loc[mask, "prediction"].sum())
        per_route[str(route)] = max(0., 1 - float(error[mask].sum()) / denominator) \
            if denominator else None
    slices: dict[str, Any] = {}
    day_offsets = (pd.to_datetime(route_y.date)
                   - pd.Timestamp(date(2025, 1, 1) + timedelta(days=origin))).dt.days
    for kind, groups in (("month", pd.to_datetime(route_y.date).dt.month),
                         ("lead_week", day_offsets // 7 + 1)):
        slices[kind] = {}
        for group in sorted(groups.unique()):
            mask = groups.to_numpy() == group
            denominator = float(route_y.loc[mask, "prediction"].sum())
            slices[kind][str(group)] = {
                "actual": denominator, "absolute_error": float(error[mask].sum()),
                "score": max(0., 1 - float(error[mask].sum()) / denominator)
                if denominator else None,
            }
    known_mass = float(actual[:, known].sum())
    eligible_mass = float(actual[eligible].sum())
    return {
        "route_score": max(0., 1 - float(error.sum()) / total) if total else None,
        "pseudo_wape_all": float(np.abs(prediction - actual).sum()) / total if total else None,
        "pseudo_wape_eligible": float(np.abs(prediction[eligible] - actual[eligible]).sum())
        / eligible_mass if eligible_mass else None,
        "pseudo_wape_known": float(np.abs(prediction[:, known] - actual[:, known]).sum())
        / known_mass if known_mass else None,
        "known_mass": known_mass, "unallocated_mass": float(actual[:, ~known].sum()),
        "unseen_identity_mass": float(actual[:, unseen].sum()), "route_scores": per_route,
        "slices": slices,
    }


def run_experiment(
    data: StopData, output: Any, reuse: Any = None, *,
    reuse_code_sha256: str | None = None, reuse_scores_sha256: str | None = None,
) -> None:
    """Select on May/July; September remains an explicitly non-blind diagnostic."""
    import hashlib
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_audit.json").write_text(json.dumps(data.audit, indent=2))
    candidates = ("profile28", "profile56", "profile84", "cat_mae", "cat_rmse",
                  "cat_residual", "cat_relative84", "cat_relative84_shrunk",
                  "calendar_profile", "calendar_weekday_blend")
    results: list[dict[str, Any]] = []
    reused_run = None
    if reuse is not None:
        prior = json.loads((reuse / "run.json").read_text())
        score_hash = hashlib.sha256((reuse / "scores.json").read_bytes()).hexdigest()
        if prior["code_sha256"] != reuse_code_sha256 or score_hash != reuse_scores_sha256:
            raise ValueError("explicit compatible code and score checksums required for reuse")
        if prior["source_manifest_sha256"] != data.audit["source_manifest_sha256"]:
            raise ValueError("reused scores belong to a different source")
        results = json.loads((reuse / "scores.json").read_text())
        keys = [(r["model"], r["origin"]) for r in results]
        if len(keys) != len(set(keys)) or any(
            name not in candidates or day not in {"2025-05-01", "2025-07-01", "2025-09-01"}
            for name, day in keys
        ):
            raise ValueError("reused scores contain duplicate or unknown model-origin keys")
        if any(not np.isfinite(r["route_score"]) or not 0 <= r["route_score"] <= 1
               for r in results):
            raise ValueError("reused scores contain invalid metrics")
        reused_run = {"run_sha256": hashlib.sha256((reuse / "run.json").read_bytes()).hexdigest(),
                      "code_sha256": prior["code_sha256"],
                      "scores_sha256": hashlib.sha256(
                          (reuse / "scores.json").read_bytes()).hexdigest()}
    for origin in (120, 181, 243):
        for name in candidates:
            origin_date = str(date(2025, 1, 1) + timedelta(days=origin))
            if any(r["model"] == name and r["origin"] == origin_date for r in results):
                continue
            prediction = predict(data, origin, 61, name)
            result = {"model": name, "origin": str(date(2025, 1, 1) + timedelta(days=origin)),
                      **evaluate(data, prediction, origin)}
            results.append(result)
            print(json.dumps(result), flush=True)
            (output / "scores.json").write_text(json.dumps(results, indent=2))
    development = pd.DataFrame([r for r in results if r["origin"] < "2025-09-01"])
    selected = str(development.groupby("model").route_score.mean().idxmax())
    prediction = predict(data, 304, 61, selected)
    route = aggregate(prediction, data.identities, 304)
    route["prediction"] = np.rint(route.prediction).astype(int)
    if len(route) != 14640 or route.duplicated(["route", "date", "hour"]).any():
        raise ValueError("invalid submission grid")
    route.to_csv(output / "submission.csv", sep=";", index=False)
    count = len(data.identities)
    stop = pd.DataFrame({
        "route": np.tile(np.repeat(data.identities.route, 24), 61),
        "direction": np.tile(np.repeat(data.identities.direction, 24), 61),
        "stop_id": np.tile(np.repeat(data.identities.stop_id, 24), 61),
        "hour": np.tile(np.arange(24), 61 * count),
        "date": np.repeat(pd.date_range("2025-11-01", periods=61).date, count * 24),
        "prediction": prediction.ravel(),
        "label_origin": "forecast_of_inferred_or_unallocated_counts",
    })
    stop["model_version"] = FEATURE_VERSION + ":" + selected
    stop["generated_at"] = datetime.now(UTC).isoformat()
    stop["lower_bound"] = np.nan
    stop["upper_bound"] = np.nan
    stop["uncertainty_status"] = "unavailable"
    stop.to_csv(output / "stop_predictions.csv.gz", index=False)
    (output / "run.json").write_text(json.dumps({
        "reused_run": reused_run,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(candidates).encode()).hexdigest(),
        "source_manifest_sha256": data.audit["source_manifest_sha256"],
        "submission_sha256": hashlib.sha256((output / "submission.csv").read_bytes()).hexdigest(),
        "selected": selected, "selection": "May+July mean actual route score",
        "diagnostic_blind": False, "timezone": "Europe/Moscow", "seed": 20260927,
        "feature_version": FEATURE_VERSION, "horizon_days": 61,
        "uncertainty": "unavailable; point forecasts only",
        "sampling": "up to 60000 examples per monthly origin; no future features",
        "stop_ground_truth": False, "aggregate_constraint": "none; raw independent stop sums",
    }, indent=2))


def load_dataset(source: Any, archive: Any, proof: Any) -> StopData:
    """Checksum-bind new mapping and organizer labels before inferred zero filling."""
    import hashlib
    import json

    from tramflow_ml.competition import complete_labels, load_labels

    def digest(path: Any) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    manifest = json.loads((source / "manifest.json").read_text())
    reconciliation = json.loads(proof.read_text())
    if not reconciliation.get("union_all_labels_exact_match"):
        raise ValueError("raw validation reconciliation missing")
    if digest(archive) != reconciliation["archive_sha256"]:
        raise ValueError("archive differs from raw reconciliation")
    frames = []
    hashes = {}
    for split in ("train", "development", "diagnostic"):
        info = manifest["files"][f"soft_{split}"]
        path = source / info["name"]
        checksum = digest(path)
        if checksum != info["sha256"]:
            raise ValueError("mapped source checksum mismatch")
        frame = pd.read_csv(path, dtype={"stop_id": str})
        if len(frame) != info["rows"]:
            raise ValueError("mapped source row count mismatch")
        frames.append(frame)
        hashes[info["name"]] = checksum
    labels = complete_labels(load_labels(archive), date(2025, 1, 1), date(2025, 10, 31),
                             missing_as_zero=True)
    catalog_hash = digest(source / "stop_catalog.csv")
    if catalog_hash != manifest["catalog_sha256"]:
        raise ValueError("catalog checksum mismatch")
    coordinates = pd.read_csv(source / "stop_catalog.csv", dtype={"stop_id": str})
    data = prepare(pd.concat(frames, ignore_index=True), coordinates, labels,
                   complete_decode=manifest.get("complete") is True, raw_reconciled=True,
                   source_version=str(manifest["schema_version"]))
    data.audit.update({"source_manifest_sha256": digest(source / "manifest.json"),
                       "source_file_hashes": hashes,
                       "catalog_sha256": digest(source / "stop_catalog.csv"),
                       "archive_sha256": reconciliation["archive_sha256"]})
    return data


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "archive", "proof", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--reuse-scores-from", type=Path)
    parser.add_argument("--reuse-code-sha256")
    parser.add_argument("--reuse-scores-sha256")
    args = parser.parse_args()
    run_experiment(load_dataset(args.source, args.archive, args.proof), args.output,
                   args.reuse_scores_from, reuse_code_sha256=args.reuse_code_sha256,
                   reuse_scores_sha256=args.reuse_scores_sha256)


if __name__ == "__main__":
    main()

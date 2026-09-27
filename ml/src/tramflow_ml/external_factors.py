"""Offline, source-switchable covariates frozen before each forecast origin."""

import argparse
import gzip
import hashlib
import html
import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.competition import feature_frame, hour_grid, make_training_frame
from tramflow_ml.competition_profile import robust_hour_shape_predict

VERSION = "external-factors.v1"
SOURCES = frozenset({"calendar", "weather", "traffic", "news"})
BASE_FIELDS = ["route", "hour", "day_of_week", "lead_day", "route_hour_28", "route_hour_56",
               "route_hour_all", "route_day_28", "route_day_56"]
CALENDAR_FIELDS = ["month", "day_of_year", "is_summer",
                   "calendar_day_type", "calendar_exception"]
WEATHER_FIELDS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]
CONFIG = {"iterations": 200, "depth": 5, "learning_rate": .05,
          "loss_function": "MAE", "random_seed": 20260927, "thread_count": 4,
          "verbose": False, "allow_writing_files": False, "l2_leaf_reg": 10}


def moscow_publication_day(value: str) -> date:
    """Civil publication date never depends on the collector host timezone."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("source timestamp must carry timezone")
    return stamp.tz_convert("Europe/Moscow").date()


def parse_traffic_page(content: str) -> list[dict[str, Any]]:
    """Extract public numerical facts; ambiguous traffic mentions remain unavailable."""
    output = []
    for block in re.split(r'<div class="tgme_widget_message ', content)[1:]:
        identity = re.search(r'data-post="(DtOperativno/\d+)"', block)
        timestamp = re.search(r'<time[^>]+datetime="([^"]+)"', block)
        body = re.search(
            r'<div class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>',
            block, re.S,
        )
        if not (identity and timestamp and body):
            continue
        stamp = pd.Timestamp(timestamp[1])
        if stamp.tzinfo is None:
            raise ValueError("source timestamp must carry timezone")
        text = html.unescape(re.sub("<[^>]+>", " ", body[1]))
        text = re.sub(r"\s+", " ", text).strip()
        scores = re.findall(r"\b(10|[0-9])\s+балл?[а-я]*", text)
        current = re.search(
            r"(оценивает|оценива[ею]тся|на дорогах сейчас|"
            r"на дорогах\s*[—–:-]?\s*\d+\s+балл?|загруженность.*балл?)", text, re.I,
        )
        speeds = []
        for match in re.finditer(r"\b(\d{1,3})\s*км/ч", text):
            prefix = text[max(0, match.start() - 95):match.start()].lower()
            if ("скорость" in prefix and ("движени" in prefix or "поток" in prefix)
                    and "ветер" not in prefix):
                speeds.append(match[1])
        score = int(scores[0]) if len(set(scores)) == 1 and current else None
        speed = int(speeds[0]) if len(set(speeds)) == 1 else None
        if "ЦОДД" not in text or len(set(scores)) > 1 or len(set(speeds)) > 1:
            score = speed = None
        if re.search(r"(?:ожидаем|ожидается|прогноз).*\d+\s+балл", text, re.I):
            score = speed = None
        output.append({
            "source_url": "https://t.me/" + identity[1],
            "published_at": stamp.isoformat(),
            "edited": bool(re.search(r"\bedited\b", block)),
            "congestion_score": score, "mean_speed_kmh": speed,
            "closure_notice": bool(re.search(
                r"перекры|закрыт.*движени|огранич.*движени", text, re.I,
            )),
            "incident_notice": bool(re.search(r"\bДТП\b|авари|задержива[ею]тся", text, re.I)),
        })
    return output


def _dated_posts(posts: pd.DataFrame, origin: date, days: int) -> pd.DataFrame:
    if posts.empty:
        return posts.copy()
    result = posts.copy()
    stamps = pd.to_datetime(result.published_at, utc=True).dt.tz_convert("Europe/Moscow")
    result["date"] = stamps.dt.date
    result["hour"] = stamps.dt.hour
    result["day_of_week"] = stamps.dt.dayofweek
    return result.loc[(result.date < origin) & (result.date >= origin - timedelta(days=days))
                      & ~result.edited.astype(bool)]


def source_features(
    frame: pd.DataFrame, origin: date, *, enabled: frozenset[str],
    weather: pd.DataFrame, posts: pd.DataFrame,
) -> pd.DataFrame:
    """Disabled sources contribute no columns; all dynamic observations precede origin."""
    if not enabled <= SOURCES:
        raise ValueError("unknown external source")
    if frame.empty or (frame.date < origin).any():
        raise ValueError("target dates must begin at or after origin")
    result = frame[BASE_FIELDS].copy().reset_index(drop=True)
    keys = frame[["route", "hour", "day_of_week"]].reset_index(drop=True)
    if "calendar" in enabled:
        result[CALENDAR_FIELDS] = frame[CALENDAR_FIELDS].reset_index(drop=True)
    if "weather" in enabled:
        past = weather.loc[(weather.date < origin)
                           & (weather.date >= origin - timedelta(days=28))]
        means = past.groupby(["route", "hour"])[WEATHER_FIELDS].mean().reset_index()
        joined = keys[["route", "hour"]].merge(means, how="left", validate="many_to_one")
        for column in WEATHER_FIELDS:
            result["weather_" + column] = joined[column].fillna(-999.)
        result["weather_available"] = joined.temperature_2m.notna().astype(int)
    if "traffic" in enabled:
        past = _dated_posts(posts, origin, 56)
        for column in ("congestion_score", "mean_speed_kmh"):
            if past.empty:
                result["traffic_" + column] = -1.
                result["traffic_" + column + "_support"] = 0
                continue
            observations = past.loc[past[column].notna()]
            summaries = observations.groupby(["day_of_week", "hour"])[column].agg(
                mean="mean", support="count",
            ).reset_index()
            joined = keys[["day_of_week", "hour"]].merge(
                summaries, how="left", validate="many_to_one",
            )
            result["traffic_" + column] = joined["mean"].fillna(-1.)
            result["traffic_" + column + "_support"] = joined.support.fillna(0)
    if "news" in enabled:
        past = _dated_posts(posts, origin, 28)
        for column in ("closure_notice", "incident_notice"):
            result["news_" + column] = float(past[column].sum()) / 28 if len(past) else 0.
        result["news_post_count"] = len(past)
    result["route"] = result.route.astype(int).astype(str)
    return result.fillna(-999.)


def load_labels(directory: Path) -> pd.DataFrame:
    """Only the checksum-bound, fully reconciled 72,960-key export is accepted."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("complete") is not True
            or manifest.get("timezone") != "Europe/Moscow"
            or manifest.get("target") != "successful_validation_count_route_date_hour"
            or manifest.get("date_range") != ["2025-01-01", "2025-10-31"]
            or manifest.get("archive_sha256") !=
            "7e34e56b93f7379cb5abe9aca8e87967cb986111b9939caead5a3fea6c296e5a"
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("reconciliation_sha256", ""))):
        raise ValueError("incompatible label provenance contract")
    filename = "route-hour-labels.csv.gz"
    raw = (directory / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["files"][filename]["sha256"]:
        raise ValueError("label export checksum differs")
    labels = pd.read_csv(directory / filename)
    labels["date"] = pd.to_datetime(labels.date).dt.date
    if (len(labels) != 72960 or labels.duplicated(["route", "date", "hour"]).any()
            or labels.date.min() != date(2025, 1, 1)
            or labels.date.max() != date(2025, 10, 31)):
        raise ValueError("incomplete January-October label export")
    expected = hour_grid(date(2025, 1, 1), date(2025, 10, 31))
    keys = ["route", "date", "hour"]
    if (not set(map(tuple, labels[keys].to_numpy())) ==
            set(map(tuple, expected[keys].to_numpy()))
            or not np.isfinite(labels.boardings).all() or (labels.boardings < 0).any()
            or labels.boardings.sum() != manifest.get("target_mass")
            or manifest.get("target_mass") != 59667191):
        raise ValueError("label grid or target mass differs")
    return labels


def write_bundle_manifest(output: Path, sources: dict[str, Path]) -> None:
    """Bind the experimental artifact and exact source files without raw records."""
    artifact = output / "model-variants.json.gz"
    manifest = {
        "schema": "external-forecast-variants-manifest.v1", "file": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "cutoff": "2025-10-31", "grid_rows": 14640, "source_version": VERSION,
        "status": "experimental_not_promoted",
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in sources.items()},
    }
    (output / "model-variants-manifest.json").write_text(json.dumps(manifest, indent=2))


def run(labels: pd.DataFrame, weather: pd.DataFrame, posts: pd.DataFrame, out: Path) -> None:
    """Fixed six-way ablation; no tuning or production promotion from these results."""
    from catboost import CatBoostRegressor  # type: ignore[import-untyped]

    variants = {"base": frozenset(), **{source: frozenset({source}) for source in sorted(SOURCES)},
                }
    origins = [date(2025, 5, 1), date(2025, 7, 1), date(2025, 9, 1)]
    out.mkdir(parents=True, exist_ok=True)
    plan = {"version": VERSION, "config": CONFIG, "origins": list(map(str, origins)),
            "horizons": [61, 1], "variants": {k: sorted(v) for k, v in variants.items()},
            "selection": "none; source ablation only, no promoted model",
            "acceptance": "report every origin, including losses; never claim four proven gains",
            "target": "organizer route-date-hour boardings, rounded once, WAPE-score",
            "one_day": "three sparse first-day diagnostics, no claim of daily-refit validation",
            "mix_rule": "mean(base plus selected branches), then round once; all=mean of five",
            "availability": "past-only civil dates; weather is revised historical data, "
                            "not vintage",
            "weather": "28-day pre-origin route-hour means, no target-period actual weather",
            "traffic": "56-day pre-origin weekday-hour means and observation counts; "
                       "edited excluded",
            "news": "28-day prior official closure/incident publication frequency, edited excluded"}
    plan_path = out / "frozen_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("existing experiment plan differs")
    plan_path.write_text(json.dumps(plan, indent=2))
    results = []
    final_predictions: dict[str, np.ndarray] = {}
    final_keys: list[list[Any]] = []
    for origin in [*origins, date(2025, 11, 1)]:
        history = labels.loc[labels.date < origin]
        train = make_training_frame(history, origin - timedelta(days=1))
        training_origins = (pd.to_datetime(train.date)
                            - pd.to_timedelta(train.lead_day - 1, unit="D")).dt.date
        future = feature_frame(history, origin)
        truth = future[["route", "date", "hour"]].merge(
            labels, how="left", validate="one_to_one",
        ).boardings.to_numpy(dtype=float)
        anchor = robust_hour_shape_predict(history, future, "calendar_robust_28", group="route")
        predictions = {"incumbent_robust28": anchor}
        for name, sources in variants.items():
            pieces = []
            for training_origin in sorted(training_origins.unique()):
                part = train.loc[training_origins == training_origin]
                x = source_features(part, training_origin, enabled=sources,
                                    weather=weather, posts=posts)
                x.index = part.index
                pieces.append(x)
            xtrain = pd.concat(pieces).sort_index()
            xfuture = source_features(future, origin, enabled=sources,
                                      weather=weather, posts=posts)
            model = CatBoostRegressor(**CONFIG, cat_features=["route"])
            model.fit(xtrain, train.boardings.to_numpy(dtype=float))
            pred = np.maximum(0, np.asarray(model.predict(xfuture), dtype=float))
            pred[future.route.isin(history.groupby("route").boardings.sum().loc[
                lambda x: x == 0
            ].index)] = 0
            predictions[name] = pred
        if origin == date(2025, 11, 1):
            final_predictions = {k: v for k, v in predictions.items() if k in variants}
            final_keys = [[int(r), str(d), int(h)]
                          for r, d, h in future[["route", "date", "hour"]].itertuples(index=False)]
            break
        raw_branches = {k: v.copy() for k, v in predictions.items() if k in variants}
        predictions["all"] = np.mean(list(raw_branches.values()), axis=0)
        for name in SOURCES:
            predictions[name] = .5 * raw_branches["base"] + .5 * raw_branches[name]
        for name, pred in predictions.items():
            for horizon in (61, 1):
                mask = (future.lead_day <= horizon).to_numpy()
                errors = np.abs(truth[mask] - np.rint(pred[mask]))
                denominator = float(truth[mask].sum())
                slices = {}
                for route in sorted(future.route.unique()):
                    selected = mask & (future.route.to_numpy() == route)
                    mass = float(truth[selected].sum())
                    err = float(np.abs(truth[selected] - np.rint(pred[selected])).sum())
                    slices[str(route)] = {"actual": mass, "absolute_error": err,
                                          "score": max(0., 1 - err / mass) if mass else None}
                row = {"origin": str(origin), "horizon_days": horizon, "variant": name,
                       "score": max(0., 1 - float(errors.sum()) / denominator),
                       "absolute_error": float(errors.sum()), "actual": denominator,
                       "route_slices": slices}
                results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "route_slices"}), flush=True)
        (out / "scores.json").write_text(json.dumps(results, indent=2))
    bundle = {
        "schema": "external-forecast-variants.v1", "generated_at": datetime.now(UTC).isoformat(),
        "cutoff": "2025-10-31", "source_version": VERSION,
        "mix_rule": "arithmetic mean of base and each enabled branch; round once after mixing",
        "keys": final_keys,
        "branches": {name: values.tolist() for name, values in final_predictions.items()},
        "validation": {"origins": plan["origins"], "scores": results,
                       "single_source_scores": "mean(base, source branch)", "blind": False},
        "limitations": ["Experimental unpromoted forecast, not approved incumbent.",
                        "Traffic is citywide sparse public snapshots, not route-level telemetry.",
                        "News counts describe lagged official incident/closure publications.",
                        "Historical weather and posts lack original publication-vintage proof.",
                        "One-day metrics are three first-day slices, "
                        "not daily rolling validation."],
    }
    with gzip.GzipFile(filename=str(out / "model-variants.json.gz"), mode="wb", mtime=0) as stream:
        stream.write(json.dumps(bundle, separators=(",", ":")).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = load_labels(args.labels)
    weather = pd.read_csv(args.weather, sep=";")
    weather["date"] = pd.to_datetime(weather.date).dt.date
    posts = pd.read_json(args.posts, lines=True)
    run(labels, weather, posts, args.output)
    write_bundle_manifest(args.output, {
        "observations.jsonl": args.posts,
        "traffic_manifest.json": args.posts.parent / "manifest.json",
        "route_labels_manifest.json": args.labels / "manifest.json",
        "route_labels.csv.gz": args.labels / "route-hour-labels.csv.gz",
        "route_weather.csv.gz": args.weather,
        "external_factors.py": Path(__file__),
        "frozen_plan.json": args.output / "frozen_plan.json",
        "scores.json": args.output / "scores.json",
    })


if __name__ == "__main__":
    main()

"""Frozen statistical comparison with strictly pre-origin direct 61-day forecasts."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.competition import (
    LABEL_ROUTES,
    calendar_weekday_blend_predict,
    complete_labels,
    feature_frame,
    hour_grid,
    load_labels,
    rounded,
    score,
    write_submission,
)
from tramflow_ml.competition_cli import _digest
from tramflow_ml.route_models.features import OFF, WORK

ORIGINS = tuple(date(2025, month, 1) for month in (4, 5, 6, 7, 8, 9))


@dataclass(frozen=True)
class Config:
    name: str
    statistic: str = "mean"
    window: int = 0
    half_life: int = 28
    seasonal: bool = False
    daytype_weight: float = 0.0
    correction: float = 0.0
    baseline_weight: float = 0.0
    calendar_months: int = 0
    projection: str = ""
    alpha: float = 0.2
    damping: float = 0.9


def configurations() -> tuple[Config, ...]:
    result = [Config("baseline", statistic="baseline")]
    for window in (7, 14, 28, 56, 84, 112, 0):
        for statistic in ("mean", "median", "trim10", "q40", "q60"):
            result.append(Config(f"{statistic}_w{window}", statistic, window))
    for seasonal in (False, True):
        for half_life in (7, 14, 28, 56, 112):
            for weight in (0.0, 0.5, 1.0):
                result.append(Config(
                    f"exp_h{half_life}_s{int(seasonal)}_d{weight}", "exp",
                    half_life=half_life, seasonal=seasonal, daytype_weight=weight,
                ))
        for statistic in ("mean", "median", "trim10", "q40", "q60"):
            for weight in (0.0, 0.5, 1.0):
                result.append(Config(
                    f"{statistic}_s{int(seasonal)}_d{weight}", statistic,
                    seasonal=seasonal, daytype_weight=weight,
                ))
    for correction in (0.25, 0.5, 1.0):
        for window in (28, 56, 84):
            result.append(Config(
                f"level_w{window}_c{correction}", window=window, seasonal=True,
                daytype_weight=0.5, correction=correction,
            ))
    for weight in (0.25, 0.5, 0.75):
        for half_life in (14, 28, 56):
            result.append(Config(
                f"blend_h{half_life}_b{weight}", "exp", half_life=half_life,
                seasonal=True, daytype_weight=0.5, baseline_weight=weight,
            ))
    return tuple(result)


def monthly_configurations() -> tuple[Config, ...]:
    """Second preregistered phase; every validation window was already reused in v1."""
    result = [Config("baseline", statistic="baseline")]
    for statistic in ("mean", "median"):
        result.append(Config(f"previous_month_{statistic}", statistic, calendar_months=1))
    for months in (2, 3, 6, 0):
        result.append(Config(f"equal_months_{months}", "month_equal", calendar_months=months))
    for seasonal in (False, True):
        for half_life in (1, 2, 3):
            result.append(Config(
                f"month_exp_h{half_life}_s{int(seasonal)}", "month_exp",
                half_life=half_life, seasonal=seasonal,
            ))
    return tuple(result)


def extended_configurations() -> tuple[Config, ...]:
    """Phase 3 fixed grid; monthly baseline selection uses April-August origins only."""
    result = [Config("baseline", statistic="baseline"),
              Config("v1_robust", seasonal=True)]
    for statistic in ("mean", "median"):
        for months in (1, 2, 3, 6, 0):
            for seasonal in (False, True):
                for weight in (0.0, 0.5, 1.0):
                    result.append(Config(
                        f"monthly_{statistic}_m{months}_s{int(seasonal)}_d{weight}",
                        statistic, seasonal=seasonal, daytype_weight=weight,
                        calendar_months=months or 99,
                    ))
    for months in (2, 3, 6, 0):
        for seasonal in (False, True):
            for weight in (0.0, 0.5, 1.0):
                result.append(Config(
                    f"monthly_equal_m{months}_s{int(seasonal)}_d{weight}", "month_equal",
                    seasonal=seasonal, daytype_weight=weight, calendar_months=months,
                ))
    for half_life in (1, 2, 3):
        for seasonal in (False, True):
            for weight in (0.0, 0.5, 1.0):
                result.append(Config(
                    f"monthly_exp_h{half_life}_s{int(seasonal)}_d{weight}", "month_exp",
                    half_life=half_life, seasonal=seasonal, daytype_weight=weight,
                ))
    for method in ("mean", "median", "trim10"):
        for window in (28, 56, 84):
            result.append(Config(f"daily_{method}_w{window}", window=window,
                                 projection=method, daytype_weight=0.5))
    for alpha in (0.1, 0.3, 0.5):
        for window in (28, 84):
            result.append(Config(f"daily_ses_a{alpha}_w{window}", window=window,
                                 projection="ses", alpha=alpha, daytype_weight=0.5))
    for alpha in (0.1, 0.3):
        for damping in (0.5, 0.8, 0.95):
            result.append(Config(f"daily_holt_a{alpha}_d{damping}", window=84,
                                 projection="holt", alpha=alpha, damping=damping,
                                 daytype_weight=0.5))
    return tuple(result)


def _project_daily(
    cube: np.ndarray, days: list[date], origin: date, config: Config, horizon: int,
) -> np.ndarray:
    """Fit calendar-normalized daily level, then allocate with observed mean hourly shares."""
    selected = np.array([(origin - day).days <= config.window for day in days])
    history = cube[:, selected, :]
    categories = np.array([_daytype(day) for day, keep in zip(days, selected, strict=True) if keep])
    daily = history.sum(axis=2)
    normalizers = np.zeros_like(daily)
    profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for category in (0, 1, 2):
        mask = categories == category
        if not mask.any():
            mask = np.ones(len(categories), dtype=bool)
        totals = daily[:, mask]
        base = totals.mean(axis=1)
        if config.projection == "median":
            base = np.median(totals, axis=1)
        elif config.projection == "trim10":
            trim = int(totals.shape[1] * 0.1)
            base = np.sort(totals, axis=1)[:, trim:totals.shape[1] - trim].mean(axis=1)
        shares = np.divide(history[:, mask, :], totals[:, :, None],
                           out=np.zeros_like(history[:, mask, :]), where=totals[:, :, None] > 0)
        share = shares.mean(axis=1)
        share_sum = share.sum(axis=1)
        share = np.divide(share, share_sum[:, None], out=np.zeros_like(share),
                          where=share_sum[:, None] > 0)
        normalizers[:, categories == category] = base[:, None]
        profiles[category] = base, share
    normalized = np.divide(daily, normalizers, out=np.ones_like(daily), where=normalizers > 0)
    level = np.ones(len(LABEL_ROUTES))
    trend = np.zeros(len(LABEL_ROUTES))
    if config.projection in ("ses", "holt"):
        level = normalized[:, 0].copy()
        for index in range(1, normalized.shape[1]):
            previous = level.copy()
            projected = level + config.damping * trend if config.projection == "holt" else level
            level = config.alpha * normalized[:, index] + (1 - config.alpha) * projected
            if config.projection == "holt":
                trend = 0.05 * (level - previous) + 0.95 * config.damping * trend
    forecasts = []
    for lead in range(horizon):
        day = origin + timedelta(days=lead)
        base, shares = profiles[_daytype(day)]
        accumulated = sum(config.damping ** step for step in range(1, lead + 2))
        factor = np.clip(level + accumulated * trend, 0.8, 1.2)
        forecasts.append(base[:, None] * shares * factor[:, None])
    return np.stack(forecasts, axis=1).reshape(-1)


def _daytype(day: date) -> int:
    return (0 if day in WORK or (day.weekday() < 5 and day not in OFF)
            else 1 if day.weekday() == 5 and day not in OFF else 2)


def predict(
    labels: pd.DataFrame, origin: date, config: Config, horizon: int = 61,
) -> np.ndarray:
    """Require dense historical keys; future rows are discarded before statistics."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    past = labels.loc[labels.date < origin, ["route", "date", "hour", "boardings"]]
    if past.empty:
        raise ValueError("no history before origin")
    past = complete_labels(past, min(past.date), origin - timedelta(days=1))
    days = sorted(past.date.unique())
    cube = past.boardings.to_numpy(dtype=float).reshape(len(LABEL_ROUTES), len(days), 24)
    if config.projection:
        return _project_daily(cube, days, origin, config, horizon)
    baseline = None
    if config.statistic == "baseline" or config.baseline_weight:
        baseline = calendar_weekday_blend_predict(feature_frame(past, origin, horizon))
        if config.statistic == "baseline":
            return np.asarray(baseline, dtype=float)
    age = np.array([(origin - day).days for day in days])
    dow = np.array([day.weekday() for day in days])
    dtype = np.array([_daytype(day) for day in days])
    summer = np.array([day.month in (6, 7, 8) for day in days])
    recent = age <= config.window if config.window else np.ones(len(days), dtype=bool)
    month_age = np.array([(origin.year - day.year) * 12 + origin.month - day.month
                          for day in days])
    if config.calendar_months or config.statistic.startswith("month_"):
        recent &= month_age > 0
        if config.calendar_months:
            recent &= month_age <= config.calendar_months
        if not recent.any():
            raise ValueError("no completed calendar month available")

    def reduce(mask: np.ndarray) -> np.ndarray:
        values = cube[:, mask, :]
        if config.statistic in ("month_equal", "month_exp"):
            months = np.unique(month_age[mask])
            profiles = np.stack([cube[:, mask & (month_age == month), :].mean(axis=1)
                                 for month in months], axis=1)
            weights = (np.exp2(-months / config.half_life) if config.statistic == "month_exp"
                       else np.ones(len(months)))
            return np.asarray(np.average(profiles, axis=1, weights=weights))
        if config.statistic == "mean":
            return np.asarray(values.mean(axis=1))
        if config.statistic == "exp":
            weights = np.exp2(-age[mask] / config.half_life)
            return np.asarray(np.average(values, axis=1, weights=weights))
        if config.statistic in ("median", "q40", "q60"):
            quantile = {"median": 0.5, "q40": 0.4, "q60": 0.6}[config.statistic]
            return np.asarray(np.quantile(values, quantile, axis=1))
        if config.statistic == "trim10":
            ordered = np.sort(values, axis=1)
            trim = int(values.shape[1] * 0.1)
            return np.asarray(ordered[:, trim:values.shape[1] - trim, :].mean(axis=1))
        raise ValueError(f"unknown statistic {config.statistic}")

    forecasts = []
    cache: dict[tuple[int, int, bool, bool], np.ndarray] = {}
    for lead in range(horizon):
        day = origin + timedelta(days=lead)
        key = (day.weekday(), _daytype(day), day.month in (6, 7, 8), day in OFF | WORK)
        if key not in cache:
            parts = []
            for categories, target in ((dow, key[0]), (dtype, key[1])):
                mask = recent & (categories == target)
                seasonal = mask & (summer == key[2])
                if config.seasonal and seasonal.any():
                    mask = seasonal
                if not mask.any():
                    mask = categories == target
                if not mask.any():
                    mask = np.ones(len(days), dtype=bool)
                parts.append(reduce(mask))
            weight = 1.0 if key[3] else config.daytype_weight
            cache[key] = (1 - weight) * parts[0] + weight * parts[1]
        forecasts.append(cache[key])
    output = np.stack(forecasts, axis=1)
    if config.correction:
        recent_level = cube[:, age <= 14, :].mean(axis=(1, 2))
        old_mask = (age > 14) & (age <= 42)
        if old_mask.any():
            old_level = cube[:, old_mask, :].mean(axis=(1, 2))
            ratio = np.divide(recent_level, old_level, out=np.ones_like(old_level),
                              where=old_level > 0)
            factor = 1 + config.correction * (np.clip(ratio, 0.8, 1.2) - 1)
            output *= factor[:, None, None]
    result = output.reshape(-1)
    if baseline is not None:
        result = (1 - config.baseline_weight) * result + config.baseline_weight * baseline
    if not np.isfinite(result).all():
        raise ValueError("nonfinite prediction")
    return np.asarray(np.maximum(0, result), dtype=float)


def run(archive: Path, proof: Path, output: Path, *, phase: int = 1) -> None:
    archive_hash = _digest(archive)
    evidence = json.loads(proof.read_text())
    source = load_labels(archive)
    if not (
        evidence.get("archive_sha256") == archive_hash
        and evidence.get("union_all_labels_exact_match") is True
        and evidence.get("keys") == len(source) == 57551
        and evidence.get("target_sum") == int(source.boardings.sum()) == 59667191
    ):
        raise ValueError("legacy raw-reconciliation evidence does not match archive and labels")
    output.mkdir(parents=True, exist_ok=True)
    grid = {1: configurations, 2: monthly_configurations, 3: extended_configurations}[phase]()
    plan = {
        "version": f"statistical-sweep.v{phase}", "configs": [asdict(item) for item in grid],
        "origins": [str(item) for item in ORIGINS], "horizon_days": 61,
        "selection": "descending minimum origin score, descending mean score, ascending name",
        "development_origins": [str(origin) for origin in ORIGINS[:-1]],
        "diagnostic_origin": str(ORIGINS[-1]),
        "monthly_baseline_rule": "best monthly mean development min, then mean, then name",
        "archive_sha256": archive_hash, "timezone": "Europe/Moscow (repository assumption)",
        "target": "successful validations per route-date-hour",
        "test_status": ("all origins model-selection; September previously reused" if phase == 1
                        else f"secondary phase{phase}; all six origins reused after v1"),
        "missing_policy": "zeros after checksum-bound raw reconciliation; not proof of operation",
    }
    serialized = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    (output / "frozen_plan.json").write_text(serialized)
    labels = complete_labels(source, date(2025, 1, 1), date(2025, 10, 31), missing_as_zero=True)
    summaries = []
    slices = []
    for origin in ORIGINS:
        future = hour_grid(origin, origin + timedelta(days=60)).merge(
            labels, on=["route", "date", "hour"], validate="one_to_one")
        future["month"] = pd.to_datetime(future.date).dt.month
        future["lead_week"] = [(day - origin).days // 7 + 1 for day in future.date]
        for config in grid:
            rows = future.copy()
            rows["prediction"] = rounded(predict(labels, origin, config))
            value = score(rows.boardings.to_numpy(), rows.prediction.to_numpy())
            summaries.append({"model": config.name, "origin": str(origin), "score": value})
            for dimension in ("route", "month", "lead_week"):
                for key, part in rows.groupby(dimension):
                    actual = part.boardings.to_numpy()
                    forecast = part.prediction.to_numpy()
                    slices.append({
                        "model": config.name, "origin": str(origin), "dimension": dimension,
                        "key": int(key), "score": score(actual, forecast),
                        "actual_total": int(actual.sum()),
                        "absolute_error": int(np.abs(actual - forecast).sum()),
                    })
        print(f"completed {origin}: {len(grid)} candidates", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(slices).to_csv(output / "slices.csv", index=False)
    ranking = summary.groupby("model").score.agg(["min", "mean", "max"]).reset_index()
    ranking = ranking.sort_values(["min", "mean", "model"], ascending=[False, False, True])
    ranking.to_csv(output / "ranking.csv", index=False)
    tops = summary.sort_values(["origin", "score", "model"], ascending=[True, False, True])
    tops.groupby("origin").head(5).to_csv(output / "top_by_origin.csv", index=False)
    development = summary.loc[summary.origin != str(ORIGINS[-1])]
    dev_ranking = development.groupby("model").score.agg(["min", "mean", "max"]).reset_index()
    dev_ranking = dev_ranking.sort_values(["min", "mean", "model"],
                                        ascending=[False, False, True])
    dev_ranking.to_csv(output / "development_ranking.csv", index=False)
    monthly = dev_ranking.loc[dev_ranking.model.str.startswith("monthly_mean_")]
    if not monthly.empty:
        monthly_name = str(monthly.iloc[0].model)
        summary.loc[summary.model == monthly_name].to_csv(
            output / "monthly_baseline.csv", index=False)
    else:
        monthly_name = "baseline"
    selected = str(ranking.iloc[0].model)
    exports = []
    for name in dict.fromkeys((selected, "baseline", monthly_name, str(dev_ranking.iloc[0].model))):
        for origin in ORIGINS:
            rows = hour_grid(origin, origin + timedelta(days=60)).merge(
                labels, on=["route", "date", "hour"], validate="one_to_one")
            config = next(item for item in grid if item.name == name)
            rows["prediction"] = rounded(predict(labels, origin, config))
            rows["origin"] = str(origin)
            rows["model"] = name
            exports.append(rows)
    pd.concat(exports).to_csv(output / "selected_baseline_oof.csv.gz", index=False)
    chosen = next(item for item in grid if item.name == selected)
    submission = hour_grid(date(2025, 11, 1), date(2025, 12, 31))
    submission["prediction"] = rounded(predict(labels, date(2025, 11, 1), chosen))
    write_submission(submission, output / "submission.csv")
    result = {
        "selected": selected, "configs": len(grid), "evaluations": len(summaries),
        "development_selected": str(dev_ranking.iloc[0].model),
        "monthly_baseline": monthly_name,
        "minimum_score": float(ranking.iloc[0]["min"]),
        "mean_score": float(ranking.iloc[0]["mean"]),
        "stable_above_088": bool(ranking.iloc[0]["min"] > 0.88),
        "stable_above_093": bool(ranking.iloc[0]["min"] > 0.93),
        "plan_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "submission_rows": len(submission), "promotion": "experimental; no untouched holdout",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    run(args.archive, args.reconciliation, args.output, phase=args.phase)


if __name__ == "__main__":
    main()

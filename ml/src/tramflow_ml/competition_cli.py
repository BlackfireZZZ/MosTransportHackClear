"""Real-label evaluation and exact-grid submission for the Data Science track."""

import argparse
import hashlib
import json
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.competition import (
    FEATURE_VERSION,
    LABEL_ROUTES,
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
from tramflow_ml.competition_limits import assess_limits
from tramflow_ml.competition_models import ModelName, model_spec, predict
from tramflow_ml.competition_reconcile import reconcile_archive
from tramflow_ml.competition_shift import (
    LevelShiftPolicy,
    ShapeShiftPolicy,
    detect_hour_shape_shifts,
    detect_level_shifts,
)
from tramflow_ml.decision_guard import Evaluation, GuardReport, review_switch
from tramflow_ml.weather import load_weather


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def _ready_labels(archive: Path, end: date) -> pd.DataFrame:
    source = load_labels(archive)
    return complete_labels(
        source.loc[source.date <= end], date(2025, 1, 1), end, missing_as_zero=True
    )


def _verified_proof(archive: Path, path: Path) -> str:
    proof = json.loads(path.read_text(encoding="utf-8"))
    archive_sha256 = _digest(archive)
    if (
        proof.get("schema") != "route-hour-reconciliation.v1"
        or proof.get("archive_sha256") != archive_sha256
        or proof.get("period") != ["2025-01-01", "2025-10-31"]
        or proof.get("missing_keys_are_zero_in_supplied_raw") is not True
        or proof.get("raw_keys") != proof.get("label_keys")
    ):
        raise ValueError("reconciliation proof does not match this organizer archive")
    return archive_sha256


def _training_examples(model: ModelName, train: pd.DataFrame) -> int:
    if model in ("catboost_daily_bounded_50", "catboost_daily_recency_ensemble_50"):
        first = min(train.date)
        last = max(train.date)
        earliest = first + timedelta(days=56)
        latest = last - timedelta(days=60)
        origins = len(range(0, (latest - earliest).days + 1, 14)) if latest >= earliest else 0
        return origins * 61 * len(LABEL_ROUTES)
    if model == "fourier_ridge_daily":
        return len(train) // 24
    if model == "direct_extratrees_daily":
        days = len(train) // (len(LABEL_ROUTES) * 24)
        return len(range(28, days - 61 + 1, 3)) * len(LABEL_ROUTES)
    return len(train)


def _evaluate_one(
    labels: pd.DataFrame,
    origin: date,
    model: ModelName,
    direction_graph: pd.DataFrame | None,
    weather: pd.DataFrame | None = None,
) -> dict[str, object]:
    end = origin + timedelta(days=60)
    if end > max(labels.date):
        raise ValueError("61-day evaluation horizon exceeds available labels")
    past = labels.loc[labels.date < origin]
    if model in ("profile", "calendar_profile", "calendar_weekday_blend"):
        train = pd.DataFrame()
    elif model in (
        "calendar_robust_short", "calendar_robust_28", "calendar_robust_blend",
        "calendar_robust_hourshape", "calendar_robust_28_hourshape",
        "calendar_robust_routeshape", "calendar_robust_28_routeshape",
    ):
        train = past.copy()
    elif model in (
        "fourier_ridge_daily", "direct_extratrees_daily", "catboost_daily_bounded_50",
        "catboost_daily_recency_ensemble_50",
    ):
        train = past.copy()
    else:
        train = make_training_frame(
            past, origin - timedelta(days=1), direction_graph=direction_graph, weather=weather
        )
    features = feature_frame(labels, origin, direction_graph=direction_graph, weather=weather)
    predictions = rounded(predict(model, train, features))
    actual = labels.loc[
        (labels.date >= origin) & (labels.date <= end),
        ["route", "date", "hour", "boardings"],
    ]
    scored = features[["route", "date", "hour", "lead_day"]].merge(
        actual, on=["route", "date", "hour"], validate="one_to_one"
    )
    if len(scored) != len(features):
        raise ValueError("missing evaluation labels")
    scored["prediction"] = predictions
    scored["month"] = pd.to_datetime(scored.date).dt.month
    scored["week"] = ((scored.lead_day.astype(int) - 1) // 7) + 1
    return {
        "origin": origin.isoformat(),
        "end": end.isoformat(),
        "model": model,
        "model_spec": model_spec(model),
        "feature_version": (
            FEATURE_VERSION + "+moscow-route-zone-weather.v2" if weather is not None
            else FEATURE_VERSION
        ),
        "rows": len(scored),
        "training_rows": len(train),
        "training_examples": _training_examples(model, train),
        "history_rows": len(past),
        "training_cutoff": (origin - timedelta(days=1)).isoformat(),
        "weather_mode": "retrospective_actual_after_origin" if weather is not None else None,
        "training_origin_stride_days": (
            3 if model == "direct_extratrees_daily"
            else None if model in (
                "profile", "calendar_profile", "calendar_weekday_blend",
                "calendar_robust_short", "calendar_robust_28", "calendar_robust_blend",
                "calendar_robust_hourshape", "calendar_robust_28_hourshape",
                "calendar_robust_routeshape", "calendar_robust_28_routeshape",
                "fourier_ridge_daily"
            ) else 14
        ),
        "target": "successful_validation_boardings_route_date_hour",
        "timezone_assumption": "Europe/Moscow; raw CSV timezone unspecified",
        "graph_coverage": float((features.graph_available > 0).mean()),
        "score": score(scored.boardings.to_numpy(), predictions),
        "by_route": {
            str(key): score(part.boardings.to_numpy(), part.prediction.to_numpy())
            for key, part in scored.groupby("route")
        },
        "by_month": {
            str(key): score(part.boardings.to_numpy(), part.prediction.to_numpy())
            for key, part in scored.groupby("month")
        },
        "by_week": {
            str(key): score(part.boardings.to_numpy(), part.prediction.to_numpy())
            for key, part in scored.groupby("week")
        },
    }


def _review_model_switch(
    labels: pd.DataFrame, origin: date, incumbent: ModelName,
    candidates: tuple[ModelName, ...], window_days: int,
) -> GuardReport:
    if origin > SUBMISSION_START:
        raise ValueError("switch review cannot use labels after the organizer cutoff")
    if "catboost_weather_relative" in (incumbent, *candidates):
        raise ValueError("retrospective actual weather cannot enter switch review")

    @cache
    def evaluate(model: str, review_origin: date) -> Evaluation:
        result = _evaluate_one(labels, review_origin, cast(ModelName, model), None)
        raw_score = result["score"]
        if raw_score is None:
            raise ValueError("switch review requires a defined global score")
        return Evaluation(
            cast(float, raw_score), cast(dict[str, float | None], result["by_route"])
        )

    return review_switch(
        origin=origin,
        first_review_origin=date(2025, 5, 1),
        incumbent=incumbent,
        candidates=candidates,
        incumbent_window_days=window_days,
        detect=lambda day: (
            *detect_level_shifts(labels, day), *detect_hour_shape_shifts(labels, day)
        ),
        evaluate=evaluate,
    )


def _guard_payload(
    labels: pd.DataFrame, origin: date, incumbent: ModelName, report: GuardReport
) -> dict[str, object]:
    capability = model_spec(incumbent)
    limits = assess_limits(
        labels.loc[labels.date < origin], origin, 61, report.signals, capability
    )
    return {
        **report.to_dict(),
        "incumbent_spec": capability,
        "limit_flags": [flag.to_dict() for flag in limits],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="tramflow-competition")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path)
    parser.add_argument("--direction-graph", type=Path)
    parser.add_argument("--weather", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    reconciliation = sub.add_parser("reconcile")
    reconciliation.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--origin", type=date.fromisoformat, required=True)
    choices = (
        "profile", "calendar_profile", "calendar_weekday_blend", "calendar_robust_short",
        "calendar_robust_28", "calendar_robust_blend", "calendar_robust_hourshape",
        "calendar_robust_28_hourshape",
        "calendar_robust_routeshape", "calendar_robust_28_routeshape",
        "hgb", "catboost",
        "catboost_residual",
        "catboost_relative", "catboost_weather_relative", "catboost_daily_relative",
        "catboost_daily_bounded_50", "catboost_daily_recency_ensemble_50",
        "fourier_ridge_daily", "direct_extratrees_daily",
    )
    evaluation.add_argument("--model", choices=choices, required=True)
    submission = sub.add_parser("submit")
    submission.add_argument("--model", choices=choices, required=True)
    submission.add_argument("--output", type=Path, required=True)
    submission.add_argument("--guard-candidate", choices=choices, action="append", default=[])
    submission.add_argument("--incumbent-window-days", type=int)
    guard = sub.add_parser("guard")
    guard.add_argument("--origin", type=date.fromisoformat, default=SUBMISSION_START)
    guard.add_argument("--model", choices=choices, required=True)
    guard.add_argument("--candidate", choices=choices, action="append", required=True)
    guard.add_argument("--incumbent-window-days", type=int, required=True)
    args = parser.parse_args()
    if args.command == "reconcile":
        proof = reconcile_archive(args.archive)
        proof["archive_sha256"] = _digest(args.archive)
        args.output.write_text(
            json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(proof, ensure_ascii=False, sort_keys=True))
        return
    if args.reconciliation is None:
        parser.error("--reconciliation is required for evaluation and submission")
    if args.command == "submit" and (
        args.weather is not None or args.model == "catboost_weather_relative"
    ):
        parser.error("submit cannot use retrospective actual weather after the forecast cutoff")
    archive_sha256 = _verified_proof(args.archive, args.reconciliation)
    if args.command == "guard":
        if args.origin > SUBMISSION_START:
            parser.error("guard origin cannot exceed 2025-11-01")
        if args.direction_graph is not None or args.weather is not None:
            parser.error("guard currently requires label-only candidates")
        labels = _ready_labels(args.archive, args.origin - timedelta(days=1))
        guard_out = _review_model_switch(
            labels, args.origin, args.model, tuple(args.candidate), args.incumbent_window_days
        )
        print(json.dumps({**_guard_payload(labels, args.origin, args.model, guard_out),
                          "archive_sha256": archive_sha256,
                          "detectors": {
                              "demand_level": LevelShiftPolicy().__dict__,
                              "hour_shape": ShapeShiftPolicy().__dict__,
                          }},
                         ensure_ascii=False, indent=2, sort_keys=True))
        return
    direction_graph = pd.read_csv(args.direction_graph, sep=";") if args.direction_graph else None
    if args.model == "catboost_weather_relative" and args.weather is None:
        parser.error("--weather is required for catboost_weather_relative")
    weather = load_weather(args.weather) if args.weather else None
    if args.command == "evaluate":
        cutoff = args.origin + timedelta(days=60)
        labels = _ready_labels(args.archive, cutoff)
        report = _evaluate_one(labels, args.origin, args.model, direction_graph, weather)
        report["archive_sha256"] = archive_sha256
        report["graph_sha256"] = _digest(args.direction_graph) if args.direction_graph else None
        report["weather_sha256"] = _digest(args.weather) if args.weather else None
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    labels = _ready_labels(args.archive, date(2025, 10, 31))
    guard_payload = None
    if args.guard_candidate:
        if args.incumbent_window_days is None:
            parser.error("--incumbent-window-days is required with --guard-candidate")
        if direction_graph is not None or weather is not None:
            parser.error("guarded submission currently requires label-only candidates")
        guard_report = _review_model_switch(
            labels, SUBMISSION_START, args.model, tuple(args.guard_candidate),
            args.incumbent_window_days,
        )
        guard_payload = _guard_payload(labels, SUBMISSION_START, args.model, guard_report)
        args.model = guard_report.selected_model
    elif args.incumbent_window_days is not None:
        parser.error("--incumbent-window-days requires --guard-candidate")
    if args.model in ("profile", "calendar_profile", "calendar_weekday_blend"):
        train = pd.DataFrame()
    elif args.model in (
        "calendar_robust_short", "calendar_robust_28", "calendar_robust_blend",
        "calendar_robust_hourshape", "calendar_robust_28_hourshape",
        "calendar_robust_routeshape", "calendar_robust_28_routeshape",
    ):
        train = labels.copy()
    elif args.model in (
        "fourier_ridge_daily", "direct_extratrees_daily", "catboost_daily_bounded_50",
        "catboost_daily_recency_ensemble_50",
    ):
        train = labels.copy()
    else:
        train = make_training_frame(
            labels, date(2025, 10, 31), direction_graph=direction_graph, weather=weather
        )
    features = feature_frame(
        labels, SUBMISSION_START, direction_graph=direction_graph, weather=weather
    )
    output = features[["route", "date", "hour"]].copy()
    output["route"] = output.route.astype(int)
    output["hour"] = output.hour.astype(int)
    output["prediction"] = rounded(predict(args.model, train, features))
    if max(output.date) != SUBMISSION_END:
        raise ValueError("wrong submission horizon")
    write_submission(output, args.output)
    print(
        json.dumps(
            {
                "model": args.model,
                "guard": guard_payload,
                "model_spec": model_spec(args.model),
                "feature_version": (
                    FEATURE_VERSION + "+moscow-route-zone-weather.v2" if weather is not None
                    else FEATURE_VERSION
                ),
                "cutoff": "2025-10-31",
                "weather_mode": (
                    "retrospective_actual_after_2025-10-31" if weather is not None else None
                ),
                "rows": len(output),
                "training_rows": len(train),
                "training_examples": _training_examples(args.model, train),
                "history_rows": len(labels),
                "training_origin_stride_days": (
                    3 if args.model == "direct_extratrees_daily"
                    else None if args.model in (
                        "profile", "calendar_profile", "calendar_weekday_blend",
                        "calendar_robust_short", "calendar_robust_28", "calendar_robust_blend",
                        "calendar_robust_hourshape", "calendar_robust_28_hourshape",
                        "calendar_robust_routeshape", "calendar_robust_28_routeshape",
                        "fourier_ridge_daily"
                    ) else 14
                ),
                "target": "successful_validation_boardings_route_date_hour",
                "timezone_assumption": "Europe/Moscow; raw CSV timezone unspecified",
                "archive_sha256": archive_sha256,
                "graph_coverage": float((features.graph_available > 0).mean()),
                "graph_sha256": _digest(args.direction_graph) if args.direction_graph else None,
                "weather_sha256": _digest(args.weather) if args.weather else None,
                "submission_sha256": _digest(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Verify trained route output against serving schemas and organizer submission keys."""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml/src"))

from app.application.services.forecast_map import ForecastMapService  # noqa: E402
from app.domain.forecast import (  # noqa: E402
    ForecastHorizon,
    ForecastPoint,
    ForecastRunMetadata,
    ForecastSnapshot,
    ResolvedForecastSelection,
    RouteSummary,
)
from app.schemas.forecast import ForecastResponse  # noqa: E402
from tramflow_ml.route_models.artifact import MOSCOW, RouteForecast, aware  # noqa: E402
from tramflow_ml.route_models.data import ROUTES  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_views(artifact: RouteForecast) -> list[dict[str, object]]:
    serving_ids = {route: 1000 + i for i, route in enumerate(ROUTES)}
    results: list[dict[str, object]] = []
    for route in ROUTES:
        for horizon, start, days in [
            ("day", datetime(2025, 11, 1, tzinfo=MOSCOW), 1),
            ("month", datetime(2025, 11, 1, tzinfo=MOSCOW), 30),
            ("month", datetime(2025, 12, 1, tzinfo=MOSCOW), 31),
        ]:
            end = start + timedelta(days=days)
            projected = artifact.project(route=route, start=start, horizon=horizon)
            snapshot = ForecastSnapshot(
                route=RouteSummary(
                    serving_ids[route], str(route), f"Route {route}", "#d9342b"
                ),
                horizon=ForecastHorizon(horizon),
                generated_at=artifact.generated_at,
                model_version=artifact.model_version,
                points=[
                    ForecastPoint(p.timestamp, p.predicted, None, None, None)
                    for p in projected
                ],
                stops=[],
                stop_points=[],
                run=ForecastRunMetadata(
                    run_id=artifact.run_id,
                    dataset_id=artifact.dataset_id,
                    source_version=artifact.source_version,
                    feature_version=artifact.feature_version,
                    entity_version="explicit-probe-route-map.v1",
                    calendar_version="moscow-midnight.v1",
                    graph_version=None,
                    target=artifact.target,
                    unit=artifact.unit,
                    synthetic=artifact.synthetic,
                    forecast_origin=artifact.forecast_origin,
                    data_cutoff=artifact.data_cutoff,
                    interval_level=None,
                    interval_method=None,
                ),
                selection=ResolvedForecastSelection(
                    start, end, None, None, "route_bucket"
                ),
            )
            response = ForecastResponse.model_validate(
                ForecastMapService().enrich(snapshot)
            )
            restored = ForecastResponse.model_validate_json(response.model_dump_json())
            total = math.fsum(p.predicted_passengers for p in restored.points)
            expected = math.fsum(
                p.predicted
                for p in artifact.points
                if p.route == route and start <= aware(p.bucket_start) < end
            )
            require(restored == response, "JSON response changed values")
            require(
                math.isclose(total, expected, rel_tol=1e-12, abs_tol=1e-9),
                "projection total differs from source hours",
            )
            require(
                len(restored.points) == (24 if horizon == "day" else days),
                "bucket count",
            )
            require(
                restored.stops == [] and restored.stop_points == [], "fabricated stops"
            )
            require(
                restored.map is not None
                and restored.map.status == "unavailable"
                and restored.map.positions == [],
                "fabricated map",
            )
            require(
                restored.peak_load_percent is None
                and all(
                    p.capacity is None
                    and p.lower_bound is None
                    and p.upper_bound is None
                    for p in restored.points
                ),
                "fabricated capacity or uncertainty",
            )
            require(
                restored.run is not None
                and restored.run.run_id == artifact.run_id
                and restored.run.model_dump()["feature_version"]
                == artifact.feature_version
                and restored.model_version == artifact.model_version,
                "metadata changed",
            )
            results.append(
                dict(
                    route=route,
                    horizon=horizon,
                    start=start.isoformat(),
                    buckets=len(restored.points),
                    total=total,
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "ml/artifacts/route-ml-2025-cat-native",
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("verification.json")
    )
    args = parser.parse_args()
    forecast_path = args.artifact_dir / "forecast.json"
    forecast_bytes = forecast_path.read_bytes()
    artifact = RouteForecast.model_validate_json(forecast_bytes)
    views = verify_views(artifact)
    with zipfile.ZipFile(args.archive) as archive:
        template = list(
            csv.DictReader(
                io.StringIO(archive.read("test_submission.csv").decode("utf-8-sig")),
                delimiter=";",
            )
        )
    with (args.artifact_dir / "submission.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream, delimiter=";")
        require(
            reader.fieldnames == ["route", "date", "hour", "prediction"], "CSV header"
        )
        submission = list(reader)

    def key(row: dict[str, str]) -> tuple[int, str, int]:
        return int(row["route"]), row["date"], int(row["hour"])

    expected_keys = {key(row) for row in template}
    supplied_keys = {key(row) for row in submission}
    require(
        len(submission)
        == len(template)
        == len(expected_keys)
        == len(supplied_keys)
        == 14640,
        "submission/template cardinality or duplicate mismatch",
    )
    require(
        expected_keys == supplied_keys, "submission keys differ from organizer template"
    )
    expected_values = {
        (
            p.route,
            aware(p.bucket_start).date().isoformat(),
            aware(p.bucket_start).hour,
        ): round(p.predicted)
        for p in artifact.points
    }
    for row in submission:
        value = float(row["prediction"])
        require(
            math.isfinite(value) and value >= 0 and value == expected_values[key(row)],
            "submission differs from rounded artifact or contains invalid prediction",
        )
    report = dict(
        status="passed",
        run_id=artifact.run_id,
        model_version=artifact.model_version,
        forecast_sha256=hashlib.sha256(forecast_bytes).hexdigest(),
        checked_routes=len(ROUTES),
        serving_views=len(views),
        submission_rows=len(submission),
        template_keys_exact=True,
        rounded_predictions_exact=True,
        scope="offline trained-artifact/domain/HTTP serialization; no DB publication or UI run",
        route_identity="explicit synthetic verification mapping; production catalog not resolved",
        totals_tolerance=dict(relative=1e-12, absolute=1e-9),
        views=views,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "views"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()

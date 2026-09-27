import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from contracts.forecast_v1 import ForecastArtifact, validate_forecast_json


def payload(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = json.loads(
        (Path(__file__).parents[1] / "fixtures/forecast_v1.valid.json").read_text()
    )
    result.update(overrides)
    return result


def first_point() -> dict[str, object]:
    return cast(list[dict[str, object]], payload()["points"])[0]


def test_json_entrypoint_accepts_valid_moscow_bucket() -> None:
    artifact = validate_forecast_json(json.dumps(payload()))

    assert isinstance(artifact, ForecastArtifact)
    assert artifact.points[0].bucket_start.utcoffset() is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "forecast.v2"),
        ("data_cutoff", "2026-01-01T00:01:00Z"),
        ("generated_at", "2024-12-31T23:59:00Z"),
        ("interval_level", 1.0),
    ],
)
def test_temporal_and_version_boundaries_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(payload(**{field: value})))


def test_duplicate_point_and_partial_bounds_are_rejected() -> None:
    first = first_point()
    duplicate = {**first}
    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(payload(points=[first, duplicate])))
    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(payload(points=[{**first, "upper_bound": None}])))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_predictions_are_rejected(bad: float) -> None:
    point = {**first_point(), "predicted": bad}
    with pytest.raises((ValidationError, ValueError)):
        validate_forecast_json(json.dumps(payload(points=[point]), allow_nan=True))


def test_same_instant_in_utc_and_moscow_is_accepted() -> None:
    points = cast(list[dict[str, object]], payload()["points"])
    for point in points:
        for field in ("bucket_start", "bucket_end"):
            point[field] = datetime.fromisoformat(str(point[field])).astimezone(UTC).isoformat()
    artifact = validate_forecast_json(json.dumps(payload(points=points)))
    assert artifact.points[0].bucket_start.hour == 21


def test_horizon_cadence_and_target_unit_are_explicit() -> None:
    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(payload(bucket_granularity="daily")))
    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(payload(unit="passengers")))


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_forecast_json(
            json.dumps(
                payload(
                    points=[
                        {
                            **first_point(),
                            "bucket_start": "2026-01-01T01:00:00",
                        }
                    ]
                )
            )
        )

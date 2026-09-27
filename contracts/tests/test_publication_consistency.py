import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.forecast_v1 import DatasetManifest, validate_forecast_json

FIXTURE = Path(__file__).parents[1] / "fixtures/forecast_v1.valid.json"


def payload():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("mutation", ["length", "alignment", "outside", "missing", "override"])
def test_bad_calendar_and_per_point_metadata_are_rejected(mutation):
    data = payload()
    point = data["points"][0]
    if mutation == "length":
        point["bucket_end"] = "2030-01-01T00:00:00+03:00"
    elif mutation == "alignment":
        point["bucket_start"] = "2026-01-01T00:17:00+03:00"
    elif mutation == "outside":
        point["bucket_start"] = "2030-01-01T00:00:00+03:00"
        point["bucket_end"] = "2030-01-01T01:00:00+03:00"
    elif mutation == "missing":
        data["points"].pop()
    else:
        point["run_id"] = "another-run"
    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(data))


@pytest.mark.parametrize(
    "mutation", ["false_synthetic", "no_metadata", "partial_bounds", "no_bounds"]
)
def test_synthetic_and_uncertainty_are_consistent(mutation):
    data = payload()
    if mutation == "false_synthetic":
        data["synthetic"] = False
    elif mutation == "no_metadata":
        data["interval_level"] = data["interval_method"] = None
    elif mutation == "partial_bounds":
        data["points"][0]["lower_bound"] = data["points"][0]["upper_bound"] = None
    else:
        for point in data["points"]:
            point["lower_bound"] = point["upper_bound"] = None
    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(data))


def manifest_payload():
    data = payload()
    return {
        **{
            key: data[key]
            for key in (
                "schema_version",
                "dataset_id",
                "source_version",
                "feature_version",
                "target",
                "unit",
                "synthetic",
                "entity_version",
                "calendar_version",
            )
        },
        "source_hash": "a" * 64,
        "date_from": "2024-01-01T00:00:00+03:00",
        "date_to": "2026-01-01T00:00:00+03:00",
        "availability_policy": "event-and-availability.v1",
    }


def test_matching_manifest_and_uncertainty_unavailable_are_accepted():
    manifest = DatasetManifest.model_validate_json(json.dumps(manifest_payload()))
    data = payload()
    for point in data["points"]:
        point["lower_bound"] = point["upper_bound"] = None
    data["interval_level"] = data["interval_method"] = None
    assert (
        validate_forecast_json(json.dumps(data), manifest=manifest).dataset_id
        == manifest.dataset_id
    )


@pytest.mark.parametrize(
    "field", ["dataset_id", "source_version", "feature_version", "entity_version"]
)
def test_artifact_manifest_identity_must_match(field):
    raw = manifest_payload()
    raw[field] = "different.v1"
    manifest = DatasetManifest.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError, match="manifest"):
        validate_forecast_json(FIXTURE.read_text(), manifest=manifest)


@pytest.mark.parametrize("mutation", ["target", "synthetic", "unit", "naive", "range", "policy"])
def test_manifest_rejects_invalid_provenance_and_time(mutation):
    raw = manifest_payload()
    if mutation == "target":
        raw["target"] = "validation_count"
    elif mutation == "synthetic":
        raw["synthetic"] = False
    elif mutation == "unit":
        raw["unit"] = "vehicles"
    elif mutation == "naive":
        raw["date_from"] = "2024-01-01T00:00:00"
    elif mutation == "range":
        raw["date_to"] = raw["date_from"]
    else:
        raw["availability_policy"] = "unknown"
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate_json(json.dumps(raw))


def test_schema_snapshot_is_current():
    from contracts.forecast_v1 import ForecastArtifact

    stored = json.loads((FIXTURE.parents[1] / "forecast_v1.schema.json").read_text())
    assert stored == ForecastArtifact.model_json_schema()


@pytest.mark.parametrize(
    "target,unit", [("validation_count", "event_count"), ("onboard_load", "passengers")]
)
def test_valid_but_different_manifest_target_is_rejected(target, unit):
    raw = manifest_payload()
    raw.update(target=target, unit=unit, synthetic=False)
    manifest = DatasetManifest.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError, match="manifest"):
        validate_forecast_json(FIXTURE.read_text(), manifest=manifest)


def test_cutoff_outside_manifest_is_rejected():
    raw = manifest_payload()
    raw["date_to"] = "2025-01-01T00:00:00+03:00"
    manifest = DatasetManifest.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError, match="manifest"):
        validate_forecast_json(FIXTURE.read_text(), manifest=manifest)


@pytest.mark.parametrize(
    "year,month,days", [(2024, 2, 29), (2025, 2, 28), (2025, 4, 30), (2025, 12, 31)]
)
def test_complete_month_artifacts_follow_real_calendar(year, month, days):
    from datetime import datetime, timedelta, timezone

    data = payload()
    origin = datetime(year, month, 1, tzinfo=timezone(timedelta(hours=3)))
    data.update(
        horizon="month",
        bucket_granularity="daily",
        forecast_origin=origin.isoformat(),
        generated_at=origin.isoformat(),
        data_cutoff=origin.isoformat(),
    )
    point = data["points"][0]
    data["points"] = [
        {
            **point,
            "bucket_start": (origin + timedelta(days=i)).isoformat(),
            "bucket_end": (origin + timedelta(days=i + 1)).isoformat(),
        }
        for i in range(days)
    ]
    assert len(validate_forecast_json(json.dumps(data)).points) == days
    data["points"].pop()
    with pytest.raises(ValidationError, match="complete horizon"):
        validate_forecast_json(json.dumps(data))


def test_complete_year_artifact_and_subset_entities():
    data = payload()
    origin = "2024-01-01T00:00:00+03:00"
    data.update(
        horizon="year",
        bucket_granularity="monthly",
        forecast_origin=origin,
        generated_at=origin,
        data_cutoff=origin,
    )
    boundaries = [f"2024-{month:02}-01T00:00:00+03:00" for month in range(1, 13)]
    boundaries.append("2025-01-01T00:00:00+03:00")
    point = data["points"][0]
    data["points"] = [
        {**point, "bucket_start": start, "bucket_end": end}
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    assert len(validate_forecast_json(json.dumps(data)).points) == 12
    data["points"].append({**data["points"][0], "stop_id": "partially-covered-stop"})
    with pytest.raises(ValidationError, match="complete horizon"):
        validate_forecast_json(json.dumps(data))

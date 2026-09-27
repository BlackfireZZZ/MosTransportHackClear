from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_planning_service
from app.application.services.planning import PlanningService
from app.infrastructure.planning import FilePlanningRepository
from app.main import app
from app.schemas.planning import PlanningRequest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "1e79f596a6723de6b80dcdbe3915fa66561e0466d6299da8a179c760e7154bed"


@pytest.fixture(scope="module")
def planner():
    return PlanningService(FilePlanningRepository(ROOT / "data/planning"))


def request(**kwargs):
    return PlanningRequest.model_validate(
        {"route": "1", "start_date": "2025-11-01", "forecast_mode": "stop_model", **kwargs}
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    "day,expected,unknown",
    [
        ("2025-11-01", 125.58041958041959, 287.84615384615387),
        ("2025-11-08", 33.246428571428574, 69.06309523809523),
        ("2025-12-31", 31.390243902439025, 8.21951219512195),
    ],
)
def test_raw_source_cells_exact_without_weekday_averaging(planner, day, expected, unknown):
    result = planner.forecast(request(start_date=day))
    point = result["points"][8]
    cell = next(
        row for row in point["spatial"] if row["stop_id"] == "gtfs:2594" and row["direction"] == "0"
    )
    assert cell["baseline"] == expected
    assert point["unallocated_baseline"] == unknown
    assert result["provenance"]["stop_predictions_sha256"] == SOURCE_SHA
    assert (
        result["model_version"] == "direct-stop-calendar-history.anchored-v3:calendar_weekday_blend"
    )
    assert "csv_sha256" not in result["provenance"]
    assert result["provenance"]["mapping_method"] == (
        "duty-calendar-clock.first-stop-local-anchors-transfer.v3"
    )
    assert any("перенос расписания" in warning for warning in result["warnings"])


def test_stop_model_filters_weighted_scenario_and_unknown(planner):
    base = planner.forecast(request())
    weights = {"weather": {"enabled": True, "multiplier": 1.2}}
    whole_scenario = planner.forecast(request(factors=weights))
    result = planner.forecast(
        request(
            stop_id="gtfs:2594",
            direction="0",
            factors=weights,
        )
    )
    for point, unfiltered, weighted in zip(
        result["points"], base["points"], whole_scenario["points"], strict=True
    ):
        assert point["scenario"] == pytest.approx(
            weighted["route_scenario"] * point["baseline"] / unfiltered["route_baseline"]
        )
        assert len(point["spatial"]) == 1
        assert point["unallocated_baseline"] == 0
        assert point["route_unallocated_baseline"] == unfiltered["unallocated_baseline"]
        assert point["route_baseline"] == unfiltered["baseline"]


@pytest.mark.parametrize("horizon,count", [("day", 24), ("month", 30), ("year", 12)])
def test_exact_model_mass_and_horizons(planner, horizon, count):
    result = planner.forecast(request(horizon=horizon))
    assert len(result["points"]) == count
    for point in result["points"]:
        assert (
            point["baseline"]
            == sum(row["baseline"] for row in point["spatial"]) + point["unallocated_baseline"]
        )
    assert result["qualitative"] == (horizon == "year")
    if horizon == "year":
        assert result["points"][-1]["bucket_end"] == "2026-11-01T00:00:00+03:00"


def test_month_is_sum_of_exact_hours_not_approved_route(planner):
    month = planner.forecast(request(horizon="month"))
    day = planner.forecast(request())
    assert month["points"][0]["baseline"] == pytest.approx(
        sum(p["baseline"] for p in day["points"])
    )
    source = planner.repository.stop_model()
    route_rows = [r for r in source["identities"] if r["route"] == "1"]
    assert sum(p["baseline"] for p in month["points"]) == pytest.approx(
        sum(sum(r["values"][: 30 * 24]) for r in route_rows)
    )
    approved = planner.forecast(request(forecast_mode="approved"))
    assert day["points"][8]["baseline"] != approved["points"][8]["baseline"]


def test_learned_source_switches_never_change_direct_model(planner):
    result = planner.forecast(
        request(source_enabled={"weather": True, "calendar": True, "events": True, "traffic": True})
    )
    assert result["points"] == planner.forecast(request())["points"]
    assert any("таких вариантов обучения нет" in warning for warning in result["warnings"])
    assert {s["status"] for s in result["sources"]} == {"stop_model_fixed", "manual_only"}


def test_corrupt_direct_asset_returns_503(tmp_path):
    import shutil

    shutil.copytree(ROOT / "data/planning", tmp_path / "planning")
    (tmp_path / "planning/stop-model.json.gz").write_bytes(b"broken")
    app.dependency_overrides[get_planning_service] = lambda: PlanningService(
        FilePlanningRepository(tmp_path / "planning")
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/planning/forecast", json=request())
            assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_http_exact_raw_model_cell(planner):
    app.dependency_overrides[get_planning_service] = lambda: planner
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/planning/forecast", json=request(stop_id="gtfs:2594", direction="0")
            )
            assert response.status_code == 200
            assert response.json()["points"][8]["baseline"] == 125.58041958041959
    finally:
        app.dependency_overrides.clear()

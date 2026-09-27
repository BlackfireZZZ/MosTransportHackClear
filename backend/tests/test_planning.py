import csv
import gzip
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_planning_service
from app.application.services.planning import PlanningService, PlanningUnavailable, month_after
from app.infrastructure.planning import FilePlanningRepository
from app.main import app
from app.schemas.planning import PlanningRequest, PlanningResponse

ROOT = Path(__file__).resolve().parents[2]


def service():
    return PlanningService(FilePlanningRepository(ROOT / "data/planning"))


def request(**kwargs):
    return PlanningRequest(**({"route": "1", "start_date": date(2025, 11, 1)} | kwargs)).model_dump(
        mode="json"
    )


@pytest.mark.parametrize("horizon,count", [("day", 24), ("month", 30), ("year", 12)])
@pytest.mark.parametrize("start_date", [date(2025, 11, 1), date(2025, 12, 1)])
def test_real_artifact_conservation(horizon, count, start_date):
    if horizon == "month" and start_date.month == 12:
        count = 31
    result = service().forecast(request(horizon=horizon, start_date=start_date))
    PlanningResponse.model_validate(result)
    assert len(result["points"]) == count
    assert result["stops"]
    assert all(point["timestamp"].endswith("+03:00") for point in result["points"])
    for point in result["points"]:
        assert all(
            point[key] >= 0
            for key in (
                "baseline", "scenario", "route_baseline", "route_scenario",
                "unallocated_baseline", "route_unallocated_baseline",
            )
        )
        assert point["baseline"] == pytest.approx(
            sum(s["baseline"] for s in point["spatial"]) + point["unallocated_baseline"]
        )
        assert point["route_baseline"] >= point["baseline"]
        assert point["route_baseline"] == pytest.approx(
            sum(s["baseline"] for s in point["spatial"])
            + point["route_unallocated_baseline"]
        )
        assert point["scenario"] == pytest.approx(point["baseline"])
    if horizon == "year":
        expected = []
        month = start_date
        for _ in range(12):
            expected.append(f"{month.isoformat()}T00:00:00+03:00")
            month = month_after(month)
        assert [p["timestamp"] for p in result["points"]] == expected
        assert result["points"][-1]["bucket_end"] == f"{month.isoformat()}T00:00:00+03:00"
        contest_months = 2 if start_date.month == 11 else 1
        assert [p["basis"] for p in result["points"]] == (
            ["competition_period"] * contest_months
            + ["scenario_projection"] * (12 - contest_months)
        )
        assert result["qualitative"]


def test_normalized_weights_and_filter_do_not_mutate_baseline():
    base = service().forecast(request())
    stop = max(base["points"][8]["spatial"], key=lambda s: s["baseline"])
    selected = service().forecast(
        request(
            stop_id=stop["stop_id"],
            direction=stop["direction"],
            factors={
                "weather": {"enabled": True, "multiplier": 2},
                "traffic": {"enabled": False, "multiplier": 3},
            },
        )
    )
    assert selected["points"][8]["baseline"] == pytest.approx(stop["baseline"])
    assert selected["points"][8]["scenario"] == pytest.approx(
        selected["points"][8]["route_scenario"]
        * stop["baseline"] / base["points"][8]["route_baseline"]
    )
    assert json.loads(selected["provenance"]["normalized_weights"]) == {
        "base": pytest.approx(1 / 3), "weather": pytest.approx(2 / 3)
    }
    assert selected["points"][8]["route_scenario"] != pytest.approx(
        base["points"][8]["route_baseline"] * 2
    )
    assert selected["points"][8]["unallocated_baseline"] == 0
    assert selected["stops"] == base["stops"]
    assert service().forecast(request())["points"] == base["points"]


@pytest.mark.parametrize("horizon", ["day", "month", "year"])
def test_weighted_predictions_conserve_route_and_stops(horizon):
    result = service().forecast(request(horizon=horizon, factors={
        "weather": {"enabled": True, "multiplier": 2},
        "events": {"enabled": True, "multiplier": 1},
    }))
    assert json.loads(result["provenance"]["normalized_weights"]) == {
        "base": 0.25, "weather": 0.5, "events": 0.25,
    }
    for point in result["points"]:
        assert point["route_scenario"] == pytest.approx(
            sum(stop["scenario"] for stop in point["spatial"])
            + point["route_unallocated_scenario"]
        )
        assert point["scenario"] == pytest.approx(point["route_scenario"])
        assert point["route_scenario"] >= 0
    assert result["run_id"] != service().forecast(request())["run_id"]


def test_approved_scenario_uses_normalized_source_predictions():
    provider = FilePlanningRepository(ROOT / "data/planning")
    baseline = PlanningService(provider).forecast(request())
    weighted = PlanningService(provider).forecast(request(factors={
        "weather": {"enabled": True, "multiplier": 2},
        "events": {"enabled": True, "multiplier": 1},
    }))
    variants = provider.approved_sources()
    index = next(
        i for i, (route, day, hour) in enumerate(variants["keys"])
        if str(route) == "1" and day == "2025-11-01" and hour == 8
    )
    expected = (
        baseline["points"][8]["route_baseline"]
        + 2 * variants["branches"]["weather"][index]
        + variants["branches"]["events"][index]
    ) / 4
    assert weighted["points"][8]["route_scenario"] == pytest.approx(expected)
    assert weighted["points"][8]["route_scenario"] == pytest.approx(
        weighted["points"][8]["scenario"]
    )
    assert weighted["factors"]["weather"]["multiplier"] == 2
    assert weighted["provenance"]["scenario_source_sha256"] == variants["artifact_sha256"]


def test_approved_checkbox_and_weight_do_not_mix_branch_twice():
    planner = service()
    weighted = planner.forecast(request(
        source_enabled={"weather": True},
        factors={"weather": {"enabled": True, "multiplier": 2}},
    ))
    factor_only = planner.forecast(request(
        factors={"weather": {"enabled": True, "multiplier": 2}},
    ))
    assert [point["route_scenario"] for point in weighted["points"]] == pytest.approx(
        [point["route_scenario"] for point in factor_only["points"]]
    )
    assert weighted["points"][8]["baseline"] == pytest.approx(
        planner.forecast(request())["points"][8]["baseline"]
    )


def test_positive_weight_requires_versioned_source_artifact():
    class Repository:
        def load(self):
            return FilePlanningRepository(ROOT / "data/planning").load()

        def approved_sources(self):
            raise PlanningUnavailable("Approved source artifact unavailable or invalid")

    planner = PlanningService(Repository())
    assert planner.forecast(request())["points"]
    with pytest.raises(PlanningUnavailable, match="Approved source artifact unavailable"):
        planner.forecast(request(factors={
            "weather": {"enabled": True, "multiplier": 2},
        }))


def test_route_totals_equal_selected_csv():
    provider = FilePlanningRepository(ROOT / "data/planning")
    result = PlanningService(provider).forecast(request())
    rows = [
        row
        for row in provider.load()["route_rows"]
        if row["route"] == "1" and row["date"] == "2025-11-01"
    ]
    assert sum(p["baseline"] for p in result["points"]) == pytest.approx(
        sum(float(row["prediction"]) for row in rows)
    )


def test_approved_source_switches_use_bound_branch_artifact():
    base = service().forecast(request())
    variants = {}
    for name in ("calendar", "weather", "traffic", "events"):
        result = service().forecast(request(source_enabled={name: True}))
        PlanningResponse.model_validate(result)
        assert result["model_version"] == "approved-source-variants.v1:unsubmitted"
        assert result["provenance"]["approved_reference_model_version"] == base["model_version"]
        assert result["provenance"]["approved_reference_csv_sha256"] == (
            base["provenance"]["csv_sha256"]
        )
        assert "csv_sha256" not in result["provenance"]
        assert result["provenance"]["model_version"] == result["model_version"]
        assert result["provenance"]["generated_at"] == result["generated_at"]
        assert result["provenance"]["source_variant_sha256"]
        assert result["sources"][[s["id"] for s in result["sources"]].index(name)][
            "status"
        ] == "approved_source_enabled"
        if name == "weather":
            assert next(s for s in result["sources"] if s["id"] == "weather")["url"] == (
                "https://open-meteo.com/en/docs/historical-forecast-api"
            )
        variants[name] = [point["baseline"] for point in result["points"]]
    baseline = [point["baseline"] for point in base["points"]]
    assert all(values != baseline for values in variants.values())
    assert len({tuple(values) for values in variants.values()}) == 4
    enabled = service().forecast(request(
        start_date=date(2025, 11, 2),
        source_enabled={name: True for name in variants},
    ))
    assert enabled["provenance"]["source_mix_weights"] == (
        '{"calendar": 0.05, "events": 0.05, "incumbent": 0.8, '
        '"traffic": 0.05, "weather": 0.05}'
    )
    reference_day = service().forecast(request(start_date=date(2025, 11, 2)))
    assert [point["baseline"] for point in enabled["points"]] != [
        point["baseline"] for point in reference_day["points"]
    ]
    assert service().forecast(request())["points"] == base["points"]
    experiment = service().forecast(request(
        forecast_mode="external_experiment", source_enabled={"weather": True},
    ))
    assert next(s for s in experiment["sources"] if s["id"] == "weather")["url"] == (
        "https://open-meteo.com/en/docs/historical-weather-api"
    )


def test_corrupt_approved_source_bundle_returns_503(tmp_path):
    for name in ("forecast.json.gz", "manifest.json", "approved-source-variants.json.gz",
                 "approved-source-variants-manifest.json"):
        (tmp_path / name).write_bytes((ROOT / "data/planning" / name).read_bytes())
    (tmp_path / "approved-source-variants.json.gz").write_bytes(b"corrupt")
    app.dependency_overrides[get_planning_service] = lambda: PlanningService(
        FilePlanningRepository(tmp_path)
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/planning/forecast",
                json=request(source_enabled={"weather": True}),
            )
            assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_approved_all_source_artifact_matches_candidate_csv():
    repository = FilePlanningRepository(ROOT / "data/planning")
    variants = repository.approved_sources()
    base = {
        (int(row["route"]), row["date"], int(row["hour"])): int(row["prediction"])
        for row in repository.load()["route_rows"]
    }
    path = ROOT / (
        "ml/competition_submissions/2026-09-27/approved-external-candidate/"
        "approved-sources-all.csv"
    )
    with path.open() as stream:
        expected = list(csv.DictReader(stream, delimiter=";"))
    assert len(expected) == len(variants["keys"]) == 14640
    for index, (number, day, hour) in enumerate(variants["keys"]):
        row = expected[index]
        assert (number, day, hour) == (int(row["route"]), row["date"], int(row["hour"]))
        actual = round(
            (1 - sum(variants["weights"].values())) * base[(number, day, hour)]
            + sum(variants["weights"][name] * values[index]
                  for name, values in variants["branches"].items())
        )
        assert actual == int(row["prediction"])


def test_calendar_month_boundaries():
    assert month_after(date(2028, 2, 1)) == date(2028, 3, 1)
    assert month_after(date(2025, 12, 1)) == date(2026, 1, 1)
    payload = request(horizon="month")
    payload["start_date"] = "2025-12-31"
    result = service().forecast(payload)
    assert len(result["points"]) == 1
    assert result["points"][0]["bucket_end"] == "2026-01-01T00:00:00+03:00"


@pytest.mark.parametrize(
    "change",
    [
        {"start_date": "2025-10-31"},
        {"route": "999"},
        {"horizon": "year", "start_date": "2025-11-02"},
        {"stop_id": "gtfs:nope"},
        {"direction": "99"},
    ],
)
def test_invalid_selection(change):
    with pytest.raises(ValueError):
        service().forecast({**request(), **change})


def test_artifact_hash_rejection(tmp_path):
    (tmp_path / "forecast.json.gz").write_bytes(b"bad")
    (tmp_path / "manifest.json").write_text('{"sha256":"not-the-hash"}')
    with pytest.raises(PlanningUnavailable, match="invalid"):
        FilePlanningRepository(tmp_path).load()


def test_asgi_real_artifact_and_bad_factors():
    app.dependency_overrides[get_planning_service] = service
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/planning/forecast", json=request())
            assert response.status_code == 200
            assert len(response.json()["points"]) == 24
            for multiplier in [-0.1, 3.1]:
                response = client.post(
                    "/api/v1/planning/forecast",
                    json=request()
                    | {"factors": {"events": {"enabled": True, "multiplier": multiplier}}},
                )
                assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_stop_direction_mismatch_rejected():
    stops = service().forecast(request())["stops"]
    keys = {(s["stop_id"], s["direction"]) for s in stops}
    invalid = next(
        (s["stop_id"], d) for s in stops for d in ("0", "1") if (s["stop_id"], d) not in keys
    )
    with pytest.raises(ValueError, match="selected direction"):
        service().forecast(request(stop_id=invalid[0], direction=invalid[1]))


def test_filtered_unknown_mass_stays_explicit():
    base = service().forecast(request())
    selected = service().forecast(request(direction="0"))
    for a, b in zip(base["points"], selected["points"], strict=True):
        assert b["route_baseline"] == a["baseline"]
        assert b["route_unallocated_baseline"] == a["unallocated_baseline"]
        assert b["baseline"] <= b["route_baseline"]


def test_experimental_mix_rounded_before_daily_sum():
    class Repository:
        def load(self):
            return FilePlanningRepository(ROOT / "data/planning").load()

        def experimental(self):
            rows = self.load()["route_rows"]
            return {
                "keys": [[int(r["route"]), r["date"], int(r["hour"])] for r in rows],
                "branches": {
                    name: [value] * len(rows)
                    for name, value in [
                        ("base", 10.0),
                        ("calendar", 11.0),
                        ("weather", 19.0),
                        ("traffic", 30.0),
                        ("news", 40.0),
                    ]
                },
                "generated_at": "2026-09-27T00:00:00+00:00",
                "cutoff": "2025-10-31",
                "validation": {"test": "fixture"},
                "artifact_sha256": "fixture-hash",
            }

    planner = PlanningService(Repository())
    zero = planner.forecast(request(forecast_mode="external_experiment"))
    calendar = planner.forecast(
        request(forecast_mode="external_experiment", source_enabled={"calendar": True})
    )
    weather = planner.forecast(
        request(forecast_mode="external_experiment", source_enabled={"weather": True})
    )
    news = planner.forecast(
        request(forecast_mode="external_experiment", source_enabled={"events": True})
    )
    assert zero["points"][0]["baseline"] == pytest.approx(10)
    assert calendar["points"][0]["baseline"] == pytest.approx(10)
    assert weather["points"][0]["baseline"] == pytest.approx(14)
    assert news["points"][0]["baseline"] == pytest.approx(25)
    assert zero["run_id"] != calendar["run_id"]
    assert news["provenance"]["experimental_branches"] == "base,news"


def test_artifact_corruption_is_unavailable_not_bad_request(tmp_path):
    (tmp_path / "forecast.json.gz").write_bytes(b"broken")
    (tmp_path / "manifest.json").write_text('{"sha256":"bad"}')
    app.dependency_overrides[get_planning_service] = lambda: PlanningService(
        FilePlanningRepository(tmp_path)
    )
    try:
        with TestClient(app) as client:
            assert client.post("/api/v1/planning/forecast", json=request()).status_code == 503
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("kind", ["short_cell", "truncated_gzip"])
def test_structurally_invalid_artifact_is_unavailable(tmp_path, kind):
    raw = (ROOT / "data/planning/forecast.json.gz").read_bytes()
    if kind == "short_cell":
        content = json.loads(gzip.decompress(raw))
        content["shares"][next(iter(content["shares"]))] = [["short-cell"]]
        raw = gzip.compress(json.dumps(content).encode())
    else:
        raw = raw[: len(raw) // 2]
    (tmp_path / "forecast.json.gz").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": hashlib.sha256(raw).hexdigest()}))
    with pytest.raises(PlanningUnavailable):
        FilePlanningRepository(tmp_path).load()

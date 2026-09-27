import importlib.util
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location("benchmark_forecast", Path(__file__).with_name("benchmark_forecast.py"))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def response(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body)


def test_fixture_covers_required_modes_and_dry_run_has_no_metrics() -> None:
    data = benchmark.fixture()
    assert len(data["cases"]) == 11
    assert {case["horizon"] for case in data["cases"] if case.get("mode") == "stop_model"} == {"day", "month", "year"}


def test_rejects_empty_published_row_and_missing_provenance() -> None:
    case = next(case for case in benchmark.fixture()["cases"] if case["id"] == "get_day")
    with pytest.raises(ValueError, match="missing points"):
        benchmark.check_response(case, response({"horizon": "day", "points": [], "model_version": "x"}))
    with pytest.raises(ValueError, match="provenance"):
        benchmark.check_response(case, response({"horizon": "day", "points": [{}], "model_version": "x"}))


def test_rejects_wrong_year_semantics_and_error_status() -> None:
    year = next(case for case in benchmark.fixture()["cases"] if case["id"] == "post_stop_year")
    body = {"horizon": "year", "points": [{}] * 12, "model_version": "x", "forecast_mode": "stop_model", "timezone": "Europe/Moscow", "qualitative": False}
    with pytest.raises(ValueError, match="qualitative"):
        benchmark.check_response(year, response(body))
    invalid = next(case for case in benchmark.fixture()["cases"] if case["id"] == "post_invalid_date")
    with pytest.raises(ValueError, match="wrong error"):
        benchmark.check_response(invalid, response({"detail": "bad"}, 200))


def test_public_result_requires_complete_successful_fixed_cases() -> None:
    ids = {case["id"] for case in benchmark.fixture()["cases"]}
    template = {"required_cases": [{"id": case_id, "label": case_id} for case_id in ids]}
    runs = [{"case_id": case_id, "errors": 0, "conditions": {"artifact_sha256": benchmark.digest(benchmark.CASES)}} for case_id in ids]
    raw = {"schema_version": 1, "status": "measured", "updated_at": "2026-09-27T00:00:00Z", "runs": runs}
    assert benchmark.public_result(raw, template, ids, "sha")["source_sha256"] == "sha"
    with pytest.raises(ValueError, match="exactly"):
        benchmark.public_result({**raw, "runs": runs[:-1]}, template, ids, "sha")
    with pytest.raises(ValueError, match="completed"):
        benchmark.public_result({**raw, "status": "failed"}, template, ids, "sha")

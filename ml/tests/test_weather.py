import hashlib
import json
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from tramflow_ml import weather, weather_spatial
from tramflow_ml.competition import LABEL_ROUTES, feature_frame
from tramflow_ml.weather_spatial import package_spatial, scored_stops


def _response(start: date, end: date) -> dict[str, object]:
    times = [
        (datetime.combine(start, datetime.min.time()) + timedelta(hours=offset)).strftime(
            "%Y-%m-%dT%H:%M"
        )
        for offset in range(((end - start).days + 1) * 24)
    ]
    return {
        "timezone": "Europe/Moscow", "utc_offset_seconds": 10800,
        "hourly": {"time": times, **{
            name: list(range(len(times))) for name in weather.VARIABLES
        }},
    }


def test_rejects_missing_hour():
    result = _response(date(2025, 1, 1), date(2025, 1, 1))
    result["hourly"]["time"].pop()
    with pytest.raises(ValueError, match="grid"):
        weather._parse_chunk(result, date(2025, 1, 1), date(2025, 1, 1))


def test_fetch_year_shifts_hourly_accumulations_and_checks_digest(monkeypatch, tmp_path):
    monkeypatch.setattr(weather, "_request", _response)
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)
    output = tmp_path / "weather.csv"
    manifest = weather.fetch_year(2025, output)
    rows = weather.load_weather(output)
    assert len(rows) == 8760
    assert manifest["requests"] == 27
    assert rows.iloc[0].temperature_2m == 0
    assert rows.iloc[0].precipitation == 1
    assert rows.iloc[-1].date == date(2025, 12, 31)
    assert rows.iloc[-1].hour == 23
    assert rows.iloc[-1].precipitation == 24
    assert manifest["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        weather.fetch_year(2025, output)
    content = json.loads(output.with_suffix(".manifest.json").read_text())
    content["csv_sha256"] = "bad"
    output.with_suffix(".manifest.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="checksum"):
        weather.load_weather(output)


def test_load_rejects_duplicate_hour(tmp_path):
    output = tmp_path / "weather.csv"
    frame = pd.DataFrame({"date": ["2025-01-01"] * 2, "hour": [0, 0], **{
        name: [1, 1] for name in weather.VARIABLES
    }})
    frame.to_csv(output, sep=";", index=False)
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "schema": weather.SCHEMA, "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": 2,
    }))
    with pytest.raises(ValueError, match="duplicate"):
        weather.load_weather(output)


def test_scored_stops_include_all_competition_routes():
    from pathlib import Path

    stops = scored_stops(Path("data/tram_stops.csv"))
    assert len(stops) == 445
    assert len(stops) == len({row["osm_id"] for row in stops})
    assert {str(route) for route in LABEL_ROUTES} <= {
        route for stop in stops for route in stop["routes"].split(";")
    }


def test_stops_use_their_own_returned_api_cell(monkeypatch, tmp_path):
    stops = [
        {"osm_id": "1", "name": "west", "lat": "55.0", "lon": "37.01",
         "routes": "1;7"},
        {"osm_id": "2", "name": "east", "lat": "55.0", "lon": "37.19",
         "routes": "7"},
        {"osm_id": "3", "name": "east again", "lat": "55.0", "lon": "37.20",
         "routes": "7"},
    ]
    def fetch(start, end, requested):
        result = []
        for row in requested:
            payload = _response(start, end)
            payload.update({"latitude": 55.0, "longitude": (
                37.0 if float(row["longitude"]) < 37.1 else 37.2
            )})
            result.append(payload)
        return result
    monkeypatch.setattr(weather_spatial, "_fetch_chunk", fetch)
    monkeypatch.setattr(weather_spatial.time, "sleep", lambda _: None)
    zones, mapped, weights, coordinates, requests, cached = (
        weather_spatial.discover_stop_cells(stops, tmp_path)
    )
    assert len(zones) == 2
    assert [stop["zone_id"] for stop in mapped] == ["G001", "G002", "G002"]
    assert coordinates == {"G001": [55.0, 37.0], "G002": [55.0, 37.2]}
    assert (requests, cached) == (1, 0)
    assert weights == [
        {"route": 1, "zone_id": "G001", "stop_count": 1},
        {"route": 7, "zone_id": "G001", "stop_count": 1},
        {"route": 7, "zone_id": "G002", "stop_count": 2},
    ]


def test_route_weather_features_join_by_route_and_hour():
    start = date(2025, 11, 1)
    grid = pd.MultiIndex.from_product(
        [LABEL_ROUTES, pd.date_range(start, periods=61).date, range(24)],
        names=["route", "date", "hour"],
    ).to_frame(index=False)
    for name in weather.VARIABLES:
        grid[name] = grid.route.astype(float)
    historical = pd.DataFrame([
        (route, date(2025, 10, 1), hour, 1)
        for route in LABEL_ROUTES for hour in range(24)
    ], columns=["route", "date", "hour", "boardings"])
    features = feature_frame(historical, start, weather=grid)
    assert features.loc[features.route == 1, "weather_temperature_2m"].eq(1).all()
    assert features.loc[features.route == 50, "weather_temperature_2m"].eq(50).all()
    with pytest.raises(ValueError, match="missing values"):
        feature_frame(historical, start, weather=grid.iloc[:-1])


def test_packaged_weather_is_hash_checked_and_loadable(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    dates = [date(2025, 1, 1)] * 24
    frame = pd.DataFrame({
        "route": [route for route in LABEL_ROUTES for _ in range(24)],
        "date": dates * len(LABEL_ROUTES),
        "hour": list(range(24)) * len(LABEL_ROUTES),
        **{name: [1] * (24 * len(LABEL_ROUTES)) for name in weather.VARIABLES},
    })
    frame.to_csv(raw / "route-weather.csv", sep=";", index=False)
    for name in ("zones.csv", "stops.csv", "route-zones.csv", "zone-weather.csv"):
        (raw / name).write_text("test\n")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in raw.iterdir()}
    (raw / "manifest.json").write_text(json.dumps({
        "schema": weather.SPATIAL_SCHEMA, "sha256": hashes,
        "route_rows": len(frame), "date_range": ["2025-01-01", "2025-01-01"],
    }))
    first = package_spatial(raw, tmp_path / "first")
    second = package_spatial(raw, tmp_path / "second")
    assert first["sha256"] == second["sha256"]
    loaded = weather.load_weather(tmp_path / "first" / "route-weather.csv.gz")
    assert len(loaded) == len(frame)

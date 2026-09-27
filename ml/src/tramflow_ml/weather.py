"""Open-Meteo hourly Moscow weather aligned to route-hour boarding buckets."""

import argparse
import csv
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 55.7558
LONGITUDE = 37.6173
ZONE = "Europe/Moscow"
VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "precipitation", "rain",
    "snowfall", "snow_depth", "weather_code", "cloud_cover",
    "wind_speed_10m", "wind_gusts_10m",
)
ACCUMULATIONS = ("precipitation", "rain", "snowfall")
SCHEMA = "moscow-hourly-weather.v1"
SPATIAL_SCHEMA = "moscow-route-zone-weather.v2"


def _request(start: date, end: date) -> dict[str, object]:
    parameters = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": ZONE, "models": "best_match", "hourly": ",".join(VARIABLES),
    }
    request = urllib.request.Request(
        URL + "?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": "tramflow-weather-research/1.0"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(2**attempt)
        except (
            TimeoutError, urllib.error.URLError, http.client.RemoteDisconnected,
            ConnectionResetError,
        ):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable retry state")


def _parse_chunk(payload: dict[str, object], start: date, end: date) -> list[dict[str, object]]:
    if payload.get("timezone") != ZONE or payload.get("utc_offset_seconds") != 10800:
        raise ValueError("unexpected Open-Meteo timezone")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("missing hourly response")
    expected = [
        (datetime.combine(start, datetime.min.time()) + timedelta(hours=offset)).strftime(
            "%Y-%m-%dT%H:%M"
        )
        for offset in range(((end - start).days + 1) * 24)
    ]
    if hourly.get("time") != expected:
        raise ValueError("Open-Meteo returned an incomplete or reordered hourly grid")
    for name in VARIABLES:
        values = hourly.get(name)
        if not isinstance(values, list) or len(values) != len(expected):
            raise ValueError(f"incomplete Open-Meteo variable: {name}")
    return [
        {"time": stamp, **{name: hourly[name][index] for name in VARIABLES}}
        for index, stamp in enumerate(expected)
    ]


def fetch_year(year: int, output: Path) -> dict[str, object]:
    """Write one checked year; hourly accumulations align to their ending timestamp."""
    if not 1940 <= year <= date.today().year - 1:
        raise ValueError("year must be a completed archive year since 1940")
    if output.exists():
        raise FileExistsError(output)
    first = date(year, 1, 1)
    last = date(year + 1, 1, 1)
    collected: list[dict[str, object]] = []
    requests = 0
    cursor = first
    while cursor <= last:
        chunk_end = min(cursor + timedelta(days=13), last)
        collected.extend(_parse_chunk(_request(cursor, chunk_end), cursor, chunk_end))
        requests += 1
        cursor = chunk_end + timedelta(days=1)
        if cursor <= last:
            time.sleep(0.25)
    expected_hours = (last - first).days * 24
    if len(collected) != expected_hours + 24:
        raise ValueError("Open-Meteo year is incomplete")
    shifted = {row["time"]: row for row in collected}
    columns = ["date", "hour", *VARIABLES]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    missing = {name: 0 for name in VARIABLES}
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter=";")
            writer.writeheader()
            for raw in collected[:expected_hours]:
                stamp = datetime.strptime(str(raw["time"]), "%Y-%m-%dT%H:%M")
                next_stamp = (stamp + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
                row = {"date": stamp.date().isoformat(), "hour": stamp.hour}
                for name in VARIABLES:
                    value = shifted[next_stamp][name] if name in ACCUMULATIONS else raw[name]
                    missing[name] += value is None
                    row[name] = "" if value is None else value
                writer.writerow(row)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "schema": SCHEMA, "source": URL, "source_model": "best_match",
        "source_licence": "CC BY 4.0", "latitude": LATITUDE, "longitude": LONGITUDE,
        "timezone": ZONE, "date_range": [first.isoformat(), (last - timedelta(days=1)).isoformat()],
        "rows": expected_hours, "variables": list(VARIABLES), "units": {
            "temperature_2m": "°C", "relative_humidity_2m": "%",
            "precipitation": "mm", "rain": "mm", "snowfall": "cm",
            "snow_depth": "m", "weather_code": "WMO code", "cloud_cover": "%",
            "wind_speed_10m": "km/h", "wind_gusts_10m": "km/h",
        },
        "accumulation_alignment": "precipitation/rain/snowfall for [H,H+1) use API H+1",
        "requests": requests, "missing_values": missing, "csv_sha256": digest,
        "feature_version": SCHEMA,
        "leakage": "retrospective actual weather, including dates after 2025-10-31",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_weather(path: Path) -> pd.DataFrame:
    spatial = path.name in ("route-weather.csv", "route-weather.csv.gz")
    manifest_path = path.parent / "manifest.json" if spatial else path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = (
        manifest.get("sha256", {}).get(path.name) if spatial else manifest.get("csv_sha256")
    )
    if manifest.get("schema") != (SPATIAL_SCHEMA if spatial else SCHEMA) or (
        expected_hash != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("weather manifest or checksum mismatch")
    rows = pd.read_csv(path, sep=";")
    keys = ["route"] if spatial else []
    if list(rows.columns) != [*keys, "date", "hour", *VARIABLES]:
        raise ValueError("unexpected weather columns")
    rows["date"] = pd.to_datetime(rows.date, format="%Y-%m-%d").dt.date
    expected_rows = manifest["route_rows"] if spatial else manifest["rows"]
    if rows.duplicated([*keys, "date", "hour"]).any() or len(rows) != expected_rows:
        raise ValueError("duplicate or missing weather keys")
    if not rows.hour.between(0, 23).all():
        raise ValueError("invalid weather hour")
    first, last = (date.fromisoformat(value) for value in manifest["date_range"])
    expected_hours = ((last - first).days + 1) * 24
    if not rows.date.between(first, last).all():
        raise ValueError("weather date outside manifest range")
    if spatial:
        from tramflow_ml.competition import LABEL_ROUTES

        if rows.groupby("route").size().to_dict() != {
            route: expected_hours for route in LABEL_ROUTES
        }:
            raise ValueError("weather route-hour grid is incomplete")
    elif len(rows) != expected_hours:
        raise ValueError("weather hourly grid is incomplete")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tramflow_ml.weather")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch_year(args.year, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

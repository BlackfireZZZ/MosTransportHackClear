"""Archived operational weather forecasts usable only after their issue time."""

import argparse
import gzip
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.competition import LABEL_ROUTES

SCHEMA = "moscow-issued-weather.v1"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LATITUDE = 55.7558
LONGITUDE = 37.6173
START = date(2025, 1, 1)
END = date(2025, 10, 31)
FIELDS = ("temperature_2m", "precipitation")


def _url() -> str:
    return URL + "?" + urllib.parse.urlencode({
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": START.isoformat(), "end_date": END.isoformat(),
        "hourly": ",".join(FIELDS), "timezone": "Europe/Moscow",
    })


def parse_response(payload: dict[str, object]) -> pd.DataFrame:
    """Require a complete Moscow civil-hour grid without target-period values."""
    if payload.get("timezone") != "Europe/Moscow":
        raise ValueError("issued weather timezone differs")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("issued weather hourly body missing")
    expected = pd.date_range(START, END + timedelta(days=1), freq="h", inclusive="left")
    times = list(expected.strftime("%Y-%m-%dT%H:%M"))
    if hourly.get("time") != times:
        raise ValueError("issued weather hour grid differs")
    data = {field: hourly.get(field) for field in FIELDS}
    if any(not isinstance(values, list) or len(values) != len(times)
           for values in data.values()):
        raise ValueError("issued weather field incomplete")
    frame = pd.DataFrame({"date": expected.date, "hour": expected.hour, **data})
    if frame[list(FIELDS)].isna().any().any():
        raise ValueError("issued weather contains missing values")
    return frame


def collect(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    request = urllib.request.Request(_url(), headers={"User-Agent": "tramflow-issued-weather/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read()
    frame = parse_response(json.loads(raw))
    output.parent.mkdir(parents=True, exist_ok=True)
    csv = frame.to_csv(index=False, sep=";", float_format="%.4f").encode()
    with output.open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0,
                                                   filename="") as zipped:
        zipped.write(csv)
    manifest = {
        "schema": SCHEMA, "source": _url(), "source_kind": "stitched archived operational forecast",
        "source_licence": "CC BY 4.0", "timezone": "Europe/Moscow",
        "date_range": [START.isoformat(), END.isoformat()], "rows": len(frame),
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "csv_gz_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "availability_policy": "use only valid dates at least two days before forecast origin",
        "limitations": (
            "city-centre proxy; archived first-hours forecast, not a full 61-day forecast"
        ),
    }
    (output.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def load_issued_weather(path: Path) -> pd.DataFrame:
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("date_range") !=
            [START.isoformat(), END.isoformat()] or
            hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("csv_gz_sha256")):
        raise ValueError("issued weather manifest or checksum differs")
    rows = pd.read_csv(path, sep=";")
    if list(rows.columns) != ["date", "hour", *FIELDS]:
        raise ValueError("issued weather columns differ")
    rows["date"] = pd.to_datetime(rows.date, format="%Y-%m-%d").dt.date
    expected = pd.MultiIndex.from_product(
        [pd.date_range(START, END).date, range(24)], names=["date", "hour"],
    ).to_frame(index=False)
    if not rows[["date", "hour"]].equals(expected) or rows[list(FIELDS)].isna().any().any():
        raise ValueError("issued weather grid differs")
    repeated = pd.concat([rows.assign(route=route) for route in LABEL_ROUTES],
                         ignore_index=True)
    return repeated[["route", "date", "hour", *FIELDS]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.output), sort_keys=True))


if __name__ == "__main__":
    main()

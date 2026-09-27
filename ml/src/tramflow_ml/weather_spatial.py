"""Spatial Open-Meteo weather for Moscow tram stop zones and scored routes."""

import argparse
import csv
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from shutil import copyfile

from tramflow_ml.competition import LABEL_ROUTES
from tramflow_ml.weather import ACCUMULATIONS, SPATIAL_SCHEMA, URL, VARIABLES, ZONE, _parse_chunk

SCHEMA = SPATIAL_SCHEMA
INVENTORY_BATCH_SIZE = 25
MAX_WEIGHTED_CALLS_PER_MINUTE = 480


def _pace(locations: int) -> None:
    time.sleep(locations * 60 / MAX_WEIGHTED_CALLS_PER_MINUTE)


def scored_stops(path: Path) -> list[dict[str, object]]:
    """Select OSM stop-position nodes carrying scored route references."""
    with path.open(encoding="utf-8", newline="") as stream:
        stops = list(csv.DictReader(stream))
    scored = {str(route) for route in LABEL_ROUTES}
    selected = [row for row in stops if set(row["routes"].split(";")) & scored]
    if not selected:
        raise ValueError("no scored-route stops in graph")
    if len({row["osm_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate scored stop id")
    if not scored <= {route for row in selected for route in row["routes"].split(";")}:
        raise ValueError("scored route has no OSM stop")
    return selected


def discover_stop_cells(
    stops: list[dict[str, object]], cache: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]],
           dict[str, list[float]], int, int]:
    """Ask Open-Meteo where each stop resolves, then fetch one year per unique cell."""
    locations = []
    new_requests = 0
    cached = 0
    for batch_index, start in enumerate(range(0, len(stops), INVENTORY_BATCH_SIZE)):
        batch = stops[start:start + INVENTORY_BATCH_SIZE]
        requested = [{"latitude": row["lat"], "longitude": row["lon"]} for row in batch]
        path = cache / f"inventory-{batch_index:03d}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached += 1
        else:
            payload = _fetch_chunk(date(2025, 1, 1), date(2025, 1, 1), requested)
            path.write_text(json.dumps(payload), encoding="utf-8")
            new_requests += 1
            _pace(len(requested))
        if not isinstance(payload, list) or len(payload) != len(batch):
            raise ValueError("incomplete stop-to-weather-grid inventory")
        for stop, point in zip(batch, payload, strict=True):
            _parse_chunk(point, date(2025, 1, 1), date(2025, 1, 1))
            locations.append((stop, (float(point["latitude"]), float(point["longitude"]))))
    cells = sorted({point for _, point in locations})
    ids = {point: f"G{index:03d}" for index, point in enumerate(cells, start=1)}
    representatives: dict[str, dict[str, object]] = {}
    counts: Counter[str] = Counter()
    exposure: Counter[tuple[int, str]] = Counter()
    mapped = []
    scored = {str(route) for route in LABEL_ROUTES}
    for stop, point in locations:
        zone_id = ids[point]
        representatives.setdefault(zone_id, stop)
        counts[zone_id] += 1
        routes = sorted(set(str(stop["routes"]).split(";")) & scored)
        mapped.append({"osm_id": stop["osm_id"], "name": stop["name"],
                       "lat": stop["lat"], "lon": stop["lon"],
                       "routes": ";".join(routes), "zone_id": zone_id})
        for route in routes:
            exposure[int(route), zone_id] += 1
    zones = [{"zone_id": ids[point],
              "latitude": representatives[ids[point]]["lat"],
              "longitude": representatives[ids[point]]["lon"],
              "grid_latitude": point[0], "grid_longitude": point[1],
              "stop_count": counts[ids[point]]} for point in cells]
    weights = [{"route": route, "zone_id": zone_id, "stop_count": count}
               for (route, zone_id), count in sorted(exposure.items())]
    coordinates = {ids[point]: [point[0], point[1]] for point in cells}
    return zones, mapped, weights, coordinates, new_requests, cached


def _fetch_chunk(start: date, end: date, zones: list[dict[str, object]]) -> list[dict[str, object]]:
    parameters = {
        "latitude": ",".join(str(zone["latitude"]) for zone in zones),
        "longitude": ",".join(str(zone["longitude"]) for zone in zones),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": ZONE, "models": "best_match", "hourly": ",".join(VARIABLES),
    }
    request = urllib.request.Request(
        URL + "?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": "tramflow-weather-research/1.0"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
            if not isinstance(payload, list) or len(payload) != len(zones):
                raise ValueError("unexpected number of Open-Meteo locations")
            return payload
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable retry state")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_spatial(source_dir: Path, output_dir: Path) -> dict[str, object]:
    """Publish only verified open weather and OSM mappings, with reproducible gzip bytes."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected spatial weather manifest")
    original = manifest["sha256"]
    for name, digest in original.items():
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source weather checksum mismatch: {name}")
    output_dir.mkdir(parents=True)
    hashes = {}
    for name in ("zones.csv", "stops.csv", "route-zones.csv"):
        target = output_dir / name
        copyfile(source_dir / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ("zone-weather.csv", "route-weather.csv"):
        target = output_dir / (name + ".gz")
        with (source_dir / name).open("rb") as source, target.open("wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
                while chunk := source.read(1024 * 1024):
                    compressed.write(chunk)
        hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["raw_csv_sha256"] = original
    manifest["sha256"] = hashes
    manifest["storage"] = "gzip hourly CSV; deterministic mtime=0"
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def fetch_spatial_year(year: int, stops_path: Path, output_dir: Path) -> dict[str, object]:
    if year != 2025:
        raise ValueError("this scored-route export is fixed to 2025")
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(output_dir / "manifest.json")
    stops = scored_stops(stops_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "chunks"
    cache.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(stops_path.read_bytes()).hexdigest()
    marker = cache / "stop-source.sha256"
    if marker.exists():
        if marker.read_text(encoding="ascii").strip() != source_hash:
            raise ValueError("cached stop inventory belongs to a different source")
    elif any(cache.iterdir()):
        raise ValueError("unbound weather cache has no stop-source fingerprint")
    else:
        marker.write_text(source_hash + "\n", encoding="ascii")
    zones, stops, weights, actual_coordinates, inventory_requests, inventory_cached = (
        discover_stop_cells(stops, cache)
    )
    requested_zone_count = len(zones)
    first, exclusive_end = date(year, 1, 1), date(year + 1, 1, 1)
    cursor = first
    raw: dict[str, dict[str, dict[str, object]]] = {str(zone["zone_id"]): {} for zone in zones}
    requests = 0
    cached_chunks = 0
    while cursor <= exclusive_end:
        end = min(cursor + timedelta(days=13), exclusive_end)
        chunk_path = cache / f"{cursor}_{end}.json"
        if chunk_path.exists():
            payload = json.loads(chunk_path.read_text(encoding="utf-8"))
            cached_chunks += 1
        else:
            payload = _fetch_chunk(cursor, end, zones)
            chunk_path.write_text(json.dumps(payload), encoding="utf-8")
            requests += 1
            if end < exclusive_end:
                _pace(len(zones))
        for zone, location in zip(zones, payload, strict=True):
            key = str(zone["zone_id"])
            returned = [float(location["latitude"]), float(location["longitude"])]
            if returned != actual_coordinates[key]:
                raise ValueError("weather grid changed after stop-cell inventory")
            for row in _parse_chunk(location, cursor, end):
                raw[key][str(row["time"])] = row
        cursor = end + timedelta(days=1)
    n_hours = (exclusive_end - first).days * 24
    zone_rows = []
    route_rows = []
    for offset in range(n_hours):
        stamp = datetime.combine(first, datetime.min.time()) + timedelta(hours=offset)
        key = stamp.strftime("%Y-%m-%dT%H:%M")
        next_key = (stamp + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
        period = {"date": stamp.date().isoformat(), "hour": stamp.hour}
        hour_zone = {}
        for zone in zones:
            zone_id = str(zone["zone_id"])
            values = {name: (raw[zone_id][next_key][name] if name in ACCUMULATIONS
                             else raw[zone_id][key][name]) for name in VARIABLES}
            zone_rows.append({"zone_id": zone_id, **period, **values})
            hour_zone[zone_id] = values
        for route in LABEL_ROUTES:
            counts = [(str(item["zone_id"]), int(str(item["stop_count"])))
                      for item in weights if item["route"] == route]
            total = sum(count for _, count in counts)
            values = {}
            for name in VARIABLES:
                observed = [(hour_zone[zone][name], count) for zone, count in counts]
                if any(value is None for value, _ in observed):
                    values[name] = ""
                elif name == "weather_code":
                    by_code: Counter[int] = Counter()
                    for value, count in observed:
                        by_code[int(str(value))] += count
                    values[name] = by_code.most_common(1)[0][0]
                else:
                    weighted = sum(float(str(value)) * count for value, count in observed)
                    values[name] = round(weighted / total, 4)
            route_rows.append({"route": route, **period, **values})
    hashes = {}
    hashes["zones.csv"] = _write_csv(output_dir / "zones.csv",
                                     ["zone_id", "latitude", "longitude", "grid_latitude",
                                      "grid_longitude", "stop_count"], zones)
    hashes["stops.csv"] = _write_csv(output_dir / "stops.csv",
                                     ["osm_id", "name", "lat", "lon", "routes", "zone_id"], stops)
    hashes["route-zones.csv"] = _write_csv(output_dir / "route-zones.csv",
                                           ["route", "zone_id", "stop_count"], weights)
    hashes["zone-weather.csv"] = _write_csv(output_dir / "zone-weather.csv",
                                            ["zone_id", "date", "hour", *VARIABLES], zone_rows)
    hashes["route-weather.csv"] = _write_csv(output_dir / "route-weather.csv",
                                             ["route", "date", "hour", *VARIABLES], route_rows)
    manifest: dict[str, object] = {
        "schema": SCHEMA, "year": year, "source": URL, "source_model": "best_match",
        "source_licence": "CC BY 4.0", "timezone": ZONE,
        "stop_source": str(stops_path),
        "stop_source_sha256": source_hash,
        "stop_source_date": "2026-09-18", "zone_count": len(zones),
        "requested_zone_count": requested_zone_count, "stop_count": len(stops),
        "zone_rows": len(zone_rows), "route_rows": len(route_rows),
        "date_range": [first.isoformat(), (exclusive_end - timedelta(days=1)).isoformat()],
        "variables": list(VARIABLES), "actual_grid_coordinates": actual_coordinates,
        "units": {"temperature_2m": "°C", "relative_humidity_2m": "%",
                  "precipitation": "mm", "rain": "mm", "snowfall": "cm",
                  "snow_depth": "m", "weather_code": "WMO code", "cloud_cover": "%",
                  "wind_speed_10m": "km/h", "wind_gusts_10m": "km/h"},
        "accumulation_alignment": "precipitation/rain/snowfall for [H,H+1) use API H+1",
        "http_requests_this_run": requests + inventory_requests,
        "cached_chunks": cached_chunks + inventory_cached,
        "total_chunks": requests + cached_chunks,
        "inventory_locations": len(stops),
        "inventory_requests": inventory_requests + inventory_cached,
        "estimated_weighted_calls": (
            len(stops) + (requests + cached_chunks) * requested_zone_count
        ),
        "sha256": hashes, "feature_version": SCHEMA,
        "leakage": "retrospective actual weather, including dates after 2025-10-31",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tramflow_ml.weather_spatial")
    parser.add_argument("--stops", type=Path)
    parser.add_argument("--package-from", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.stops is None) == (args.package_from is None):
        parser.error("choose exactly one of --stops or --package-from")
    result = (
        package_spatial(args.package_from, args.output_dir)
        if args.package_from is not None
        else fetch_spatial_year(2025, args.stops, args.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

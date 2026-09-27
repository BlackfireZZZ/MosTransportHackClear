"""Derive dated service planning proxies from one checksum-bound GTFS snapshot."""

import argparse
import csv
import hashlib
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

SOURCE_URL = "https://files.mobilitydatabase.org/mdb-3226/mdb-3226-202606301640/mdb-3226-202606301640.zip"
SOURCE_SHA256 = "dec3ec0ccc2efb7f1327ee6f26a0636a4a271494490e6debedcb2c1ccc579e76"
ROUTES = {"1", "5", "7", "11", "12", "17", "25", "26", "28", "50"}


def seconds(value: str) -> int:
    hours, minutes, seconds_value = map(int, value.split(":"))
    if minutes >= 60 or seconds_value >= 60 or hours >= 48:
        raise ValueError(f"invalid service-day clock: {value}")
    return hours * 3600 + minutes * 60 + seconds_value


def percentile(values: list[float], share: float) -> int:
    return round(values[int((len(values) - 1) * share)])


def build(archive: Path) -> dict:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("unexpected GTFS archive hash")
    with ZipFile(archive) as source:
        def rows(name: str):
            with source.open(name) as raw:
                yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))

        route_ids = {
            row["route_id"]: row["route_short_name"]
            for row in rows("routes.txt")
            if row["route_type"] == "0" and row["route_short_name"] in ROUTES
        }
        trips = {}
        for row in rows("trips.txt"):
            if row["route_id"] not in route_ids:
                continue
            parts = row["trip_id"].split("_")
            if len(parts) != 4 or parts[:2] != [row["route_id"], row["service_id"]]:
                raise ValueError("trip identifier cannot support duty grouping")
            trips[row["trip_id"]] = {
                "route": route_ids[row["route_id"]], "service": row["service_id"],
                "direction": row["direction_id"], "duty": parts[3],
                "first": None, "last": None,
            }
        if any(row["block_id"] for row in rows("trips.txt") if row["trip_id"] in trips):
            raise ValueError("source block_ids changed; reassess duty inference")
        calendars = {row["service_id"]: row for row in rows("calendar.txt")}
        if "calendar_dates.txt" in source.namelist():
            raise ValueError("calendar exceptions require explicit date handling")
        for row in rows("stop_times.txt"):
            trip = trips.get(row["trip_id"])
            if trip is None:
                continue
            sequence = int(row["stop_sequence"])
            if trip["first"] is None or sequence < trip["first"][0]:
                trip["first"] = (sequence, seconds(row["departure_time"]))
            if trip["last"] is None or sequence > trip["last"][0]:
                trip["last"] = (sequence, seconds(row["arrival_time"]))
        groups = defaultdict(list)
        starts = defaultdict(lambda: [0] * 48)
        for trip in trips.values():
            if trip["first"] is None or trip["last"] is None:
                raise ValueError("trip without stop times")
            route_service = (trip["route"], trip["service"])
            groups[(*route_service, trip["duty"])].append(trip)
            starts[route_service][trip["first"][1] // 3600] += 1
        cycles = defaultdict(list)
        for (route, service, _), group in groups.items():
            group.sort(key=lambda trip: trip["first"][1])
            for first, reverse, next_outbound in zip(group, group[1:], group[2:]):
                if first["direction"] == reverse["direction"] or first["direction"] != next_outbound["direction"]:
                    continue
                first_layover = reverse["first"][1] - first["last"][1]
                second_layover = next_outbound["first"][1] - reverse["last"][1]
                duration = (next_outbound["first"][1] - first["first"][1]) / 60
                if 0 <= first_layover <= 30 * 60 and 0 <= second_layover <= 30 * 60 and 20 <= duration <= 240:
                    cycles[(route, service)].append(duration)
        profiles = {}
        for (route, service), values in sorted(cycles.items(), key=lambda item: (int(item[0][0]), item[0][1])):
            if len(values) < 20:
                raise ValueError(f"too few paired cycles: {route}/{service}")
            values.sort()
            calendar = calendars[service]
            weekdays = [index for index, name in enumerate(("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")) if calendar[name] == "1"]
            profile = {
                "serviceId": service, "weekdays": weekdays,
                "validFrom": calendar["start_date"], "validTo": calendar["end_date"],
                "cycleMinutes": round(statistics.median(values)),
                "cycleP10": percentile(values, 0.1), "cycleP90": percentile(values, 0.9),
                "pairedCycles": len(values), "departureStartsByServiceHour": starts[(route, service)],
            }
            profiles.setdefault(route, []).append(profile)
        if set(profiles) != ROUTES - {"5"} or any(len(items) != 2 for items in profiles.values()):
            raise ValueError("incomplete route-service profiles")
        return {
            "schema": "network-service-estimates.v1",
            "sourceUrl": SOURCE_URL, "sourceSha256": SOURCE_SHA256,
            "captureDate": "2026-06-30", "timezone": "Europe/Moscow",
            "calendarExceptionsAvailable": False, "blockIdAvailable": False,
            "method": "same trip_id duty suffix; alternating directions; two layovers 0–30 minutes; first departure to next same-direction departure",
            "profiles": profiles,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.archive), ensure_ascii=False, indent=2) + "\n")

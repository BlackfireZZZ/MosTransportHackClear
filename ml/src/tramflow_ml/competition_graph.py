"""Build source-labelled route direction structure from dated OSM relation edits."""

import argparse
import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from tramflow_ml.competition import LABEL_ROUTES

MOSCOW = ZoneInfo("Europe/Moscow")
STOP_ROLES = frozenset({"stop", "stop_entry_only", "stop_exit_only"})
COLUMNS = (
    "route", "direction_id", "available_at", "valid_from", "valid_to",
    "stop_count", "edge_count", "length_m", "source_version",
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_osm_structure(
    histories: Path, source_index: Path, *, through: date = date(2025, 10, 31)
) -> tuple[list[tuple[object, ...]], dict[str, object]]:
    """Use only relation versions edited before the requested civil-date end."""
    index_bytes = source_index.read_bytes()
    index = json.loads(index_bytes)
    references = [entry for entry in index["osm"] if entry["file"].startswith("relation-")]
    rows: list[tuple[object, ...]] = []
    sources = []
    excluded_incomplete_versions = 0
    for entry in references:
        name = entry["file"]
        parsed = urlparse(entry["url"])
        relation_text = name.removeprefix("relation-").removesuffix("-history.json")
        if (
            Path(name).name != name
            or parsed.scheme != "https"
            or parsed.netloc != "api.openstreetmap.org"
            or not parsed.path.endswith(f"/{relation_text}/history.json")
        ):
            raise ValueError("unexpected OSM relation source")
        content = (histories / name).read_bytes()
        source_hash = _digest(content)
        payload = json.loads(content)
        versions = payload.get("elements")
        if not isinstance(versions, list) or not versions or "remark" in payload:
            raise ValueError(f"invalid OSM relation history: {name}")
        relation_id = int(relation_text)
        if any(
            item.get("type") != "relation" or item.get("id") != relation_id
            for item in versions
        ):
            raise ValueError(f"OSM relation identity mismatch: {name}")
        sources.append(
            {
                "file": name,
                "sha256": source_hash,
                "reference_sha256": entry["sha256"],
                "reference_hash_matches": source_hash == entry["sha256"],
                "url": entry["url"],
            }
        )
        latest_by_day: dict[tuple[int, date], tuple[datetime, tuple[object, ...]]] = {}
        previous_route: int | None = None
        for item in sorted(versions, key=lambda version: version["timestamp"]):
            instant = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError("OSM edit timestamp must be timezone-aware")
            edited = instant.astimezone(MOSCOW).date()
            if edited > through:
                continue
            tags = item.get("tags", {})
            route_label = tags.get("ref", "")
            invalid_route = (
                tags.get("type") != "route"
                or tags.get("route") != "tram"
                or not isinstance(route_label, str)
                or not route_label.isdecimal()
                or int(route_label) not in LABEL_ROUTES
            )
            stops = [
                member for member in item.get("members", [])
                if member.get("type") == "node" and member.get("role") in STOP_ROLES
            ]
            current_route = int(route_label) if not invalid_route and len(stops) >= 2 else None
            if not invalid_route and len(stops) < 2:
                excluded_incomplete_versions += 1
            affected = {route for route in (previous_route, current_route) if route is not None}
            for route in affected:
                active = route == current_route
                row: tuple[object, ...] = (
                    route, f"osm:{relation_id}", edited.isoformat(),
                    edited.isoformat(), date.max.isoformat(),
                    len(stops) if active else 0, len(stops) - 1 if active else 0,
                    "", f"osm-relation:{relation_id}:v{item['version']}"
                    if active else f"osm-inactive:{relation_id}:v{item['version']}",
                )
                key = (route, edited)
                previous = latest_by_day.get(key)
                if previous is None or instant > previous[0]:
                    latest_by_day[key] = (instant, row)
            previous_route = current_route
        rows.extend(row for _, row in latest_by_day.values())
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    report: dict[str, object] = {
        "schema": "osm-route-direction-structure.v1",
        "source_index_sha256": _digest(index_bytes),
        "through": through.isoformat(),
        "timezone": "Europe/Moscow",
        "rows": len(rows),
        "excluded_incomplete_versions": excluded_incomplete_versions,
        "routes": sorted({int(str(row[0])) for row in rows}),
        "sources": sources,
        "direction_identity": "local OSM relation ID; not organizer direction_id",
        "length_policy": "unavailable; no 2025-verified geometry",
        "service_history_verified": False,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(prog="tramflow-direction-structure")
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = build_osm_structure(args.histories, args.source_index)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    report["output_sha256"] = _digest(args.output.read_bytes())
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "routes", "output_sha256")}))


if __name__ == "__main__":
    main()

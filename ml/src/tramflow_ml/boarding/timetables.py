"""Parse published stop departure clocks; snapshots never imply operated trips."""

import hashlib
import html
import json
import re
from datetime import date
from typing import Any, Literal


def _text(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split())


def parse_timetable_html(
    raw: bytes,
    *,
    route_ref: str,
    direction: int,
    source_url: str,
    capture_date: str,
    status: Literal["historical_snapshot", "current_proxy"],
) -> dict[str, Any]:
    """Validate a complete source page and retain civil-clock seconds in Moscow time.

    Departure arrays have no vehicle/trip identity. Midnight values stay below 86400;
    the publisher does not establish whether these belong to the preceding service day.
    """
    if len(raw) > 6_000_000:
        raise ValueError("timetable exceeds 6 MB source limit")
    content = raw.decode("utf-8")
    if "</html>" not in content.lower():
        raise ValueError("incomplete HTML source")
    date.fromisoformat(capture_date)
    heading = re.search(r"<h1\b[^>]*>(.*?)</h1>", content, re.S)
    if heading is None or _text(heading[1]) != route_ref:
        raise ValueError("route heading mismatch")
    if "icon-tramway" not in heading[1]:
        raise ValueError("source route is not a tram")
    coordinates = re.search(r'data-coords="([^"]+)"', content)
    if coordinates is None:
        raise ValueError("missing stop geography")
    features = [
        feature
        for feature in json.loads(html.unescape(coordinates[1]))["features"]
        if feature["geometry"]["type"] == "Point"
    ]
    blocks = [
        block
        for block in re.findall(r"<li\b[^>]*>.*?</li>", content, re.S)
        if "data-stop=" in block
    ]
    if not blocks or len(features) != len(blocks):
        raise ValueError("incomplete stop sequence")
    rows: list[dict[str, Any]] = []
    schedule_dates: set[str] = set()
    sequences: list[int] = []
    for index, feature in enumerate(features):
        longitude, latitude = feature["geometry"]["coordinates"]
        if feature["geometry"]["type"] != "Point" or not (
            35 < float(longitude) < 40 and 54 < float(latitude) < 58
        ):
            raise ValueError("invalid Moscow stop coordinates")
        name = feature["properties"]["hintContent"]
        departures: list[int] = []
        if index < len(blocks):
            block = blocks[index]
            attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', block.split(">", 1)[0]))
            if int(attrs["data-direction"]) != direction:
                raise ValueError("direction mismatch")
            schedule_date = attrs["data-date"]
            date.fromisoformat(schedule_date)
            schedule_dates.add(schedule_date)
            sequences.append(int(attrs["data-stop"]))
            stop_heading = re.search(r'<div class="a_dotted[^\"]*">(.*?)</div>', block, re.S)
            if stop_heading is None or _text(stop_heading[1]) != _text(name):
                raise ValueError("stop name/order mismatch")
            hours = list(re.finditer(r'<div class="dt1"><strong>(\d{2}):</strong></div>', block))
            for hour_index, match in enumerate(hours):
                hour = int(match[1])
                end = hours[hour_index + 1].start() if hour_index + 1 < len(hours) else len(block)
                minute_cells = re.findall(
                    r'<div class="div10"[^>]*>(.*?)</div>', block[match.end() : end], re.S
                )
                for cell in minute_cells:
                    text = _text(cell)
                    if not text.isdigit() or hour > 23 or int(text) > 59:
                        raise ValueError("invalid departure clock")
                    departures.append(hour * 3600 + int(text) * 60)
            if not departures:
                raise ValueError("missing departure times for nonterminal stop")
        rows.append(
            {
                "route_ref": route_ref,
                "direction": direction,
                "stop_sequence": index,
                "stop_id": str(feature["id"]),
                "stop_name": name,
                "lat": float(latitude),
                "lon": float(longitude),
                "departures_seconds": sorted(departures),
                "source_url": source_url,
                "capture_date": capture_date,
                "status": status,
                "complete": True,
            }
        )
    expected_sequence = list(range(sequences[0], sequences[0] + len(features)))
    if len(schedule_dates) != 1 or sequences != expected_sequence:
        raise ValueError("inconsistent source dates or stop sequence")
    schedule_date = next(iter(schedule_dates))
    if status == "historical_snapshot":
        archive_match = re.match(r"https://web\.archive\.org/web/(\d{8})\d{6}(?:id_)?/", source_url)
        if (
            archive_match is None
            or archive_match[1] != capture_date.replace("-", "")
            or schedule_date != capture_date
        ):
            raise ValueError("historical snapshot requires matching capture and schedule date")
    elif status != "current_proxy":
        raise ValueError("invalid source status")
    for row in rows:
        row["schedule_date"] = schedule_date
    return {
        "schema_version": "published-stop-timetable-v1",
        "timezone": "Europe/Moscow",
        "source_url": source_url,
        "capture_date": capture_date,
        "schedule_date": schedule_date,
        "status": status,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "complete": True,
        "trip_identity_available": False,
        "stops": rows,
    }

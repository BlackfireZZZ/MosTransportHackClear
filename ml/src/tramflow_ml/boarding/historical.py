"""Dated OSM membership is a route hypothesis, never certified service history."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

STOP_ROLES = frozenset({"stop", "stop_entry_only", "stop_exit_only"})
PLATFORM_ROLES = frozenset({"platform", "platform_entry_only", "platform_exit_only"})


def _instant(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("OSM timestamps and as_of must have a timezone")
    return result


def _history(payload: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if "remark" in payload or not isinstance(payload.get("elements"), list):
        raise ValueError("incomplete or malformed OSM history")
    elements = payload["elements"]
    if not elements or any(e.get("type") != kind for e in elements):
        raise ValueError(f"history must contain {kind} versions")
    if len({e.get("id") for e in elements}) != 1:
        raise ValueError("history contains multiple identities")
    if any(type(e.get("version")) is not int or e["version"] < 1 for e in elements):
        raise ValueError("invalid OSM version")
    ordered = sorted(elements, key=lambda e: e["version"])
    if len({e["version"] for e in ordered}) != len(ordered):
        raise ValueError("duplicate OSM version")
    times = [_instant(e["timestamp"]) for e in ordered]
    if times != sorted(times):
        raise ValueError("OSM versions have reversed timestamps")
    return ordered


def _coordinate(node: dict[str, Any]) -> tuple[float, float]:
    lat, lon = node.get("lat"), node.get("lon")
    if (
        not isinstance(lat, (float, int))
        or not isinstance(lon, (float, int))
        or isinstance(lat, bool)
        or isinstance(lon, bool)
    ):
        raise ValueError("missing or invalid stop coordinates")
    if (
        not math.isfinite(lat)
        or not math.isfinite(lon)
        or not -90 <= lat <= 90
        or not -180 <= lon <= 180
    ):
        raise ValueError("missing or invalid stop coordinates")
    return float(lat), float(lon)


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    delta_lat = lat2 - lat1
    delta_lon = math.radians(b["lon"] - a["lon"])
    h = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def parse_history(
    payload: dict[str, Any],
    graph: dict[str, Any],
    as_of: datetime,
    *,
    node_histories: dict[int, dict[str, Any]] | None = None,
    source_url: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Select membership in [edit, next edit); expose geometry's separate date.

    Platforms do not add visits. Repeated stop members remain distinct visits.
    Missing geometry fails; a missing graph edge is an explicit straight-line prior.
    """
    at = _instant(as_of.isoformat())
    history = _history(payload, "relation")
    indices = [i for i, e in enumerate(history) if _instant(e["timestamp"]) <= at]
    if not indices:
        return None
    index = indices[-1]
    relation = history[index]
    if relation.get("visible") is False:
        return None
    tags = relation.get("tags", {})
    if tags.get("type") != "route" or tags.get("route") != "tram" or not tags.get("ref"):
        return None
    if not isinstance(relation.get("members"), list):
        raise ValueError("missing relation members")
    raw_nodes = graph.get("nodes", [])
    nodes = {n["id"]: n for n in raw_nodes}
    if len(nodes) != len(raw_nodes):
        raise ValueError("duplicate graph node")
    graph_timestamp = graph.get("metadata", {}).get("osm_data_timestamp")
    if not graph_timestamp:
        raise ValueError("graph source timestamp required")
    _instant(graph_timestamp)
    visits: list[dict[str, Any]] = []
    for member in relation["members"]:
        role = member.get("role")
        if not isinstance(role, str) or member.get("type") not in {"node", "way", "relation"}:
            raise ValueError("malformed route member")
        if type(member.get("ref")) is not int or member["ref"] < 1:
            raise ValueError("invalid route member reference")
        if role not in STOP_ROLES:
            if (
                role.startswith("stop")
                or role.startswith("platform")
                and role not in PLATFORM_ROLES
            ):
                raise ValueError("unsupported passenger-stop role")
            continue
        if member["type"] != "node":
            raise ValueError("stop role requires a stop-position node")
        ref = member["ref"]
        coordinate_source = "committed_current_graph"
        coordinate_timestamp = graph_timestamp
        node = nodes.get(ref)
        if node is None:
            if node_histories is None or ref not in node_histories:
                raise ValueError(f"missing stop geometry: {ref}")
            versions = _history(node_histories[ref], "node")
            if versions[0]["id"] != ref:
                raise ValueError("supplemental node history identity mismatch")
            candidates = [v for v in versions if _instant(v["timestamp"]) <= at]
            if not candidates or candidates[-1].get("visible") is False:
                raise ValueError(f"no historical stop geometry: {ref}")
            node = candidates[-1]
            coordinate_source = "osm_node_history"
            coordinate_timestamp = node["timestamp"]
        lat, lon = _coordinate(node)
        visits.append(
            {
                "sequence": len(visits),
                "stop_id": str(ref),
                "name": (node.get("name") or node.get("tags", {}).get("name") or f"node/{ref}"),
                "lat": lat,
                "lon": lon,
                "role": role,
                "boardable": role != "stop_exit_only",
                "coordinate_source": coordinate_source,
                "coordinate_timestamp": coordinate_timestamp,
            }
        )
    if len(visits) < 2 or not any(v["boardable"] for v in visits):
        raise ValueError("route requires two visits and a boarding stop")
    edges = {}
    for edge in graph.get("links", []):
        key = (str(edge["source"]), str(edge["target"]))
        if key in edges:
            raise ValueError("duplicate directed graph edge")
        edges[key] = edge
    lengths, sources = [], []
    for a, b in zip(visits, visits[1:], strict=False):
        edge = edges.get((a["stop_id"], b["stop_id"]))
        if edge is None:
            length = _distance(a, b)
            source = "straight_line_unverified_track"
        else:
            length = edge["length_m"]
            if type(length) not in (int, float) or not math.isfinite(length) or length < 0:
                raise ValueError("invalid graph edge length")
            source = "committed_current_graph_rail"
        lengths.append(float(length))
        sources.append(source)
    return {
        "route": tags["ref"],
        "relation_id": relation["id"],
        "version": relation["version"],
        "timestamp": relation["timestamp"],
        "valid_from": relation["timestamp"],
        "valid_to": history[index + 1]["timestamp"] if index + 1 < len(history) else None,
        "as_of": as_of.isoformat(),
        "from": tags.get("from"),
        "to": tags.get("to"),
        "source_url": source_url,
        "source_sha256": source_sha256,
        "historical_verified": False,
        "period_basis": "osm_edit_time_not_service_validity",
        "temporary_diversions_verified": False,
        "ordered_visits": visits,
        "edge_lengths_m": lengths,
        "distance_source": sources,
        "geometry_timestamp": graph_timestamp,
    }


def load_historical_catalog(
    source_directory: Path,
    graph_path: Path,
    as_of: datetime,
) -> list[dict[str, Any]]:
    """Load cached public OSM histories with verified manifest hashes; no network."""
    manifest = json.loads((source_directory / "history-manifest.json").read_text())
    sources = {}
    for entry in manifest:
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("history manifest path escapes directory")
        path = source_directory / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"history source hash mismatch: {name}")
        if name in sources:
            raise ValueError("duplicate manifest source")
        sources[name] = (json.loads(data), entry)
    graph = json.loads(graph_path.read_text())
    node_histories = {
        payload["elements"][0]["id"]: payload
        for name, (payload, _) in sources.items()
        if name.startswith("node-") and name.endswith("-history.json")
    }
    patterns = []
    for name, (payload, entry) in sorted(sources.items()):
        if not name.startswith("relation-") or not name.endswith("-history.json"):
            continue
        pattern = parse_history(
            payload,
            graph,
            as_of,
            node_histories=node_histories,
            source_url=entry["url"],
            source_sha256=entry["sha256"],
        )
        if pattern is not None:
            patterns.append(pattern)
    return patterns

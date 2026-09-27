#!/usr/bin/env python3
"""Build a routable graph of the Moscow tram network from our Overpass API.

Nodes are tram stops, edges are the track between two consecutive stops on at
least one route. Edge length follows the real rail geometry rather than a
straight line: PTv2 route relations place `stop_position` nodes directly on the
track ways, so the track can be walked node by node between two stops.

Publishes GraphML, node-link JSON, GeoJSON and a CSV pair as a versioned set
under --out-dir, selected by tram_graph.manifest.json.

Only the standard library is used, so the script runs with a bare interpreter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_API = "http://2.29.6.201/api/interpreter"

STOP_ROLES = {"stop", "stop_entry_only", "stop_exit_only"}
PLATFORM_ROLES = {"platform", "platform_entry_only", "platform_exit_only"}

QUERY = """
[out:json][timeout:{timeout}];
area["name"="{area}"]["admin_level"="{admin_level}"]->.searchArea;
relation["type"="route"]["route"="tram"](area.searchArea)->.routes;
.routes out body;
node(r.routes);
out body;
way(r.routes);
out body;
node(w);
out skel qt;
"""


def fetch(api_url: str, query: str, timeout: int) -> dict:
    """POST a query to Overpass and return the decoded JSON payload."""
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(api_url, data=data)
    with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
        raw = resp.read().decode("utf-8")
    if not raw.lstrip().startswith("{"):
        snippet = " ".join(raw.split())[:300]
        raise RuntimeError(f"Overpass did not return JSON: {snippet}")
    payload = json.loads(raw)
    validate_payload(payload)
    return payload


def validate_payload(payload: object) -> None:
    """Reject error responses and structures outside this query's output contract."""
    if not isinstance(payload, dict):
        raise ValueError("Overpass response must be an object")
    if "remark" in payload:
        raise ValueError("Overpass returned a remark; the extract may be incomplete")
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Overpass elements must be an array")
    if not isinstance(payload.get("osm3s", {}), dict):
        raise ValueError("Overpass osm3s must be an object")
    for element in elements:
        if not isinstance(element, dict):
            raise ValueError("Overpass elements must contain objects")
        kind = element.get("type")
        if kind not in ("node", "way", "relation"):
            raise ValueError("Overpass element type must be node, way or relation")
        if type(element.get("id")) is not int or element["id"] <= 0:
            raise ValueError("Overpass element id must be a positive integer")
        tags = element.get("tags", {})
        if not isinstance(tags, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in tags.items()
        ):
            raise ValueError("Overpass tags must map strings to strings")
        if kind == "node":
            for field, limit in (("lat", 90), ("lon", 180)):
                value = element.get(field)
                if type(value) not in (int, float) or not -limit <= value <= limit:
                    raise ValueError(f"Overpass node {field} is invalid")
        elif kind == "way":
            refs = element.get("nodes")
            if not isinstance(refs, list) or any(
                type(ref) is not int or ref <= 0 for ref in refs
            ):
                raise ValueError(
                    "Overpass way nodes must contain positive integer refs"
                )
        else:
            members = element.get("members")
            if not isinstance(members, list):
                raise ValueError("Overpass relation members must be an array")
            for member in members:
                if (
                    not isinstance(member, dict)
                    or member.get("type") not in ("node", "way", "relation")
                    or type(member.get("ref")) is not int
                    or member["ref"] <= 0
                    or not isinstance(member.get("role"), str)
                ):
                    raise ValueError("Overpass relation member is invalid")
    node_ids = {element["id"] for element in elements if element["type"] == "node"}
    way_ids = {element["id"] for element in elements if element["type"] == "way"}
    for element in elements:
        if element["type"] == "way" and any(
            ref not in node_ids for ref in element["nodes"]
        ):
            raise ValueError("Overpass extract is missing a referenced track node")
        if element["type"] == "relation":
            for member in element["members"]:
                if (
                    member["type"] == "node"
                    and member["ref"] not in node_ids
                    or member["type"] == "way"
                    and member["ref"] not in way_ids
                ):
                    raise ValueError(
                        "Overpass extract is missing a referenced route member"
                    )


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) pairs."""
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def order_track_nodes(way_ids: list[int], ways: dict[int, list[int]]) -> list[int]:
    """Concatenate a route's track ways into one ordered list of node ids.

    Ways are stored in OSM with an arbitrary direction, so each one is flipped
    when needed to attach to the end of what has been assembled so far. A way
    that touches neither end starts a new segment; the gap is left in place and
    the caller falls back to straight-line distance across it.
    """
    seq: list[int] = []
    for way_id in way_ids:
        nodes = ways.get(way_id)
        if not nodes:
            continue
        if not seq:
            seq = list(nodes)
            continue
        if nodes[0] == seq[-1]:
            seq.extend(nodes[1:])
        elif nodes[-1] == seq[-1]:
            seq.extend(reversed(nodes[:-1]))
        elif nodes[-1] == seq[0]:
            seq = list(nodes[:-1]) + seq
        elif nodes[0] == seq[0]:
            seq = list(reversed(nodes[1:])) + seq
        else:
            seq.extend(nodes)
    return seq


def stop_name(node: dict, fallback: str | None) -> str:
    tags = node.get("tags", {})
    return tags.get("name") or tags.get("ref") or fallback or f"node/{node['id']}"


def build(payload: dict) -> tuple[OrderedDict, OrderedDict, dict]:
    """Turn the Overpass payload into (nodes, edges, stats)."""
    elements = payload["elements"]
    nodes_raw = {e["id"]: e for e in elements if e["type"] == "node"}
    ways_raw = {e["id"]: e.get("nodes", []) for e in elements if e["type"] == "way"}
    way_tags = {e["id"]: e.get("tags", {}) for e in elements if e["type"] == "way"}
    routes = [e for e in elements if e["type"] == "relation"]

    coord = {
        nid: (n["lat"], n["lon"])
        for nid, n in nodes_raw.items()
        if "lat" in n and "lon" in n
    }

    graph_nodes: OrderedDict[int, dict] = OrderedDict()
    graph_edges: OrderedDict[tuple[int, int], dict] = OrderedDict()
    stats = {
        "routes_total": len(routes),
        "routes_used": 0,
        "routes_skipped_no_stops": 0,
        "edges_via_track": 0,
        "edges_via_straight_line": 0,
        "stops_missing_name": 0,
    }

    for rel in routes:
        tags = rel.get("tags", {})
        members = rel.get("members", [])
        track_ways = [
            m["ref"] for m in members if m["role"] == "" and m["type"] == "way"
        ]
        seq = order_track_nodes(track_ways, ways_raw)
        seq_index: dict[int, list[int]] = {}
        for i, nid in enumerate(seq):
            seq_index.setdefault(nid, []).append(i)

        platform_name_after: dict[int, str] = {}
        last_stop_ref = None
        for m in members:
            if m["role"] in STOP_ROLES and m["type"] == "node":
                last_stop_ref = m["ref"]
            elif m["role"] in PLATFORM_ROLES and last_stop_ref is not None:
                if m["type"] == "node":
                    name = nodes_raw.get(m["ref"], {}).get("tags", {}).get("name")
                else:
                    name = way_tags.get(m["ref"], {}).get("name")
                if name:
                    platform_name_after.setdefault(last_stop_ref, name)
                last_stop_ref = None

        stops = [
            m["ref"]
            for m in members
            if m["role"] in STOP_ROLES and m["type"] == "node" and m["ref"] in coord
        ]
        if len(stops) < 2:
            stats["routes_skipped_no_stops"] += 1
            continue
        stats["routes_used"] += 1

        route_label = tags.get("ref") or tags.get("name") or f"relation/{rel['id']}"

        for sid in stops:
            if sid not in graph_nodes:
                node = nodes_raw[sid]
                name = stop_name(node, platform_name_after.get(sid))
                if name.startswith("node/"):
                    stats["stops_missing_name"] += 1
                graph_nodes[sid] = {
                    "osm_id": sid,
                    "name": name,
                    "lat": coord[sid][0],
                    "lon": coord[sid][1],
                    "routes": set(),
                }
            graph_nodes[sid]["routes"].add(route_label)

        cursor = 0
        for a, b in zip(stops, stops[1:], strict=False):
            geometry: list[tuple[float, float]] = []
            length = None
            ia = next((i for i in seq_index.get(a, []) if i >= cursor), None)
            ib = None
            if ia is not None:
                ib = next((i for i in seq_index.get(b, []) if i > ia), None)
            if ia is not None and ib is not None:
                span = seq[ia : ib + 1]
                geometry = [coord[n] for n in span if n in coord]
                if len(geometry) >= 2:
                    length = sum(
                        haversine(p, q)
                        for p, q in zip(geometry, geometry[1:], strict=False)
                    )
                    stats["edges_via_track"] += 1
                    cursor = ib
            if length is None:
                geometry = [coord[a], coord[b]]
                length = haversine(coord[a], coord[b])
                stats["edges_via_straight_line"] += 1

            key = (a, b)
            edge = graph_edges.get(key)
            if edge is None:
                graph_edges[key] = {
                    "source": a,
                    "target": b,
                    "length_m": round(length, 1),
                    "routes": {route_label},
                    "geometry": geometry,
                }
            else:
                edge["routes"].add(route_label)
                if length < edge["length_m"]:
                    edge["length_m"] = round(length, 1)
                    edge["geometry"] = geometry

    return graph_nodes, graph_edges, stats


def sorted_routes(values: set[str]) -> list[str]:
    """Sort route labels so numeric refs read 1, 2, 10 rather than 1, 10, 2."""

    def key(v: str):
        return (0, int(v), "") if v.isdecimal() else (1, 0, v)

    return sorted(values, key=key)


def write_graphml(
    path: Path, nodes: OrderedDict, edges: OrderedDict, meta: dict
) -> None:
    ns = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}graphml")
    keys = [
        ("d_name", "node", "name", "string"),
        ("d_lat", "node", "lat", "double"),
        ("d_lon", "node", "lon", "double"),
        ("d_nroutes", "node", "routes", "string"),
        ("d_len", "edge", "length_m", "double"),
        ("d_eroutes", "edge", "routes", "string"),
    ]
    for kid, domain, name, typ in keys:
        ET.SubElement(
            root,
            f"{{{ns}}}key",
            {"id": kid, "for": domain, "attr.name": name, "attr.type": typ},
        )
    graph = ET.SubElement(
        root, f"{{{ns}}}graph", {"id": "moscow_tram", "edgedefault": "directed"}
    )
    desc = ET.SubElement(graph, f"{{{ns}}}desc")
    desc.text = json.dumps(meta, ensure_ascii=False, sort_keys=True)

    for nid, n in nodes.items():
        el = ET.SubElement(graph, f"{{{ns}}}node", {"id": str(nid)})
        for kid, value in (
            ("d_name", n["name"]),
            ("d_lat", f"{n['lat']:.7f}"),
            ("d_lon", f"{n['lon']:.7f}"),
            ("d_nroutes", ";".join(sorted_routes(n["routes"]))),
        ):
            ET.SubElement(el, f"{{{ns}}}data", {"key": kid}).text = str(value)

    for i, ((a, b), e) in enumerate(edges.items()):
        el = ET.SubElement(
            graph,
            f"{{{ns}}}edge",
            {"id": f"e{i}", "source": str(a), "target": str(b)},
        )
        ET.SubElement(el, f"{{{ns}}}data", {"key": "d_len"}).text = str(e["length_m"])
        ET.SubElement(el, f"{{{ns}}}data", {"key": "d_eroutes"}).text = ";".join(
            sorted_routes(e["routes"])
        )

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_json(path: Path, nodes: OrderedDict, edges: OrderedDict, meta: dict) -> None:
    doc = {
        "metadata": meta,
        "directed": True,
        "nodes": [
            {
                "id": n["osm_id"],
                "name": n["name"],
                "lat": round(n["lat"], 7),
                "lon": round(n["lon"], 7),
                "routes": sorted_routes(n["routes"]),
            }
            for n in nodes.values()
        ],
        "links": [
            {
                "source": e["source"],
                "target": e["target"],
                "length_m": e["length_m"],
                "routes": sorted_routes(e["routes"]),
            }
            for e in edges.values()
        ],
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def write_geojson(
    path: Path, nodes: OrderedDict, edges: OrderedDict, meta: dict
) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(n["lon"], 7), round(n["lat"], 7)],
            },
            "properties": {
                "id": n["osm_id"],
                "name": n["name"],
                "routes": sorted_routes(n["routes"]),
            },
        }
        for n in nodes.values()
    ]
    features += [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(lon, 7), round(lat, 7)] for lat, lon in e["geometry"]
                ],
            },
            "properties": {
                "source": e["source"],
                "target": e["target"],
                "length_m": e["length_m"],
                "routes": sorted_routes(e["routes"]),
            },
        }
        for e in edges.values()
        if len(e["geometry"]) >= 2
    ]
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "metadata": meta, "features": features},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_csv(
    nodes_path: Path, edges_path: Path, nodes: OrderedDict, edges: OrderedDict
) -> None:
    # lineterminator is explicit: csv defaults to CRLF regardless of platform.
    with nodes_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["osm_id", "name", "lat", "lon", "routes"])
        for n in nodes.values():
            w.writerow(
                [
                    n["osm_id"],
                    n["name"],
                    f"{n['lat']:.7f}",
                    f"{n['lon']:.7f}",
                    ";".join(sorted_routes(n["routes"])),
                ]
            )
    with edges_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["source", "target", "length_m", "routes"])
        for e in edges.values():
            w.writerow(
                [
                    e["source"],
                    e["target"],
                    e["length_m"],
                    ";".join(sorted_routes(e["routes"])),
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--api-url", default=DEFAULT_API)
    ap.add_argument("--area", default="Москва")
    ap.add_argument("--admin-level", default="4")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument(
        "--rollback",
        metavar="VERSION",
        help="activate a saved snapshot without fetching",
    )
    args = ap.parse_args()

    # A bare interpreter has no installed app package; this shared validator and
    # publisher uses only stdlib, and must be the same one the API reader uses.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.domain.tram_graph import TramGraphDataError
    from app.infrastructure.graph_artifacts import activate_snapshot, publish_graph

    if args.rollback:
        try:
            activate_snapshot(args.out_dir, args.rollback)
        except (TramGraphDataError, OSError) as error:
            print(f"graph rollback failed: {error}", file=sys.stderr)
            return 1
        print(f"activated graph snapshot {args.rollback}", file=sys.stderr)
        return 0

    query = QUERY.format(
        area=args.area, admin_level=args.admin_level, timeout=args.timeout
    )
    print(f"querying {args.api_url} for tram routes in {args.area}...", file=sys.stderr)
    try:
        payload = fetch(args.api_url, query, args.timeout)
    except (ValueError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        print(f"graph extraction failed: {exc}", file=sys.stderr)
        return 1

    nodes, edges, stats = build(payload)
    if not nodes:
        print("no tram stops found -- check --area and --admin-level", file=sys.stderr)
        return 1

    meta = {
        "source": "OpenStreetMap via Overpass API",
        "license": "ODbL 1.0",
        "api_url": args.api_url,
        "area": args.area,
        "osm_data_timestamp": payload.get("osm3s", {}).get("timestamp_osm_base"),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stop_count": len(nodes),
        "edge_count": len(edges),
        **stats,
    }

    def write_artifacts(directory: Path) -> None:
        write_graphml(directory / "tram_graph.graphml", nodes, edges, meta)
        write_json(directory / "tram_graph.json", nodes, edges, meta)
        write_geojson(directory / "tram_graph.geojson", nodes, edges, meta)
        write_csv(
            directory / "tram_stops.csv", directory / "tram_edges.csv", nodes, edges
        )

    try:
        version = publish_graph(args.out_dir, write_artifacts)
    except (TramGraphDataError, OSError) as error:
        print(f"graph publication failed: {error}", file=sys.stderr)
        return 1
    print(f"  graph_version: {version}", file=sys.stderr)

    for k, v in meta.items():
        print(f"  {k}: {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Versioned graph sets committed by one atomic manifest switch on local POSIX storage."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.tram_graph import TramGraphDataError
from app.infrastructure.tram_graph_validation import parse_network

MANIFEST_NAME = "tram_graph.manifest.json"
STORE_NAME = "graph-store"
ARTIFACT_NAMES = (
    "tram_graph.json",
    "tram_graph.geojson",
    "tram_graph.graphml",
    "tram_stops.csv",
    "tram_edges.csv",
)
_VERSION = re.compile(r"[0-9a-f]{64}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TramGraphDataError("duplicate JSON object key in graph artifact")
        result[key] = value
    return result


def _json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TramGraphDataError("invalid JSON graph artifact") from exc
    if not isinstance(value, dict):
        raise TramGraphDataError("graph artifact must contain a JSON object")
    return value


def _xml_text(element: ET.Element, key: str) -> str:
    matches = element.findall(f"{{http://graphml.graphdrawing.org/xmlns}}data[@key='{key}']")
    if len(matches) != 1:
        raise TramGraphDataError("missing or duplicate GraphML data field")
    return matches[0].text or ""


def _bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise TramGraphDataError("graph artifacts must not be symbolic links")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise TramGraphDataError(f"cannot read graph artifact {path.name}") from exc


def _encoded(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source(graph: dict[str, Any]) -> dict[str, Any]:
    meta = graph.get("metadata")
    if not isinstance(meta, dict):
        raise TramGraphDataError("publication requires graph source metadata")
    for key in ("source", "license", "area", "osm_data_timestamp", "generated_at"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise TramGraphDataError(f"publication requires metadata {key}")
    for key in ("osm_data_timestamp", "generated_at"):
        try:
            parsed = datetime.fromisoformat(meta[key])
        except ValueError as exc:
            raise TramGraphDataError(f"invalid publication timestamp {key}") from exc
        if parsed.tzinfo is None:
            raise TramGraphDataError(f"publication timestamp {key} requires a timezone")
    for key, collection in (("stop_count", "nodes"), ("edge_count", "links")):
        if key in meta and (type(meta[key]) is not int or meta[key] != len(graph[collection])):
            raise TramGraphDataError(f"publication metadata {key} does not match graph")
    return meta


def _validate_formats(blobs: dict[str, bytes]) -> tuple[dict[str, Any], dict[str, Any]]:
    graph, geo = _json(blobs[ARTIFACT_NAMES[0]]), _json(blobs[ARTIFACT_NAMES[1]])
    if graph.get("directed") is not True:
        raise TramGraphDataError("published node-link graph must declare directed=true")
    network = parse_network(graph, geo)
    meta = _source(graph)
    if geo.get("metadata") != meta:
        raise TramGraphDataError("JSON/GeoJSON source metadata mismatch")

    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = {(e["source"], e["target"]): e for e in graph["links"]}
    seen_nodes: set[int] = set()
    seen_edges: set[tuple[int, int]] = set()
    for feature in geo["features"]:
        props, geometry = feature["properties"], feature["geometry"]
        if geometry["type"] == "Point":
            key = props["id"]
            if key in seen_nodes or key not in nodes:
                raise TramGraphDataError("GeoJSON point set mismatch")
            node = nodes[key]
            if props != {k: node[k] for k in ("id", "name", "routes")} or geometry[
                "coordinates"
            ] != [node["lon"], node["lat"]]:
                raise TramGraphDataError("JSON/GeoJSON stop mismatch")
            seen_nodes.add(key)
        elif geometry["type"] == "LineString":
            pair = (props["source"], props["target"])
            if pair in seen_edges or pair not in edges or props != edges[pair]:
                raise TramGraphDataError("JSON/GeoJSON edge mismatch")
            coordinates = geometry["coordinates"]
            source, target = nodes[pair[0]], nodes[pair[1]]
            if coordinates[0] != [source["lon"], source["lat"]] or coordinates[-1] != [
                target["lon"],
                target["lat"],
            ]:
                raise TramGraphDataError("GeoJSON edge endpoints mismatch")
            seen_edges.add(pair)
        else:
            raise TramGraphDataError("unsupported geometry in published graph")
    if seen_nodes != nodes.keys() or seen_edges != edges.keys():
        raise TramGraphDataError("GeoJSON is not a complete graph projection")

    try:
        stops_csv = list(csv.reader(io.StringIO(blobs["tram_stops.csv"].decode("utf-8"))))
        edges_csv = list(csv.reader(io.StringIO(blobs["tram_edges.csv"].decode("utf-8"))))
        expected_stops = [["osm_id", "name", "lat", "lon", "routes"]] + [
            [str(n.id), n.name, f"{n.latitude:.7f}", f"{n.longitude:.7f}", ";".join(n.routes)]
            for n in network.stops.values()
        ]
        expected_edges = [["source", "target", "length_m", "routes"]] + [
            [str(e["source"]), str(e["target"]), str(e["length_m"]), ";".join(e["routes"])]
            for e in graph["links"]
        ]
        if stops_csv != expected_stops or edges_csv != expected_edges:
            raise TramGraphDataError("CSV graph projection mismatch")
        xml = ET.fromstring(blobs["tram_graph.graphml"])
        ns = "{http://graphml.graphdrawing.org/xmlns}"
        expected_keys = {
            "d_name": ("node", "name", "string"),
            "d_lat": ("node", "lat", "double"),
            "d_lon": ("node", "lon", "double"),
            "d_nroutes": ("node", "routes", "string"),
            "d_len": ("edge", "length_m", "double"),
            "d_eroutes": ("edge", "routes", "string"),
        }
        key_elements = xml.findall(f"{ns}key")
        actual_keys = {
            key.get("id"): (key.get("for"), key.get("attr.name"), key.get("attr.type"))
            for key in key_elements
        }
        if (
            xml.tag != f"{ns}graphml"
            or len(xml.findall(f"{ns}graph")) != 1
            or len(key_elements) != len(expected_keys)
            or actual_keys != expected_keys
        ):
            raise TramGraphDataError("GraphML schema mismatch")
        xml_graph = xml.find(f"{ns}graph")
        if xml_graph is None or xml_graph.get("edgedefault") != "directed":
            raise TramGraphDataError("GraphML must contain a directed graph")
        if _json((xml_graph.findtext(f"{ns}desc") or "").encode()) != meta:
            raise TramGraphDataError("GraphML source metadata mismatch")
        for index, edge in enumerate(xml_graph.findall(f"{ns}edge")):
            if set(edge.attrib) != {"id", "source", "target"} or edge.get("id") != f"e{index}":
                raise TramGraphDataError("GraphML edge attributes mismatch")
        xml_stops = [
            [node.attrib["id"]]
            + [_xml_text(node, key) for key in ("d_name", "d_lat", "d_lon", "d_nroutes")]
            for node in xml_graph.findall(f"{ns}node")
        ]
        xml_edges = [
            [edge.attrib["source"], edge.attrib["target"]]
            + [_xml_text(edge, key) for key in ("d_len", "d_eroutes")]
            for edge in xml_graph.findall(f"{ns}edge")
        ]
        if xml_stops != expected_stops[1:] or xml_edges != expected_edges[1:]:
            raise TramGraphDataError("GraphML graph projection mismatch")
    except (ValueError, KeyError, UnicodeError, ET.ParseError, csv.Error) as exc:
        raise TramGraphDataError("malformed CSV/GraphML graph artifact") from exc
    return graph, geo


def _manifest(blobs: dict[str, bytes], graph: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "source": _source(graph),
        "files": {
            name: {"sha256": _digest(raw), "size_bytes": len(raw)} for name, raw in blobs.items()
        },
    }
    return {**body, "version": _digest(_encoded(body))}


def _check_manifest(manifest: dict[str, Any]) -> str:
    if set(manifest) != {"schema_version", "version", "source", "files"}:
        raise TramGraphDataError("invalid graph manifest fields")
    version = manifest["version"]
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise TramGraphDataError("invalid graph snapshot version")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise TramGraphDataError("unsupported graph manifest schema")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(ARTIFACT_NAMES):
        raise TramGraphDataError("graph manifest must include exactly five artifacts")
    for entry in manifest["files"].values():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sha256", "size_bytes"}
            or not isinstance(entry["sha256"], str)
            or not _VERSION.fullmatch(entry["sha256"])
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise TramGraphDataError("invalid graph artifact checksum entry")
    try:
        expected = _digest(_encoded({k: v for k, v in manifest.items() if k != "version"}))
    except (ValueError, TypeError, RecursionError) as exc:
        raise TramGraphDataError("invalid graph manifest values") from exc
    if expected != version:
        raise TramGraphDataError("graph manifest version mismatch")
    return version


def _snapshot(root: Path, version: str) -> Path:
    if not _VERSION.fullmatch(version):
        raise TramGraphDataError("invalid graph snapshot version")
    snapshots = root / "snapshots"
    path = snapshots / version
    if snapshots.is_symlink() or path.is_symlink():
        raise TramGraphDataError("graph snapshot directories must not be symbolic links")
    return path


def _load_set(root: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = _snapshot(root, _check_manifest(manifest))
    if _json(_bytes(directory / MANIFEST_NAME)) != manifest:
        raise TramGraphDataError("active and snapshot manifests do not match")
    blobs = {name: _bytes(directory / name) for name in ARTIFACT_NAMES}
    for name, raw in blobs.items():
        if manifest["files"][name] != {"sha256": _digest(raw), "size_bytes": len(raw)}:
            raise TramGraphDataError(f"graph artifact checksum mismatch: {name}")
    graph, geo = _validate_formats(blobs)
    if manifest["source"] != _source(graph):
        raise TramGraphDataError("manifest source metadata mismatch")
    return graph, geo


def load_active_graph(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one manifest and validate the exact bytes of its immutable artifact set."""
    _, graph, geo = load_active_graph_snapshot(root)
    return graph, geo


def load_active_graph_snapshot(root: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return the version from the same manifest that selected the verified bytes."""
    store = _store(root)
    manifest = _json(_bytes(store / MANIFEST_NAME))
    graph, geo = _load_set(store, manifest)
    return _check_manifest(manifest), graph, geo


def _store(root: Path) -> Path:
    store = root / STORE_NAME
    if store.is_symlink():
        raise TramGraphDataError("graph store must not be a symbolic link")
    return store


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def activate_snapshot(root: Path, version: str) -> None:
    """Validate then select a prior complete version; cached API readers require restart."""
    _activate_in_store(_store(root), version)


def _activate_in_store(root: Path, version: str) -> None:
    manifest = _json(_bytes(_snapshot(root, version) / MANIFEST_NAME))
    if _check_manifest(manifest) != version:
        raise TramGraphDataError("snapshot directory and version mismatch")
    _load_set(root, manifest)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".manifest-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_encoded(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / MANIFEST_NAME)
        _sync_directory(root)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_into_store(root: Path, write_artifacts: Callable[[Path], None]) -> str:
    snapshots = root / "snapshots"
    if snapshots.is_symlink():
        raise TramGraphDataError("snapshot store must not be a symbolic link")
    with tempfile.TemporaryDirectory(dir=root, prefix=".graph-staging-") as staging:
        stage = Path(staging)
        write_artifacts(stage)
        blobs = {name: _bytes(stage / name) for name in ARTIFACT_NAMES}
        graph, _ = _validate_formats(blobs)
        manifest = _manifest(blobs, graph)
        version = _check_manifest(manifest)
        (stage / MANIFEST_NAME).write_bytes(_encoded(manifest))
        for name in (*ARTIFACT_NAMES, MANIFEST_NAME):
            with (stage / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(stage)
        snapshots.mkdir(exist_ok=True)
        target = _snapshot(root, version)
        try:
            os.rename(stage, target)
        except OSError:
            if not target.is_dir():
                raise
            _load_set(root, manifest)
        _sync_directory(snapshots)
        _sync_directory(root)
        _activate_in_store(root, version)
        return version


def publish_graph(root: Path, write_artifacts: Callable[[Path], None]) -> str:
    """Publish a complete set; first adoption also atomically installs the whole store."""
    root.mkdir(parents=True, exist_ok=True)
    store = _store(root)
    if store.exists():
        return _publish_into_store(store, write_artifacts)
    with tempfile.TemporaryDirectory(dir=root, prefix=".graph-initial-") as initial:
        staged_store = Path(initial)
        version = _publish_into_store(staged_store, write_artifacts)
        try:
            os.rename(staged_store, store)
        except OSError:
            if not store.is_dir():
                raise
            return _publish_into_store(_store(root), write_artifacts)
        _sync_directory(root)
        return version

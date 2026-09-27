from copy import deepcopy
from datetime import UTC, datetime

import pytest

from tramflow_ml.boarding.historical import parse_history


def graph():
    return {
        "metadata": {"osm_data_timestamp": "2026-09-18T20:20:11Z"},
        "nodes": [
            {"id": 1, "name": "A", "lat": 55.0, "lon": 37.0},
            {"id": 2, "name": "B", "lat": 55.01, "lon": 37.01},
        ],
        "links": [{"source": 1, "target": 2, "length_m": 1800}],
    }


def history():
    old = {
        "type": "relation",
        "id": 10,
        "version": 1,
        "timestamp": "2024-01-01T00:00:00Z",
        "tags": {"type": "route", "route": "tram", "ref": "1"},
        "members": [
            {"type": "node", "ref": 1, "role": "stop_entry_only"},
            {"type": "node", "ref": 91, "role": "platform_entry_only"},
            {"type": "node", "ref": 2, "role": "stop"},
            {"type": "node", "ref": 1, "role": "stop_exit_only"},
        ],
    }
    new = deepcopy(old)
    new.update(version=2, timestamp="2025-07-01T00:00:00Z")
    new["members"] = list(reversed(new["members"]))
    return {"elements": [new, old]}


def test_date_boundary_repeated_visits_boarding_roles_and_explicit_geometry():
    before = parse_history(history(), graph(), datetime(2025, 6, 30, tzinfo=UTC))
    assert before["version"] == 1
    assert [v["stop_id"] for v in before["ordered_visits"]] == ["1", "2", "1"]
    assert [v["boardable"] for v in before["ordered_visits"]] == [True, True, False]
    assert before["edge_lengths_m"][0] == 1800
    assert before["distance_source"] == [
        "committed_current_graph_rail",
        "straight_line_unverified_track",
    ]
    assert before["edge_lengths_m"][1] > 1000
    assert before["historical_verified"] is False
    assert before["temporary_diversions_verified"] is False
    assert before["geometry_timestamp"].startswith("2026")
    boundary = parse_history(history(), graph(), datetime(2025, 7, 1, tzinfo=UTC))
    assert boundary["version"] == 2
    assert before["valid_to"] == boundary["valid_from"]
    assert parse_history(history(), graph(), datetime(2023, 1, 1, tzinfo=UTC)) is None


def test_timezone_and_deleted_version():
    with pytest.raises(ValueError, match="timezone"):
        parse_history(history(), graph(), datetime(2025, 1, 1))
    data = history()
    data["elements"][0]["visible"] = False
    assert parse_history(data, graph(), datetime(2025, 8, 1, tzinfo=UTC)) is None


@pytest.mark.parametrize("mutation", ["remark", "duplicate", "timestamp", "stop_way", "bad_role"])
def test_rejects_malformed_history(mutation):
    data = history()
    if mutation == "remark":
        data["remark"] = "incomplete"
    elif mutation == "duplicate":
        data["elements"].append(data["elements"][0])
    elif mutation == "timestamp":
        data["elements"][1]["timestamp"] = "2026-01-01T00:00:00Z"
    elif mutation == "stop_way":
        data["elements"][1]["members"][0]["type"] = "way"
    else:
        data["elements"][1]["members"][0]["role"] = "stop_malformed"
    with pytest.raises(ValueError):
        parse_history(data, graph(), datetime(2025, 1, 1, tzinfo=UTC))


def test_missing_node_requires_asof_geometry_and_cannot_use_future_node_version():
    data = graph()
    data["nodes"].pop()
    at = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="missing stop geometry"):
        parse_history(history(), data, at)
    node = {
        "type": "node",
        "id": 2,
        "version": 1,
        "timestamp": "2024-01-01T00:00:00Z",
        "lat": 55.01,
        "lon": 37.01,
        "tags": {"name": "Historical B"},
    }
    future = dict(node, version=2, timestamp="2025-07-01T00:00:00Z", lat=60.0)
    result = parse_history(history(), data, at, node_histories={2: {"elements": [node, future]}})
    assert result["ordered_visits"][1]["lat"] == 55.01
    assert result["ordered_visits"][1]["name"] == "Historical B"
    with pytest.raises(ValueError, match="no historical stop geometry"):
        parse_history(history(), data, at, node_histories={2: {"elements": [future]}})


def test_source_manifest_hash_and_path_integrity(tmp_path):
    import hashlib
    import json

    from tramflow_ml.boarding.historical import load_historical_catalog

    raw = json.dumps(history()).encode()
    name = "relation-10-history.json"
    (tmp_path / name).write_bytes(raw)
    manifest = [
        {
            "file": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "url": "https://api.openstreetmap.org/api/0.6/relation/10/history.json",
        }
    ]
    manifest_path = tmp_path / "history-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph()))
    at = datetime(2025, 1, 1, tzinfo=UTC)
    loaded = load_historical_catalog(tmp_path, graph_path, at)
    assert loaded[0]["source_sha256"] == manifest[0]["sha256"]
    (tmp_path / name).write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_historical_catalog(tmp_path, graph_path, at)
    manifest[0]["file"] = "../escape.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_historical_catalog(tmp_path, graph_path, at)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, 100])
def test_coordinate_validation(value):
    data = graph()
    data["nodes"][0]["lat"] = value
    with pytest.raises(ValueError, match="coordinates"):
        parse_history(history(), data, datetime(2025, 1, 1, tzinfo=UTC))

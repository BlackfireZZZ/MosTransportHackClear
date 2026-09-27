from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from app.infrastructure.graph_artifacts import MANIFEST_NAME, STORE_NAME, load_active_graph

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_tram_graph.py"
ARTIFACTS = (
    "tram_graph.json",
    "tram_graph.geojson",
    "tram_graph.graphml",
    "tram_stops.csv",
    "tram_edges.csv",
)


@pytest.fixture
def extractor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("graph_extractor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def valid_payload() -> dict:
    return {
        "osm3s": {"timestamp_osm_base": "2026-09-23T00:00:00Z"},
        "elements": [
            {"type": "node", "id": 1, "lat": 55.7, "lon": 37.6, "tags": {"name": "A"}},
            {"type": "node", "id": 2, "lat": 55.71, "lon": 37.61, "tags": {"name": "B"}},
            {"type": "way", "id": 3, "nodes": [1, 2]},
            {
                "type": "relation",
                "id": 4,
                "tags": {"ref": "1"},
                "members": [
                    {"type": "node", "ref": 1, "role": "stop"},
                    {"type": "node", "ref": 2, "role": "stop"},
                    {"type": "way", "ref": 3, "role": ""},
                ],
            },
        ],
    }


def mock_response(monkeypatch: pytest.MonkeyPatch, extractor: ModuleType, raw: str) -> Mock:
    transport = Mock(return_value=io.BytesIO(raw.encode()))
    monkeypatch.setattr(extractor.urllib.request, "urlopen", transport)
    return transport


def existing_outputs(tmp_path: Path) -> dict[str, bytes]:
    original = {name: f"previous {name}\n".encode() for name in ARTIFACTS}
    for name, content in original.items():
        (tmp_path / name).write_bytes(content)
    return original


def snapshot(path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in path.iterdir()}


@pytest.mark.parametrize("remark", ["runtime error: Query timed out", "out of memory", "", None])
def test_remark_never_replaces_existing_extract(
    extractor: ModuleType,
    valid_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    remark: object,
) -> None:
    original = existing_outputs(tmp_path)
    valid_payload["remark"] = remark
    mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])

    assert extractor.main() == 1
    assert "remark" in capsys.readouterr().err
    assert snapshot(tmp_path) == original


@pytest.mark.parametrize(
    "raw",
    [
        "null",
        "[]",
        "42",
        "{}",
        '{"elements": null}',
        '{"elements": {}}',
        '{"elements": [null]}',
        '{"elements": [{}]}',
        '{"elements": [{"type": "area", "id": 1}]}',
        '{"elements": [{"type": "node", "id": true}]}',
        '{"elements": [{"type": "node", "id": 1, "lat": 91, "lon": 0}]}',
        '{"elements": [{"type": "way", "id": 1, "nodes": "bad"}]}',
        '{"elements": [{"type": "relation", "id": 1, "members": [null]}]}',
        '{"elements": [], "osm3s": null}',
        '{"elements":',
        "<html>query error</html>",
    ],
)
def test_malformed_response_never_replaces_existing_extract(
    extractor: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: str,
) -> None:
    original = existing_outputs(tmp_path)
    mock_response(monkeypatch, extractor, raw)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 1
    assert snapshot(tmp_path) == original


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("http://overpass.invalid", 504, "overloaded", None, None),
        URLError("offline"),
        TimeoutError("timeout"),
    ],
)
def test_transport_error_preserves_extract(
    extractor: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
) -> None:
    original = existing_outputs(tmp_path)
    monkeypatch.setattr(extractor.urllib.request, "urlopen", Mock(side_effect=error))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 1
    assert snapshot(tmp_path) == original


def test_valid_response_exports_graph(
    extractor: ModuleType,
    valid_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transport = mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 0
    manifest = json.loads((tmp_path / STORE_NAME / MANIFEST_NAME).read_text())
    directory = tmp_path / STORE_NAME / "snapshots" / manifest["version"]
    assert {path.name for path in directory.iterdir()} == {*ARTIFACTS, MANIFEST_NAME}
    graph, _ = load_active_graph(tmp_path)
    assert [node["id"] for node in graph["nodes"]] == [1, 2]
    assert len(graph["links"]) == 1
    assert graph["links"][0]["source"] == 1
    assert graph["links"][0]["target"] == 2
    assert graph["links"][0]["length_m"] > 0
    transport.assert_called_once()


@pytest.mark.parametrize("missing_id", [1, 2, 3])
def test_missing_referenced_elements_preserve_extract(
    extractor: ModuleType,
    valid_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_id: int,
) -> None:
    original = existing_outputs(tmp_path)
    valid_payload["elements"] = [e for e in valid_payload["elements"] if e["id"] != missing_id]
    mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 1
    assert snapshot(tmp_path) == original


def test_duplicate_output_clauses_can_repeat_valid_nodes(
    extractor: ModuleType,
    valid_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    valid_payload["elements"].append(dict(valid_payload["elements"][0]))
    mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 0
    assert len(load_active_graph(tmp_path)[0]["nodes"]) == 2


def test_rollback_cli_uses_saved_snapshot_without_network(
    extractor: ModuleType,
    valid_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 0
    manifest = json.loads((tmp_path / STORE_NAME / MANIFEST_NAME).read_text())
    transport = Mock(side_effect=AssertionError("rollback must not fetch"))
    monkeypatch.setattr(extractor.urllib.request, "urlopen", transport)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--out-dir",
            str(tmp_path),
            "--rollback",
            manifest["version"],
        ],
    )
    assert extractor.main() == 0
    transport.assert_not_called()


def test_standalone_rollback_needs_no_installed_packages(
    extractor: ModuleType,
    valid_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 0
    manifest = json.loads((tmp_path / STORE_NAME / MANIFEST_NAME).read_text())
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--out-dir",
            str(tmp_path),
            "--rollback",
            manifest["version"],
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "activated graph snapshot" in result.stderr


def test_partial_response_preserves_versioned_store_byte_for_byte(
    extractor: ModuleType,
    valid_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--out-dir", str(tmp_path)])
    assert extractor.main() == 0
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    valid_payload["remark"] = "timed out"
    mock_response(monkeypatch, extractor, json.dumps(valid_payload))
    assert extractor.main() == 1
    assert before == {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }

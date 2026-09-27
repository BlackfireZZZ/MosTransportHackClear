from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.tram_graph import TramGraphDataError
from app.infrastructure import graph_artifacts as artifacts
from app.infrastructure.repositories.tram_graph import FileTramGraphRepository

DATA = Path(__file__).resolve().parents[2] / "data"
ORIGINAL_TIME = "2026-09-19T13:43:17Z"
NEXT_TIME = "2026-09-23T10:00:00Z"


def copy_set(directory: Path, generated_at: str = ORIGINAL_TIME) -> None:
    for name in artifacts.ARTIFACT_NAMES:
        content = (DATA / name).read_bytes().replace(ORIGINAL_TIME.encode(), generated_at.encode())
        (directory / name).write_bytes(content)


def current_manifest(root: Path) -> bytes:
    return (root / artifacts.STORE_NAME / artifacts.MANIFEST_NAME).read_bytes()


def snapshot_directory(root: Path, version: str) -> Path:
    return root / artifacts.STORE_NAME / "snapshots" / version


def repository(root: Path) -> FileTramGraphRepository:
    return FileTramGraphRepository(root / "tram_graph.json", root / "tram_graph.geojson")


def test_committed_set_round_trip_preserves_topology_and_bytes(tmp_path: Path) -> None:
    version = artifacts.publish_graph(tmp_path, copy_set)

    graph, geo = artifacts.load_active_graph(tmp_path)
    network = repository(tmp_path).load()
    assert len(graph["nodes"]) == 856
    assert len(graph["links"]) == 919
    assert len(geo["features"]) == 856 + 919
    assert [len(component) for component in network.components] == [694, 162]
    assert network.stats() == repository(DATA).load().stats()
    manifest = json.loads(current_manifest(tmp_path))
    assert manifest["version"] == version
    assert set(manifest["files"]) == set(artifacts.ARTIFACT_NAMES)
    assert manifest["source"]["osm_data_timestamp"] == "2026-09-18T20:20:11Z"
    for name in artifacts.ARTIFACT_NAMES:
        assert (snapshot_directory(tmp_path, version) / name).read_bytes() == (
            DATA / name
        ).read_bytes()
    assert artifacts.publish_graph(tmp_path, copy_set) == version
    assert len(list((tmp_path / artifacts.STORE_NAME / "snapshots").iterdir())) == 1


@pytest.mark.parametrize("failed_file", artifacts.ARTIFACT_NAMES)
def test_interrupted_export_preserves_last_snapshot(tmp_path: Path, failed_file: str) -> None:
    previous = artifacts.publish_graph(tmp_path, copy_set)
    previous_manifest = current_manifest(tmp_path)

    def failing_writer(directory: Path) -> None:
        for name in artifacts.ARTIFACT_NAMES:
            (directory / name).write_bytes(b"partial")
            if name == failed_file:
                raise OSError("simulated disk full")

    with pytest.raises(OSError, match="simulated disk full"):
        artifacts.publish_graph(tmp_path, failing_writer)

    assert current_manifest(tmp_path) == previous_manifest
    assert len(repository(tmp_path).load().stops) == 856
    assert {p.name for p in (tmp_path / artifacts.STORE_NAME / "snapshots").iterdir()} == {previous}


@pytest.mark.parametrize("name", artifacts.ARTIFACT_NAMES)
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_reader_rejects_corrupt_or_missing_artifacts(
    tmp_path: Path, name: str, damage: str
) -> None:
    version = artifacts.publish_graph(tmp_path, copy_set)
    path = snapshot_directory(tmp_path, version) / name
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(TramGraphDataError):
        repository(tmp_path).load()


@pytest.mark.parametrize(
    "mismatch",
    ["source", "stop", "edge", "geometry", "csv", "graphml", "graphml_direction", "json_direction"],
)
def test_mismatched_projections_cannot_replace_active_set(tmp_path: Path, mismatch: str) -> None:
    artifacts.publish_graph(tmp_path, copy_set)
    before = current_manifest(tmp_path)

    def mismatched_writer(directory: Path) -> None:
        copy_set(directory, NEXT_TIME)
        if mismatch == "json_direction":
            path = directory / "tram_graph.json"
            graph = json.loads(path.read_text())
            del graph["directed"]
            path.write_text(json.dumps(graph))
        elif mismatch == "graphml_direction":
            path = directory / "tram_graph.graphml"
            path.write_bytes(path.read_bytes().replace(b"<edge ", b'<edge directed="false" ', 1))
        elif mismatch in ("csv", "graphml"):
            name = "tram_edges.csv" if mismatch == "csv" else "tram_graph.graphml"
            (directory / name).write_text("invalid")
        else:
            path = directory / "tram_graph.geojson"
            geo = json.loads(path.read_text())
            if mismatch == "source":
                geo["metadata"]["osm_data_timestamp"] = NEXT_TIME
            elif mismatch == "stop":
                geo["features"][0]["properties"]["name"] = "different stop"
            else:
                feature = next(f for f in geo["features"] if f["geometry"]["type"] == "LineString")
                if mismatch == "edge":
                    feature["properties"]["length_m"] += 1
                else:
                    feature["geometry"]["coordinates"][0][0] += 0.01
            path.write_text(json.dumps(geo))

    with pytest.raises(TramGraphDataError):
        artifacts.publish_graph(tmp_path, mismatched_writer)
    assert current_manifest(tmp_path) == before
    assert len(repository(tmp_path).load().stops) == 856


def test_failed_manifest_switch_preserves_active_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts.publish_graph(tmp_path, copy_set)
    before = current_manifest(tmp_path)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("manifest switch interrupted")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="manifest switch interrupted"):
        artifacts.publish_graph(tmp_path, lambda directory: copy_set(directory, NEXT_TIME))
    assert current_manifest(tmp_path) == before
    assert artifacts.load_active_graph(tmp_path)[0]["metadata"]["generated_at"] == ORIGINAL_TIME
    assert not list((tmp_path / artifacts.STORE_NAME).glob(".manifest-*"))


def test_rollback_validates_previous_snapshot(tmp_path: Path) -> None:
    first = artifacts.publish_graph(tmp_path, copy_set)
    second = artifacts.publish_graph(tmp_path, lambda directory: copy_set(directory, NEXT_TIME))
    assert second != first
    artifacts.activate_snapshot(tmp_path, first)
    assert repository(tmp_path).load().metadata.generated_at == ORIGINAL_TIME
    before = current_manifest(tmp_path)
    (snapshot_directory(tmp_path, second) / "tram_edges.csv").write_text("broken")
    with pytest.raises(TramGraphDataError, match="checksum"):
        artifacts.activate_snapshot(tmp_path, second)
    assert current_manifest(tmp_path) == before


@pytest.mark.parametrize("version", ["../outside", "", "f" * 64])
def test_invalid_rollback_does_not_change_manifest(tmp_path: Path, version: str) -> None:
    artifacts.publish_graph(tmp_path, copy_set)
    before = current_manifest(tmp_path)
    with pytest.raises(TramGraphDataError):
        artifacts.activate_snapshot(tmp_path, version)
    assert current_manifest(tmp_path) == before


def test_reader_captures_one_snapshot_during_concurrent_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = artifacts.publish_graph(tmp_path, copy_set)
    second = artifacts.publish_graph(tmp_path, lambda directory: copy_set(directory, NEXT_TIME))
    artifacts.activate_snapshot(tmp_path, first)
    read_bytes = artifacts._bytes
    switched = False

    def activate_during_read(path: Path) -> bytes:
        nonlocal switched
        if not switched and path.name == "tram_graph.json":
            switched = True
            artifacts.activate_snapshot(tmp_path, second)
        return read_bytes(path)

    monkeypatch.setattr(artifacts, "_bytes", activate_during_read)
    graph, geo = artifacts.load_active_graph(tmp_path)
    assert switched
    assert graph["metadata"]["generated_at"] == ORIGINAL_TIME
    assert geo["metadata"]["generated_at"] == ORIGINAL_TIME
    assert json.loads(current_manifest(tmp_path))["version"] == second


async def test_cached_repository_retains_its_complete_snapshot(tmp_path: Path) -> None:
    artifacts.publish_graph(tmp_path, copy_set)
    existing = repository(tmp_path)
    first = await existing.get_network()
    artifacts.publish_graph(tmp_path, lambda directory: copy_set(directory, NEXT_TIME))
    assert await existing.get_network() is first
    assert first.metadata.generated_at == ORIGINAL_TIME
    assert (await repository(tmp_path).get_network()).metadata.generated_at == NEXT_TIME


def test_missing_or_invalid_manifest_never_falls_back_to_legacy_files(tmp_path: Path) -> None:
    copy_set(tmp_path)
    artifacts.publish_graph(tmp_path, copy_set)
    active = tmp_path / artifacts.STORE_NAME / artifacts.MANIFEST_NAME
    active.write_text("[]")
    with pytest.raises(TramGraphDataError):
        repository(tmp_path).load()
    active.unlink()
    with pytest.raises(TramGraphDataError):
        repository(tmp_path).load()


def test_manifest_version_rejects_modified_checksums(tmp_path: Path) -> None:
    artifacts.publish_graph(tmp_path, copy_set)
    path = tmp_path / artifacts.STORE_NAME / artifacts.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["files"]["tram_graph.json"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(TramGraphDataError, match="manifest version mismatch"):
        artifacts.load_active_graph(tmp_path)


@pytest.mark.parametrize("commit_step", ["manifest", "store"])
def test_interrupted_first_publication_preserves_legacy_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commit_step: str
) -> None:
    copy_set(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in artifacts.ARTIFACT_NAMES}
    assert len(repository(tmp_path).load().stops) == 856
    rename = artifacts.os.rename

    def fail_commit(source: Path, target: Path) -> None:
        if commit_step == "manifest" or target == tmp_path / artifacts.STORE_NAME:
            raise OSError("first publication interrupted")
        rename(source, target)

    monkeypatch.setattr(
        artifacts.os, "replace" if commit_step == "manifest" else "rename", fail_commit
    )
    with pytest.raises(OSError, match="first publication interrupted"):
        artifacts.publish_graph(tmp_path, lambda directory: copy_set(directory, NEXT_TIME))

    assert not (tmp_path / artifacts.STORE_NAME).exists()
    assert len(repository(tmp_path).load().stops) == 856
    assert {name: (tmp_path / name).read_bytes() for name in artifacts.ARTIFACT_NAMES} == before


def test_deeply_nested_manifest_is_a_graph_data_error(tmp_path: Path) -> None:
    store = tmp_path / artifacts.STORE_NAME
    store.mkdir()
    (store / artifacts.MANIFEST_NAME).write_text('{"nested":' + "[" * 2000 + "0" + "]" * 2000 + "}")
    with pytest.raises(TramGraphDataError):
        repository(tmp_path).load()

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_tram_graph_repository
from app.infrastructure.db.session import get_session
from app.infrastructure.repositories.tram_graph import FileTramGraphRepository
from app.main import app


def test_liveness_does_not_require_database() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={"X-Request-ID": "health-check-1"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert response.headers["X-Request-ID"] == "health-check-1"


class _StubSession:
    """Stands in for the database so the graph half of readiness can be tested."""

    async def execute(self, _statement: object) -> None:
        return None


@contextmanager
def _overrides(**deps: object) -> Iterator[None]:
    for key, value in deps.items():
        app.dependency_overrides[_DEPENDENCIES[key]] = lambda v=value: v
    try:
        yield
    finally:
        for key in deps:
            app.dependency_overrides.pop(_DEPENDENCIES[key], None)


_DEPENDENCIES = {"session": get_session, "repository": get_tram_graph_repository}


def test_readiness_fails_when_the_graph_cannot_load(tmp_path: Path) -> None:
    """A deployment without its graph must report unready, not merely fail later.

    Readiness previously checked only the database, so a container missing the graph
    files looked healthy and then 500ed on whichever tram-graph request arrived first.
    """
    broken = FileTramGraphRepository(tmp_path / "absent.json", tmp_path / "absent.geojson")

    with _overrides(session=_StubSession(), repository=broken):
        with TestClient(app) as client:
            response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert "Tram graph is not ready" in response.json()["detail"]


def test_readiness_passes_with_the_committed_graph() -> None:
    with _overrides(session=_StubSession()):
        with TestClient(app) as client:
            response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.parametrize("content", ["null", "[]", '{"nodes": [{}], "links": []}'])
def test_readiness_reports_corrupt_graph_as_unavailable(tmp_path: Path, content: str) -> None:
    graph, geometry = tmp_path / "graph.json", tmp_path / "graph.geojson"
    graph.write_text(content)
    geometry.write_text('{"type": "FeatureCollection", "features": []}')
    repository = FileTramGraphRepository(graph, geometry)
    with _overrides(session=_StubSession(), repository=repository):
        with TestClient(app) as client:
            response = client.get("/api/v1/health/ready")
    assert response.status_code == 503
    assert "Tram graph is not ready" in response.json()["detail"]

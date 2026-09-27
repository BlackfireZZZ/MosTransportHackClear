"""File-backed graph port; graph assets are independent of PostgreSQL.

See docs/decisions/0003-tram-graph-file-repository.md.
"""

import asyncio
import json
from pathlib import Path

from app.domain.tram_graph import TramGraphDataError, TramNetwork
from app.infrastructure.graph_artifacts import STORE_NAME, load_active_graph
from app.infrastructure.tram_graph_validation import parse_network


class FileTramGraphRepository:
    """Cache the first valid network; a failed load can be retried after repair."""

    def __init__(self, graph_path: Path, geojson_path: Path) -> None:
        self._graph_path = graph_path
        self._geojson_path = geojson_path
        self._network: TramNetwork | None = None
        self._lock = asyncio.Lock()

    async def get_network(self) -> TramNetwork:
        if self._network is None:
            async with self._lock:
                if self._network is None:
                    self._network = await asyncio.to_thread(self.load)
        return self._network

    def load(self) -> TramNetwork:
        root = self._graph_path.parent
        if (
            self._geojson_path.parent == root
            and self._graph_path.name == "tram_graph.json"
            and self._geojson_path.name == "tram_graph.geojson"
            and ((root / STORE_NAME).exists() or (root / STORE_NAME).is_symlink())
        ):
            return parse_network(*load_active_graph(root))
        return parse_network(self._read_json(self._graph_path), self._read_json(self._geojson_path))

    @staticmethod
    def _read_json(path: Path) -> object:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise TramGraphDataError(f"cannot read the tram graph at {path}: {error}") from error
        try:
            return json.loads(content)
        except (ValueError, RecursionError) as error:
            raise TramGraphDataError(f"{path} is not valid JSON: {error}") from error

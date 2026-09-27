"""Exact ``event_id`` index in SQLite so memory stays bounded by chunk and cache size."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

Verdict = Literal["new", "duplicate", "conflict"]
LOOKUP_BATCH = 500
CACHE_KIB = 8192
SIDE_FILES = ("", "-wal", "-shm")


class DedupIndex:
    """Keep-first policy: a repeated id with the same payload hash is a duplicate,
    with a different hash a conflict. Rows carry the commit sequence of the chunk
    that inserted them so a resume can drop an uncheckpointed chunk."""

    def __init__(self, path: Path, fresh: bool) -> None:
        if fresh:
            self.remove(path)
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.execute(f"PRAGMA cache_size = -{CACHE_KIB}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS seen ("
            "stream TEXT NOT NULL, event_id TEXT NOT NULL, payload_hash BLOB NOT NULL, "
            "commit_seq INTEGER NOT NULL, PRIMARY KEY (stream, event_id)"
            ") WITHOUT ROWID"
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS seen_commit ON seen (commit_seq)")
        self._connection.execute("BEGIN")

    def rollback_after(self, commit_seq: int) -> None:
        self._connection.execute("DELETE FROM seen WHERE commit_seq > ?", (commit_seq,))
        self.commit()

    def classify(
        self, stream: str, candidates: Sequence[tuple[str, bytes]], commit_seq: int
    ) -> list[Verdict]:
        known = self._lookup(stream, [event_id for event_id, _ in candidates])
        verdicts: list[Verdict] = []
        inserts: list[tuple[str, str, bytes, int]] = []
        for event_id, payload_hash in candidates:
            existing = known.get(event_id)
            if existing is None:
                known[event_id] = payload_hash
                inserts.append((stream, event_id, payload_hash, commit_seq))
                verdicts.append("new")
            else:
                verdicts.append("duplicate" if existing == payload_hash else "conflict")
        self._connection.executemany("INSERT INTO seen VALUES (?, ?, ?, ?)", inserts)
        return verdicts

    def _lookup(self, stream: str, event_ids: Sequence[str]) -> dict[str, bytes]:
        unique = list(dict.fromkeys(event_ids))
        found: dict[str, bytes] = {}
        for start in range(0, len(unique), LOOKUP_BATCH):
            batch = unique[start : start + LOOKUP_BATCH]
            placeholders = ",".join("?" * len(batch))
            rows = self._connection.execute(
                f"SELECT event_id, payload_hash FROM seen "
                f"WHERE stream = ? AND event_id IN ({placeholders})",
                [stream, *batch],
            )
            for event_id, payload_hash in rows:
                found[event_id] = payload_hash
        return found

    def commit(self) -> None:
        self._connection.execute("COMMIT")
        self._connection.execute("BEGIN")

    def close(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")
        self._connection.close()

    @staticmethod
    def remove(path: Path) -> None:
        for suffix in SIDE_FILES:
            path.with_name(path.name + suffix).unlink(missing_ok=True)

"""Atomic checkpoint files and output writers that can resume at a verified prefix."""

import hashlib
import json
import os
from pathlib import Path
from typing import cast

from tramflow_ml.ingestion.records import (
    CHECKPOINT_SCHEMA,
    HASH_BLOCK_BYTES,
    STREAMS,
    Checkpoint,
    FileInventory,
    IngestionError,
)

REQUIRED_CHECKPOINT_KEYS: dict[str, type] = {
    "normalization_version": str,
    "adapter": str,
    "source_format": str,
    "chunk_size": int,
    "commit_seq": int,
    "inputs": dict,
    "streams": dict,
    "outputs": dict,
}
STREAM_COUNT_KEYS = ("input_rows", "valid", "duplicates", "quarantined")


def write_atomic(path: Path, payload: bytes) -> None:
    """Readers see either the previous file or the complete new one, never a partial."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path) -> Checkpoint:
    try:
        payload = json.loads(path.read_bytes())
    except ValueError as error:
        raise IngestionError(f"{path.name} is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise IngestionError(f"{path.name} is not a {CHECKPOINT_SCHEMA} checkpoint")
    malformed = sorted(
        key
        for key, expected in REQUIRED_CHECKPOINT_KEYS.items()
        if not isinstance(payload.get(key), expected) or isinstance(payload.get(key), bool)
    )
    streams = payload.get("streams", {})
    malformed += sorted(
        f"streams.{name}"
        for name in STREAMS
        if not _is_stream_state(streams.get(name) if isinstance(streams, dict) else None)
    )
    if malformed:
        raise IngestionError(f"{path.name} is missing or mistyped: {malformed}")
    return cast(Checkpoint, payload)


def _is_stream_state(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    counts = value.get("counts")
    scalars = all(
        isinstance(value.get(key), int) and not isinstance(value.get(key), bool)
        for key in ("offset", "row_index", "chunks")
    )
    return (
        scalars
        and isinstance(counts, dict)
        and all(isinstance(counts.get(key), int) for key in STREAM_COUNT_KEYS)
        and isinstance(value.get("reasons"), dict)
    )


class OutputWriter:
    """Append-only writer with a running SHA-256 over everything written."""

    def __init__(self, path: Path, resume: FileInventory | None) -> None:
        self.path = path
        self._digest = hashlib.sha256()
        self._size = 0
        if resume is None:
            self._stream = path.open("wb")
            return
        self._replay_prefix(resume)
        with path.open("r+b") as stream:
            stream.truncate(resume["bytes"])
        self._stream = path.open("ab")

    def _replay_prefix(self, expected: FileInventory) -> None:
        if not self.path.is_file():
            raise IngestionError(f"{self.path.name} is missing but the checkpoint records it")
        remaining = expected["bytes"]
        with self.path.open("rb") as stream:
            while remaining > 0:
                block = stream.read(min(HASH_BLOCK_BYTES, remaining))
                if not block:
                    raise IngestionError(f"{self.path.name} is shorter than the checkpoint")
                self._digest.update(block)
                remaining -= len(block)
        if self._digest.hexdigest() != expected["sha256"]:
            raise IngestionError(f"{self.path.name} prefix does not match the checkpoint")
        self._size = expected["bytes"]

    def write(self, content: bytes) -> None:
        self._stream.write(content)
        self._digest.update(content)
        self._size += len(content)

    def flush(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def inventory(self) -> FileInventory:
        return {"sha256": self._digest.hexdigest(), "bytes": self._size}

    def close(self) -> None:
        self._stream.close()

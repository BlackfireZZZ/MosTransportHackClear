"""Types, constants and errors shared by the ingestion package."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict
from zoneinfo import ZoneInfo

NORMALIZATION_VERSION = "normalization.v1"
CHECKPOINT_SCHEMA = "ingestion-checkpoint.v1"
MANIFEST_SCHEMA = "ingestion.v1"
DEFAULT_CHUNK_SIZE = 10_000
HASH_BLOCK_BYTES = 1 << 20
MOSCOW = ZoneInfo("Europe/Moscow")

StreamName = Literal["validations", "telemetry"]
STREAMS: tuple[StreamName, ...] = ("validations", "telemetry")
SourceFormat = Literal["jsonl", "csv"]

ENTITIES_NAME = "entities.json"
SOURCE_MANIFEST_NAME = "manifest.json"
QUARANTINE_NAME = "quarantine.jsonl"
CHECKPOINT_NAME = "checkpoint.json"
MANIFEST_NAME = "manifest.json"
DEDUP_NAME = "dedup.sqlite"
OUTPUT_NAMES: tuple[str, ...] = ("validations.jsonl", "telemetry.jsonl", QUARANTINE_NAME)
RUN_FILES: frozenset[str] = frozenset(
    (
        *OUTPUT_NAMES,
        CHECKPOINT_NAME,
        MANIFEST_NAME,
        DEDUP_NAME,
        DEDUP_NAME + "-wal",
        DEDUP_NAME + "-shm",
    )
)

Reason = Literal[
    "decode_error",
    "missing_field",
    "invalid_type",
    "invalid_value",
    "invalid_timestamp",
    "availability_before_event",
    "unknown_entity",
    "conflicting_duplicate",
]


class IngestionError(Exception):
    """Configuration, input or run-directory problem that stops the run."""


class InputChangedError(IngestionError):
    """A resumed run found input bytes that differ from the checkpoint."""


class ReconciliationError(IngestionError):
    """Counts do not satisfy input_rows == valid + duplicates + quarantined."""


class FileInventory(TypedDict):
    sha256: str
    bytes: int


class StreamCounts(TypedDict):
    input_rows: int
    valid: int
    duplicates: int
    quarantined: int


class StreamState(TypedDict):
    offset: int
    row_index: int
    chunks: int
    counts: StreamCounts
    reasons: dict[str, int]


class Checkpoint(TypedDict):
    schema_version: str
    normalization_version: str
    adapter: str
    source_format: str
    chunk_size: int
    commit_seq: int
    inputs: dict[str, FileInventory]
    streams: dict[str, StreamState]
    outputs: dict[str, FileInventory]


class StreamReport(TypedDict):
    input_file: str
    counts: StreamCounts
    quarantine_reasons: dict[str, int]
    reconciled: bool


class IngestionManifest(TypedDict):
    schema_version: str
    normalization_version: str
    adapter: str
    source_format: str
    chunk_size: int
    source_manifest: dict[str, object]
    inputs: dict[str, FileInventory]
    streams: dict[str, StreamReport]
    outputs: dict[str, FileInventory]
    output_hash: str


@dataclass(frozen=True, slots=True)
class RawRow:
    row_index: int
    raw: bytes


@dataclass(frozen=True, slots=True)
class Chunk:
    rows: tuple[RawRow, ...]
    end_offset: int


@dataclass(frozen=True, slots=True)
class Rejection:
    reason: Reason
    detail: str


@dataclass(frozen=True)
class ColumnAdapter:
    """Canonical field → source column; constants fill fields the source lacks.

    ``timestamp_format`` is a ``strptime`` pattern, or ``None`` for ISO 8601.
    Naive timestamps are accepted only when ``assume_timezone`` names a zone.
    """

    name: str
    columns: Mapping[str, str] = field(default_factory=dict)
    constants: Mapping[str, object] = field(default_factory=dict)
    timestamp_format: str | None = None
    assume_timezone: str | None = None

    def source_column(self, canonical: str) -> str:
        return self.columns.get(canonical, canonical)


SYNTHETIC_ADAPTER = ColumnAdapter(name="synthetic")
ADAPTERS: Mapping[str, ColumnAdapter] = {SYNTHETIC_ADAPTER.name: SYNTHETIC_ADAPTER}


@dataclass(frozen=True)
class IngestionConfig:
    input: Path
    output: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    source_format: SourceFormat = "jsonl"
    adapter: ColumnAdapter = SYNTHETIC_ADAPTER

    def __post_init__(self) -> None:
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        if self.source_format not in ("jsonl", "csv"):
            raise ValueError("source_format must be 'jsonl' or 'csv'")

    def stream_file(self, stream: StreamName) -> str:
        return f"{stream}.{self.source_format}"


def encode(value: object) -> bytes:
    """Canonical JSON line: sorted keys, compact separators, UTF-8, trailing newline."""
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")

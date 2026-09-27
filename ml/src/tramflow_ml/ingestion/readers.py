"""Streaming input hashing and chunked line readers that track byte offsets."""

import csv
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol

from tramflow_ml.ingestion.records import (
    HASH_BLOCK_BYTES,
    Chunk,
    FileInventory,
    IngestionError,
    RawRow,
    Rejection,
    SourceFormat,
)


def hash_file(path: Path) -> FileInventory:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK_BYTES):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def iter_chunks(path: Path, offset: int, row_index: int, chunk_size: int) -> Iterator[Chunk]:
    """Yield up to ``chunk_size`` physical lines per chunk, starting at ``offset``.

    ``end_offset`` is the byte position after the chunk's last line, so a resumed
    reader that seeks there continues with the next physical line.
    """
    with path.open("rb") as stream:
        stream.seek(offset)
        position = offset
        index = row_index
        rows: list[RawRow] = []
        for line in stream:
            position += len(line)
            rows.append(RawRow(index, line.rstrip(b"\r\n")))
            index += 1
            if len(rows) == chunk_size:
                yield Chunk(tuple(rows), position)
                rows = []
        if rows:
            yield Chunk(tuple(rows), position)


class LineDecoder(Protocol):
    def decode(self, raw: bytes) -> Mapping[str, object] | Rejection: ...


class JsonLineDecoder:
    def decode(self, raw: bytes) -> Mapping[str, object] | Rejection:
        try:
            value = json.loads(raw)
        except ValueError as error:
            return Rejection("decode_error", f"invalid JSON: {error}")
        if not isinstance(value, dict):
            return Rejection("decode_error", "JSON value is not an object")
        return value


class CsvLineDecoder:
    """One record per physical line; quoted line breaks are not supported.

    A UTF-8 byte-order mark before the header is dropped when the header is read.
    """

    def __init__(self, header: tuple[str, ...]) -> None:
        self.header = header

    def decode(self, raw: bytes) -> Mapping[str, object] | Rejection:
        try:
            fields = next(csv.reader([raw.decode("utf-8")]), [])
        except (UnicodeDecodeError, csv.Error) as error:
            return Rejection("decode_error", f"invalid CSV: {error}")
        if len(fields) != len(self.header):
            return Rejection(
                "decode_error", f"expected {len(self.header)} columns, got {len(fields)}"
            )
        return dict(zip(self.header, fields, strict=True))


def read_csv_header(path: Path) -> tuple[tuple[str, ...], int]:
    with path.open("rb") as stream:
        line = stream.readline()
    if not line:
        raise IngestionError(f"{path.name} has no CSV header line")
    try:
        header = next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]), [])
    except (UnicodeDecodeError, csv.Error) as error:
        raise IngestionError(f"{path.name}: unreadable CSV header") from error
    if not header or "" in header or len(set(header)) != len(header):
        raise IngestionError(f"{path.name}: CSV header needs unique non-empty names")
    return tuple(header), len(line)


def open_decoder(path: Path, source_format: SourceFormat) -> tuple[LineDecoder, int]:
    """Return the decoder and the byte offset where data rows start."""
    if source_format == "csv":
        header, data_start = read_csv_header(path)
        return CsvLineDecoder(header), data_start
    return JsonLineDecoder(), 0

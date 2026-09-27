"""One ingestion run: hash inputs, normalize in chunks, deduplicate, quarantine, checkpoint."""

import base64
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tramflow_ml.ingestion.checkpoint import OutputWriter, load_checkpoint, write_atomic
from tramflow_ml.ingestion.dedup import DedupIndex
from tramflow_ml.ingestion.normalize import CatalogIndex, Normalizer
from tramflow_ml.ingestion.readers import LineDecoder, hash_file, iter_chunks, open_decoder
from tramflow_ml.ingestion.records import (
    CHECKPOINT_NAME,
    CHECKPOINT_SCHEMA,
    DEDUP_NAME,
    ENTITIES_NAME,
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    NORMALIZATION_VERSION,
    RUN_FILES,
    SOURCE_MANIFEST_NAME,
    STREAMS,
    Checkpoint,
    Chunk,
    FileInventory,
    IngestionConfig,
    IngestionError,
    IngestionManifest,
    InputChangedError,
    RawRow,
    ReconciliationError,
    Rejection,
    StreamCounts,
    StreamName,
    StreamReport,
    StreamState,
    encode,
)

WRITER_NAMES = (*STREAMS, "quarantine")
CONFLICT = Rejection("conflicting_duplicate", "event_id already kept with a different payload")


@dataclass(frozen=True, slots=True)
class _Normalized:
    event_id: str
    encoded: bytes


@dataclass(frozen=True, slots=True)
class _ChunkOutcome:
    rows: int
    valid: int
    duplicates: int
    quarantined: int
    reasons: Mapping[str, int]


@dataclass(frozen=True)
class _Run:
    config: IngestionConfig
    catalog: CatalogIndex
    dedup: DedupIndex
    writers: Mapping[str, OutputWriter]


def commit_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    write_atomic(path, encode(checkpoint))


def _empty_stream() -> StreamState:
    return {
        "offset": 0,
        "row_index": 0,
        "chunks": 0,
        "counts": {"input_rows": 0, "valid": 0, "duplicates": 0, "quarantined": 0},
        "reasons": {},
    }


def _new_checkpoint(config: IngestionConfig, inputs: dict[str, FileInventory]) -> Checkpoint:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "normalization_version": NORMALIZATION_VERSION,
        "adapter": config.adapter.name,
        "source_format": config.source_format,
        "chunk_size": config.chunk_size,
        "commit_seq": 0,
        "inputs": inputs,
        "streams": {stream: _empty_stream() for stream in STREAMS},
        "outputs": {},
    }


def _input_inventory(config: IngestionConfig) -> dict[str, FileInventory]:
    names = [ENTITIES_NAME, SOURCE_MANIFEST_NAME, *(config.stream_file(s) for s in STREAMS)]
    missing = [name for name in names if not (config.input / name).is_file()]
    if missing:
        raise IngestionError(f"missing input files in {config.input}: {missing}")
    return {name: hash_file(config.input / name) for name in names}


def _load_source_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_bytes())
    except ValueError as error:
        raise IngestionError(f"{path.name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise IngestionError(f"{path.name} must be a JSON object")
    return payload


def _verified_resume(
    checkpoint: Checkpoint, config: IngestionConfig, inputs: dict[str, FileInventory]
) -> Checkpoint:
    changed = sorted(name for name in inputs if checkpoint["inputs"].get(name) != inputs[name])
    if changed or set(checkpoint["inputs"]) != set(inputs):
        raise InputChangedError(f"input files changed since the checkpoint: {changed}")
    expected: dict[str, object] = {
        "normalization_version": NORMALIZATION_VERSION,
        "adapter": config.adapter.name,
        "source_format": config.source_format,
        "chunk_size": config.chunk_size,
    }
    recorded: Mapping[str, object] = checkpoint
    mismatched = sorted(key for key, value in expected.items() if recorded.get(key) != value)
    if mismatched:
        raise IngestionError(f"checkpoint was written with different settings: {mismatched}")
    return checkpoint


def _prepare_run_dir(config: IngestionConfig) -> Checkpoint | None:
    """Return the checkpoint to resume from, or ``None`` for a fresh run."""
    output = config.output
    if output.resolve() == config.input.resolve():
        raise IngestionError("output directory must differ from the input directory")
    if (output / MANIFEST_NAME).exists():
        raise IngestionError(f"{output} already holds a completed ingestion manifest")
    output.mkdir(parents=True, exist_ok=True)
    allowed = RUN_FILES | {CHECKPOINT_NAME + ".tmp"}
    foreign = sorted(entry.name for entry in output.iterdir() if entry.name not in allowed)
    if foreign:
        raise IngestionError(f"{output} contains files that are not ingestion outputs: {foreign}")
    checkpoint_path = output / CHECKPOINT_NAME
    return load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None


def _normalize_row(
    decoder: LineDecoder, normalizer: Normalizer, row: RawRow
) -> _Normalized | Rejection:
    decoded = decoder.decode(row.raw)
    if isinstance(decoded, Rejection):
        return decoded
    normalized = normalizer.normalize(decoded)
    if isinstance(normalized, Rejection):
        return normalized
    return _Normalized(str(normalized["event_id"]), encode(normalized))


def _quarantine_record(stream: StreamName, row: RawRow, rejection: Rejection) -> dict[str, object]:
    """``raw`` is best-effort text; ``raw_base64`` carries the exact bytes only when
    the line is not valid UTF-8, so valid-input outputs keep their byte layout."""
    record: dict[str, object] = {
        "stream": stream,
        "row_index": row.row_index,
        "reason": rejection.reason,
        "detail": rejection.detail,
    }
    try:
        return {**record, "raw": row.raw.decode("utf-8")}
    except UnicodeDecodeError:
        return {
            **record,
            "raw": row.raw.decode("utf-8", errors="replace"),
            "raw_base64": base64.b64encode(row.raw).decode("ascii"),
        }


def _quarantine(run: _Run, stream: StreamName, row: RawRow, rejection: Rejection) -> None:
    run.writers["quarantine"].write(encode(_quarantine_record(stream, row, rejection)))


def _process_chunk(
    run: _Run,
    stream: StreamName,
    decoder: LineDecoder,
    normalizer: Normalizer,
    chunk: Chunk,
    commit_seq: int,
) -> _ChunkOutcome:
    results = [_normalize_row(decoder, normalizer, row) for row in chunk.rows]
    candidates = [
        (result.event_id, hashlib.sha256(result.encoded).digest())
        for result in results
        if isinstance(result, _Normalized)
    ]
    verdicts = iter(run.dedup.classify(stream, candidates, commit_seq))
    reasons: Counter[str] = Counter()
    valid = duplicates = 0
    for row, result in zip(chunk.rows, results, strict=True):
        if isinstance(result, Rejection):
            _quarantine(run, stream, row, result)
            reasons[result.reason] += 1
            continue
        verdict = next(verdicts)
        if verdict == "new":
            run.writers[stream].write(result.encoded)
            valid += 1
        elif verdict == "duplicate":
            duplicates += 1
        else:
            _quarantine(run, stream, row, CONFLICT)
            reasons[CONFLICT.reason] += 1
    return _ChunkOutcome(len(chunk.rows), valid, duplicates, sum(reasons.values()), reasons)


def _advance(state: StreamState, outcome: _ChunkOutcome, chunk: Chunk) -> StreamState:
    counts = state["counts"]
    merged: StreamCounts = {
        "input_rows": counts["input_rows"] + outcome.rows,
        "valid": counts["valid"] + outcome.valid,
        "duplicates": counts["duplicates"] + outcome.duplicates,
        "quarantined": counts["quarantined"] + outcome.quarantined,
    }
    reasons = Counter(state["reasons"]) + Counter(outcome.reasons)
    return {
        "offset": chunk.end_offset,
        "row_index": chunk.rows[-1].row_index + 1,
        "chunks": state["chunks"] + 1,
        "counts": merged,
        "reasons": dict(sorted(reasons.items())),
    }


def _commit_chunk(
    run: _Run, stream: StreamName, checkpoint: Checkpoint, outcome: _ChunkOutcome, chunk: Chunk
) -> Checkpoint:
    """Outputs become durable before the index commits and the index before the
    checkpoint, so a resume can always roll the index back to the checkpoint."""
    for writer in run.writers.values():
        writer.flush()
    run.dedup.commit()
    advanced: Checkpoint = {
        **checkpoint,
        "commit_seq": checkpoint["commit_seq"] + 1,
        "streams": {
            **checkpoint["streams"],
            stream: _advance(checkpoint["streams"][stream], outcome, chunk),
        },
        "outputs": {f"{name}.jsonl": writer.inventory() for name, writer in run.writers.items()},
    }
    commit_checkpoint(run.config.output / CHECKPOINT_NAME, advanced)
    return advanced


def _process_stream(run: _Run, stream: StreamName, checkpoint: Checkpoint) -> Checkpoint:
    """A finished stream resumes at its end offset, so re-scanning it yields no chunk."""
    state = checkpoint["streams"][stream]
    path = run.config.input / run.config.stream_file(stream)
    decoder, data_start = open_decoder(path, run.config.source_format)
    normalizer = Normalizer(
        run.config.adapter, run.catalog, stream, run.config.source_format == "csv"
    )
    offset = state["offset"] if state["chunks"] else data_start
    current = checkpoint
    for chunk in iter_chunks(path, offset, state["row_index"], run.config.chunk_size):
        outcome = _process_chunk(run, stream, decoder, normalizer, chunk, current["commit_seq"] + 1)
        current = _commit_chunk(run, stream, current, outcome, chunk)
    return current


def _report(state: StreamState, input_file: str) -> StreamReport:
    counts = state["counts"]
    accounted = counts["valid"] + counts["duplicates"] + counts["quarantined"]
    return {
        "input_file": input_file,
        "counts": counts,
        "quarantine_reasons": state["reasons"],
        "reconciled": counts["input_rows"] == accounted,
    }


def _build_manifest(
    run: _Run, checkpoint: Checkpoint, source_manifest: dict[str, object]
) -> IngestionManifest:
    outputs = {f"{name}.jsonl": writer.inventory() for name, writer in run.writers.items()}
    return {
        "schema_version": MANIFEST_SCHEMA,
        "normalization_version": NORMALIZATION_VERSION,
        "adapter": run.config.adapter.name,
        "source_format": run.config.source_format,
        "chunk_size": run.config.chunk_size,
        "source_manifest": source_manifest,
        "inputs": checkpoint["inputs"],
        "streams": {
            stream: _report(checkpoint["streams"][stream], run.config.stream_file(stream))
            for stream in STREAMS
        },
        "outputs": outputs,
        "output_hash": hashlib.sha256(encode(outputs)).hexdigest(),
    }


def _reconcile(manifest: IngestionManifest) -> None:
    failed = {
        stream: report["counts"]
        for stream, report in manifest["streams"].items()
        if not report["reconciled"]
    }
    if failed:
        raise ReconciliationError(f"input_rows != valid + duplicates + quarantined: {failed}")


def _open_writers(config: IngestionConfig, checkpoint: Checkpoint) -> dict[str, OutputWriter]:
    writers: dict[str, OutputWriter] = {}
    try:
        for name in WRITER_NAMES:
            resume = checkpoint["outputs"].get(f"{name}.jsonl")
            writers[name] = OutputWriter(config.output / f"{name}.jsonl", resume)
    except Exception:
        for writer in writers.values():
            writer.close()
        raise
    return writers


def ingest(config: IngestionConfig) -> IngestionManifest:
    """Run or resume an ingestion; ``manifest.json`` exists only after reconciliation.

    A run directory with ``checkpoint.json`` resumes from the last committed chunk
    after re-verifying input hashes and settings. Without a checkpoint the known
    output names are recreated. The checkpoint and the SQLite index are removed
    once the manifest is in place.
    """
    inputs = _input_inventory(config)
    source_manifest = _load_source_manifest(config.input / SOURCE_MANIFEST_NAME)
    catalog = CatalogIndex.load(config.input / ENTITIES_NAME)
    existing = _prepare_run_dir(config)
    checkpoint = (
        _new_checkpoint(config, inputs)
        if existing is None
        else _verified_resume(existing, config, inputs)
    )
    writers = _open_writers(config, checkpoint)
    dedup = DedupIndex(config.output / DEDUP_NAME, fresh=existing is None)
    run = _Run(config, catalog, dedup, writers)
    try:
        if existing is not None:
            dedup.rollback_after(checkpoint["commit_seq"])
        for stream in STREAMS:
            checkpoint = _process_stream(run, stream, checkpoint)
        manifest = _build_manifest(run, checkpoint, source_manifest)
    finally:
        dedup.close()
        for writer in writers.values():
            writer.close()
    _reconcile(manifest)
    write_atomic(config.output / MANIFEST_NAME, encode(manifest))
    (config.output / CHECKPOINT_NAME).unlink(missing_ok=True)
    DedupIndex.remove(config.output / DEDUP_NAME)
    return manifest

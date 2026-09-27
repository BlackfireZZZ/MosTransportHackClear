"""Bounded, resumable historical ingestion of fixture event streams."""

from tramflow_ml.ingestion.pipeline import ingest
from tramflow_ml.ingestion.records import (
    ADAPTERS,
    DEFAULT_CHUNK_SIZE,
    NORMALIZATION_VERSION,
    ColumnAdapter,
    IngestionConfig,
    IngestionError,
    IngestionManifest,
    InputChangedError,
    ReconciliationError,
)

__all__ = [
    "ADAPTERS",
    "DEFAULT_CHUNK_SIZE",
    "NORMALIZATION_VERSION",
    "ColumnAdapter",
    "IngestionConfig",
    "IngestionError",
    "IngestionManifest",
    "InputChangedError",
    "ReconciliationError",
    "ingest",
]

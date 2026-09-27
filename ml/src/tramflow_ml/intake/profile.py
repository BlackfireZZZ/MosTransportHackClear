"""The operator-written mapping between a sample's columns and the canonical fields.

A profile is the only place a correspondence that is not an exact name match may be
declared. Intake never infers one, so this file is also what TASK-049/TASK-050 will
fill in for the organizer's real extract.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from tramflow_ml.ingestion.normalize import STREAM_FIELDS
from tramflow_ml.ingestion.records import (
    SYNTHETIC_ADAPTER,
    ColumnAdapter,
    SourceFormat,
    StreamName,
)
from tramflow_ml.intake.records import PROFILE_SCHEMA, IntakeError

STREAMS: tuple[StreamName, ...] = get_args(StreamName)
SOURCE_FORMATS: tuple[SourceFormat, ...] = get_args(SourceFormat)
CANONICAL_FIELDS: frozenset[str] = frozenset(
    field for fields in STREAM_FIELDS.values() for field in fields
)


@dataclass(frozen=True)
class IntakeProfile:
    """Which file holds each stream, how it is encoded, and how its columns map.

    ``has_header`` is a CSV declaration intake cannot make for itself: without it the
    first line is assumed to be a header, and for a headerless extract that both
    loses row one and mistakes its cells for column names.
    """

    name: str
    source_format: SourceFormat
    files: Mapping[StreamName, str]
    adapter: ColumnAdapter
    has_header: bool = True

    def file_for(self, stream: StreamName) -> str:
        return self.files[stream]


def builtin_profile(source_format: SourceFormat) -> IntakeProfile:
    """Our own fixture shape: canonical names, one file per stream.

    This is not a guess about anyone's data — every mapping in it is the identity.
    """
    return IntakeProfile(
        name="canonical",
        source_format=source_format,
        files={stream: f"{stream}.{source_format}" for stream in STREAMS},
        adapter=SYNTHETIC_ADAPTER,
    )


def load_profile(path: Path) -> IntakeProfile:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise IntakeError(f"{path.name}: not a readable JSON profile ({error})") from error
    if not isinstance(payload, dict):
        raise IntakeError(f"{path.name}: profile must be a JSON object")
    if payload.get("schema_version") != PROFILE_SCHEMA:
        raise IntakeError(f"{path.name}: profile schema_version must be {PROFILE_SCHEMA!r}")
    name = _text(path.name, payload, "name")
    return IntakeProfile(
        name=name,
        source_format=_format(path.name, payload.get("source_format")),
        files=_files(path.name, payload.get("files")),
        has_header=_flag(path.name, payload, "has_header"),
        adapter=ColumnAdapter(
            name=name,
            columns=_columns(path.name, payload.get("columns")),
            constants=_constants(path.name, payload.get("constants")),
            timestamp_format=_optional_text(path.name, payload, "timestamp_format"),
            assume_timezone=_optional_text(path.name, payload, "assume_timezone"),
        ),
    )


def profile_skeleton(
    source_format: SourceFormat,
    files: Mapping[str, str],
    mapped: Mapping[str, str],
    has_header: bool = True,
) -> dict[str, object]:
    """A fill-in profile: every canonical field intake could not map is ``null``.

    Mapped entries carry the printable label, not necessarily the raw column name; an
    operator editing their own profile already has the name they wrote.
    """
    skeleton: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA,
        "name": "REPLACE-WITH-A-NAME-FOR-THIS-SOURCE",
        "source_format": source_format,
        "files": dict(sorted(files.items())),
        "columns": {field: mapped.get(field) for field in sorted(CANONICAL_FIELDS)},
        "constants": {},
        "timestamp_format": None,
        "assume_timezone": None,
    }
    if source_format == "csv":
        skeleton["has_header"] = has_header
    return skeleton


def _text(name: str, payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise IntakeError(f"{name}: profile.{key} must be a non-empty string")
    return value


def _flag(name: str, payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key, True)
    if type(value) is not bool:
        raise IntakeError(f"{name}: profile.{key} must be true or false")
    return value


def _optional_text(name: str, payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise IntakeError(f"{name}: profile.{key} must be a non-empty string or null")
    return value


def _format(name: str, value: object) -> SourceFormat:
    if value not in SOURCE_FORMATS:
        raise IntakeError(f"{name}: profile.source_format must be one of {list(SOURCE_FORMATS)}")
    return "csv" if value == "csv" else "jsonl"


def _files(name: str, value: object) -> dict[StreamName, str]:
    if not isinstance(value, dict) or set(value) != set(STREAMS):
        raise IntakeError(f"{name}: profile.files must name exactly the streams {list(STREAMS)}")
    files: dict[StreamName, str] = {}
    for stream in STREAMS:
        entry = value[stream]
        if not isinstance(entry, str) or not entry or "/" in entry:
            raise IntakeError(f"{name}: profile.files.{stream} must be a bare file name")
        files[stream] = entry
    return files


def _columns(name: str, value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise IntakeError(f"{name}: profile.columns must be an object")
    columns = {field: column for field, column in value.items() if column is not None}
    _reject_unknown(name, "columns", columns)
    bad = sorted(f for f, c in columns.items() if not isinstance(c, str) or not c)
    if bad:
        raise IntakeError(f"{name}: profile.columns entries {bad} must be non-empty strings")
    return {field: str(column) for field, column in sorted(columns.items())}


def _constants(name: str, value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise IntakeError(f"{name}: profile.constants must be an object")
    _reject_unknown(name, "constants", value)
    return dict(sorted(value.items()))


def _reject_unknown(name: str, section: str, value: Mapping[str, object]) -> None:
    unknown = sorted(set(value) - CANONICAL_FIELDS)
    if unknown:
        raise IntakeError(
            f"{name}: profile.{section} names fields that are not canonical: {unknown}; "
            f"canonical fields are {sorted(CANONICAL_FIELDS)}"
        )

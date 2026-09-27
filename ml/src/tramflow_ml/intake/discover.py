"""Locate each stream's file, read its column names, and refuse an unmappable schema.

Only an exact canonical name auto-maps, which is identity rather than inference. Every
other correspondence has to be written into a profile, and the refusal below is what
tells the operator exactly which ones are still missing. Names discovered in the file
are printed through ``columns.ColumnRef``, never raw.
"""

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tramflow_ml.ingestion.normalize import STREAM_FIELDS
from tramflow_ml.ingestion.records import SourceFormat, StreamName
from tramflow_ml.intake.columns import (
    ColumnRef,
    build,
    describe,
    labels,
    positional_names,
)
from tramflow_ml.intake.profile import (
    STREAMS,
    IntakeProfile,
    builtin_profile,
    profile_skeleton,
)
from tramflow_ml.intake.records import FIELD_CONTRACTS, SchemaError

_NOTHING = "-"


@dataclass(frozen=True)
class StreamSchema:
    """One stream's file, its columns, and how they map onto canonical fields."""

    stream: StreamName
    file: str
    columns: tuple[ColumnRef, ...]
    data_start: int
    mapping: Mapping[str, str]
    constants: tuple[str, ...]
    unmapped: tuple[str, ...]
    unused: tuple[str, ...]
    never_seen: tuple[str, ...]

    @property
    def header(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def trusted(self) -> bool:
        """Whether the column line read as a header rather than as a first data row."""
        return not any(column.redacted for column in self.columns)

    def label_for(self, name: str) -> str:
        found = next((c.label for c in self.columns if c.name == name), None)
        return found if found is not None else describe(name, self.trusted)

    def to_dict(self) -> dict[str, object]:
        section: dict[str, object] = {
            "columns": {
                field: self.label_for(column)
                for field, column in sorted(self.mapping.items())
            },
            "constants": list(self.constants),
            "file": self.file,
            "unused_columns": list(self.unused),
        }
        if self.never_seen:
            section["declared_columns_never_seen"] = list(self.never_seen)
        return section


def resolve_profile(source: Path, profile: IntakeProfile | None) -> IntakeProfile:
    """Without a profile, pick the built-in shape whose extension the sample uses."""
    if profile is not None:
        return profile
    candidates = [
        fmt
        for fmt in ("jsonl", "csv")
        if all((source / f"{stream}.{fmt}").is_file() for stream in STREAMS)
    ]
    if len(candidates) != 1:
        raise SchemaError(_missing_files_message(source, candidates))
    return builtin_profile("csv" if candidates[0] == "csv" else "jsonl")


def discover(source: Path, profile: IntakeProfile) -> tuple[StreamSchema, ...]:
    schemas = tuple(_stream_schema(source, profile, stream) for stream in STREAMS)
    if any(schema.unmapped for schema in schemas):
        raise SchemaError(_unmapped_message(profile, schemas))
    return schemas


def _stream_schema(source: Path, profile: IntakeProfile, stream: StreamName) -> StreamSchema:
    name = profile.file_for(stream)
    path = source / name
    if not path.is_file():
        raise SchemaError(_absent_file_message(source, stream, name))
    columns, data_start = _columns(path, profile.source_format, profile.has_header)
    observed = {column.name for column in columns}
    trusted_columns = not any(column.redacted for column in columns)
    adapter = profile.adapter
    fields = STREAM_FIELDS[stream]
    declared = {field: adapter.columns[field] for field in fields if field in adapter.columns}
    _reject_absent_declarations(path, profile, declared, columns)
    mapping = {
        field: declared.get(field, adapter.source_column(field))
        for field in fields
        if field in declared or adapter.source_column(field) in observed
    }
    unresolved = tuple(field for field in fields if field not in mapping)
    constants = tuple(field for field in unresolved if field in adapter.constants)
    return StreamSchema(
        stream=stream,
        file=name,
        columns=columns,
        data_start=data_start,
        mapping=mapping,
        constants=constants,
        unmapped=tuple(field for field in unresolved if field not in constants),
        unused=tuple(sorted(c.label for c in columns if c.name not in set(mapping.values()))),
        never_seen=tuple(
            sorted(
                describe(column, trusted_columns)
                for column in declared.values()
                if column not in observed
            )
        ),
    )


def _reject_absent_declarations(
    path: Path,
    profile: IntakeProfile,
    declared: Mapping[str, str],
    columns: tuple[ColumnRef, ...],
) -> None:
    """A CSV header is the schema, so a declared column missing from it can never be read.

    A JSON Lines file has no schema, so a declared key no row carries is simply a field
    that is always missing, and the report says so rather than refusing.
    """
    if profile.source_format != "csv":
        return
    observed = {column.name for column in columns}
    trusted = not any(column.redacted for column in columns)
    absent = sorted(
        describe(column, trusted) for column in declared.values() if column not in observed
    )
    if absent:
        raise SchemaError(
            f"profile.columns names columns absent from {path.name}: {absent}\n\n"
            f"  columns found ({len(columns)}): {', '.join(labels(columns)) or _NOTHING}\n\n"
            "Correct the spelling, or set \"has_header\": false when the file has no header row."
        )


def _columns(
    path: Path, source_format: SourceFormat, has_header: bool
) -> tuple[tuple[ColumnRef, ...], int]:
    if source_format == "csv":
        return _csv_columns(path, has_header)
    return _json_columns(path), 0


def _csv_columns(path: Path, has_header: bool) -> tuple[tuple[ColumnRef, ...], int]:
    with path.open("rb") as stream:
        line = stream.readline()
    if not line:
        raise SchemaError(f"{path.name} is empty: it holds no lines at all.")
    try:
        cells = next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]), [])
    except (UnicodeDecodeError, csv.Error) as error:
        raise SchemaError(f"{path.name}: the first line is not readable CSV ({error})") from error
    if not cells:
        raise SchemaError(f"{path.name}: the first line holds no columns.")
    if not has_header:
        return build(positional_names(len(cells))), 0
    _reject_blank_names(path, cells)
    columns = build(cells)
    _reject_repeated_names(path, columns)
    return columns, len(line)


def _reject_blank_names(path: Path, cells: Sequence[str]) -> None:
    blank = [index + 1 for index, cell in enumerate(cells) if not cell.strip()]
    if blank:
        raise SchemaError(
            f"{path.name}: header columns at positions {blank} have empty names. "
            "Give every column a name, or set \"has_header\": false in a profile."
        )


def _reject_repeated_names(path: Path, columns: Sequence[ColumnRef]) -> None:
    seen: dict[str, ColumnRef] = {}
    for column in columns:
        first = seen.setdefault(column.name, column)
        if first is not column:
            raise SchemaError(
                f"{path.name}: the header repeats a column name at positions "
                f"{first.position} and {column.position} ({first.label}). "
                "Column names must be unique."
            )


def _json_columns(path: Path) -> tuple[ColumnRef, ...]:
    """Every key in the whole file. A field that first appears late is still a field."""
    if path.stat().st_size == 0:
        raise SchemaError(f"{path.name} is empty: it holds no rows at all.")
    keys: set[str] = set()
    with path.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                keys.update(str(key) for key in row)
    if not keys:
        raise SchemaError(
            f"{path.name}: no line in the file is a readable JSON object, so intake can "
            "see no columns at all. Check the encoding and whether the file really is "
            "JSON Lines, one object per line."
        )
    return build(sorted(keys))


def _missing_files_message(source: Path, candidates: Sequence[str]) -> str:
    present = sorted(entry.name for entry in source.iterdir())
    expected = [f"{stream}.jsonl" for stream in STREAMS] + [f"{stream}.csv" for stream in STREAMS]
    ambiguity = (
        "Both the .jsonl and the .csv spelling are complete, so which one is the sample "
        "is not intake's decision."
        if len(candidates) > 1
        else "No complete set of stream files is present under the built-in names."
    )
    skeleton = profile_skeleton("jsonl", {stream: "REPLACE-ME" for stream in STREAMS}, {})
    return (
        f"intake cannot locate the streams in {source.name!r}.\n\n"
        f"  files present ({len(present)}): {', '.join(present) or _NOTHING}\n"
        f"  built-in names, one full set of which is needed: {', '.join(expected)}\n\n"
        f"{ambiguity}\n"
        "Name the files yourself in a profile and rerun with --profile <file>:\n\n"
        f"{_render(skeleton)}"
    )


def _absent_file_message(source: Path, stream: StreamName, name: str) -> str:
    present = sorted(entry.name for entry in source.iterdir())
    return (
        f"intake cannot read the {stream!r} stream: the profile names {name!r} and "
        f"{source.name!r} does not hold it.\n\n"
        f"  files present ({len(present)}): {', '.join(present) or _NOTHING}\n\n"
        "Correct profile.files and rerun."
    )


def _unmapped_message(profile: IntakeProfile, schemas: Sequence[StreamSchema]) -> str:
    blocked = [schema for schema in schemas if schema.unmapped]
    files: dict[str, str] = {schema.stream: schema.file for schema in schemas}
    mapped = {
        field: schema.label_for(column)
        for schema in schemas
        for field, column in sorted(schema.mapping.items())
    }
    skeleton = profile_skeleton(profile.source_format, files, mapped, profile.has_header)
    return "\n".join(
        [
            f"intake cannot read this sample with the {profile.name!r} mapping: "
            f"{len(blocked)} of {len(schemas)} streams have unmapped required fields.",
            "",
            *[line for schema in blocked for line in _stream_block(schema)],
            *_headerless_hint(profile, blocked),
            "No column is mapped by name similarity, position or a synonym list: only an "
            "exact canonical name maps on its own. Column names are judged a line at a "
            "time: unless every cell of the line is word-like, the line is treated as data "
            "and every cell is shown by shape and position rather than quoted, because "
            "intake cannot tell a header from a first data row. Supply the mapping yourself "
            "and rerun with --profile <file>:",
            "",
            _render(skeleton),
            "",
            "Replace every null with the source column that carries that field, or move the "
            "field into \"constants\" when the source omits it and one value is right for "
            "every row. Certifying the organizer's real mapping is TASK-049/TASK-050, not "
            "this tool's job.",
        ]
    )


def _headerless_hint(profile: IntakeProfile, blocked: Sequence[StreamSchema]) -> list[str]:
    condemned = any(not schema.trusted for schema in blocked)
    if not condemned or profile.source_format != "csv" or not profile.has_header:
        return []
    return [
        "Not every cell of the first line is word-like, which is what a file with no "
        "header row looks like: row one's cells have been read as column names. Every "
        "cell of that line is therefore shown by shape, including any that happens to "
        "look like a name. Set \"has_header\": false in a profile to treat every line "
        "as data and address columns as column_1, column_2 and so on.",
        "",
    ]


def _stream_block(schema: StreamSchema) -> list[str]:
    width = max(len(field) for field in schema.unmapped)
    mapped = ", ".join(
        f"{field} <- {schema.label_for(column)}" for field, column in sorted(schema.mapping.items())
    )
    return [
        f"stream {schema.stream!r} (file {schema.file})",
        f"  columns found ({len(schema.columns)}): {', '.join(labels(schema.columns)) or _NOTHING}",
        f"  mapped ({len(schema.mapping)}): {mapped or _NOTHING}",
        f"  supplied as constants ({len(schema.constants)}): "
        f"{', '.join(schema.constants) or _NOTHING}",
        f"  unmapped, required ({len(schema.unmapped)}):",
        *[
            f"    {field.ljust(width)}  {FIELD_CONTRACTS[field]}"
            for field in sorted(schema.unmapped)
        ],
        f"  columns not used ({len(schema.unused)}): {', '.join(schema.unused) or _NOTHING}",
        "",
    ]


def _render(skeleton: Mapping[str, object]) -> str:
    return json.dumps(skeleton, indent=2, ensure_ascii=False, sort_keys=True)

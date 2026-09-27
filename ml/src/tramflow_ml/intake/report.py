"""One profiling run: read a sample, describe it, decide nothing about it.

Every file is opened read-only and nothing is written under the sample directory.
The report is a value; where it is stored is the caller's business.
"""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tramflow_ml.identity import (
    AlignmentConfig,
    CanonicalCatalog,
    CatalogError,
    Crosswalk,
    SourceClock,
    identity_crosswalk,
)
from tramflow_ml.ingestion.normalize import STREAM_FIELDS
from tramflow_ml.ingestion.readers import CsvLineDecoder, JsonLineDecoder, iter_chunks
from tramflow_ml.ingestion.records import (
    DEFAULT_CHUNK_SIZE,
    ENTITIES_NAME,
    ColumnAdapter,
    Rejection,
    StreamName,
)
from tramflow_ml.intake.discover import StreamSchema, discover, resolve_profile
from tramflow_ml.intake.fields import (
    Accumulator,
    TimestampProfile,
    as_flag,
    canonical_text,
    digest,
    is_missing,
    lookup,
    make_accumulator,
    measure,
    parse_instant,
)
from tramflow_ml.intake.join import JoinCoverage, JoinInput
from tramflow_ml.intake.profile import IntakeProfile
from tramflow_ml.intake.records import (
    IDENTITY_CROSSWALK,
    MEASURE_CONTRACTS,
    REPORT_SCHEMA,
    IntakeError,
    IntakeReport,
    InvariantError,
    StreamSection,
    classify,
    rate,
)
from tramflow_ml.intake.supportability import ObservedSpan, horizon_support, target_support

SPAN_STREAM: StreamName = "validations"
BOOLEAN_FIELD = "synthetic"
DEFAULT_SOURCE_TIMEZONE = "Europe/Moscow"


@dataclass(frozen=True)
class IntakeRequest:
    source: Path
    profile: IntakeProfile | None = None
    catalog: Path | None = None


def profile_sample(request: IntakeRequest) -> IntakeReport:
    if not request.source.is_dir():
        raise IntakeError(f"{request.source} is not a directory")
    profile = resolve_profile(request.source, request.profile)
    schemas = discover(request.source, profile)
    zone = _zone(profile.adapter)
    catalog = _catalog(request)
    context = _JoinContext.build(catalog, profile.adapter)
    profilers = {
        schema.stream: _profile_stream(request.source, profile, schema, zone, context)
        for schema in schemas
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "source": _source_section(profile, schemas, catalog, request),
        "streams": {name: profiler.section() for name, profiler in profilers.items()},
        "supportability": _supportability(profilers[SPAN_STREAM]),
    }


def render(report: IntakeReport) -> str:
    """Byte-stable: sorted keys, no wall clock, no absolute path."""
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@dataclass(frozen=True, slots=True)
class _JoinContext:
    catalog: CanonicalCatalog | None
    crosswalk: Crosswalk | None
    config: AlignmentConfig

    @classmethod
    def build(cls, catalog: CanonicalCatalog | None, adapter: ColumnAdapter) -> "_JoinContext":
        clock = SourceClock(timezone=adapter.assume_timezone or DEFAULT_SOURCE_TIMEZONE)
        config = AlignmentConfig(clocks={stream: clock for stream in STREAM_FIELDS})
        crosswalk = None if catalog is None else identity_crosswalk(catalog, IDENTITY_CROSSWALK)
        return cls(catalog, crosswalk, config)


@dataclass
class _StreamProfiler:
    schema: StreamSchema
    adapter: ColumnAdapter
    coerce: bool
    context: _JoinContext
    accumulators: Mapping[str, Accumulator]
    join: JoinCoverage
    rows: int = 0
    readable: int = 0
    unreadable: Counter[str] = field(default_factory=Counter)
    payloads: dict[bytes, bytes] = field(default_factory=dict)
    unkeyed: int = 0
    repeated: int = 0
    conflicting: int = 0

    def observe(self, row: Mapping[str, object]) -> None:
        self.readable += 1
        for name, accumulator in self.accumulators.items():
            raw = lookup(self.adapter, name, row)
            if not is_missing(raw):
                accumulator.observe(raw)
        values = self._canonical(row)
        self._count_duplicate(values)
        if self.context.catalog is not None and self.context.crosswalk is not None:
            self.join.observe(
                self.context.catalog, self.context.crosswalk, self.context.config, _join(values)
            )

    def reject(self, reason: str) -> None:
        self.unreadable[reason] += 1

    def span(self) -> ObservedSpan:
        stamps = self.accumulators["event_at"]
        assert isinstance(stamps, TimestampProfile)
        dates = sorted(stamps.dates)
        return ObservedSpan(
            first=dates[0] if dates else None,
            last=dates[-1] if dates else None,
            observed_dates=len(dates),
            first_instant=stamps.first,
            last_instant=stamps.last,
        )

    def section(self) -> StreamSection:
        self._reconcile()
        return {
            "rows": self.rows,
            "readable": self.readable,
            "unreadable": dict(sorted(self.unreadable.items())),
            "fields": {
                name: accumulator.summary(self.readable)
                for name, accumulator in self.accumulators.items()
            },
            "duplicates": self._duplicates(),
            "date_span": self.span().to_dict(),
            "join_coverage": self._join_coverage(),
        }

    def enumerated(self, name: str) -> dict[str, int]:
        values = self.accumulators[name].summary(self.readable)["values"]
        assert isinstance(values, dict)
        return {str(key): int(count) for key, count in values.items()}

    def _reconcile(self) -> None:
        keyed = self.unkeyed + len(self.payloads) + self.repeated + self.conflicting
        if self.rows != self.readable + sum(self.unreadable.values()):
            raise InvariantError(f"{self.schema.stream}: rows != readable + unreadable")
        if self.readable != keyed:
            raise InvariantError(
                f"{self.schema.stream}: readable != unkeyed + distinct + repeated + conflicting"
            )
        if self.context.catalog is not None and not self.join.reconciles(self.readable):
            raise InvariantError(f"{self.schema.stream}: join outcomes do not reconcile")

    def _canonical(self, row: Mapping[str, object]) -> dict[str, object | None]:
        """Encoding-independent values, so a digest of them survives a re-rendering."""
        values: dict[str, object | None] = {}
        for name in STREAM_FIELDS[self.schema.stream]:
            raw = lookup(self.adapter, name, row)
            values[name] = None if is_missing(raw) else self._normalize(name, raw)
        return values

    def _normalize(self, name: str, raw: object) -> object | None:
        kind = classify(name)
        if kind == "timestamp":
            stamps = self.accumulators[name]
            assert isinstance(stamps, TimestampProfile)
            instant = parse_instant(raw, stamps.pattern, stamps.zone)
            return instant.isoformat() if instant else None
        if kind == "measure":
            return measure(raw, self.coerce, MEASURE_CONTRACTS[name]).value
        if name == BOOLEAN_FIELD:
            return as_flag(raw, self.coerce)
        return canonical_text(raw)

    def _count_duplicate(self, values: Mapping[str, object | None]) -> None:
        event_id = values.get("event_id")
        if not isinstance(event_id, str):
            self.unkeyed += 1
            return
        key = digest(event_id)
        payload = digest(json.dumps(values, sort_keys=True, ensure_ascii=False))
        known = self.payloads.get(key)
        if known is None:
            self.payloads[key] = payload
        elif known == payload:
            self.repeated += 1
        else:
            self.conflicting += 1

    def _duplicates(self) -> dict[str, object]:
        return {
            "conflict_rate": rate(self.conflicting, self.readable),
            "conflicting_rows": self.conflicting,
            "distinct_keys": len(self.payloads),
            "key": "event_id",
            "repeat_rate": rate(self.repeated, self.readable),
            "repeated_rows": self.repeated,
            "unkeyed_rows": self.unkeyed,
        }

    def _join_coverage(self) -> dict[str, object]:
        if self.context.catalog is None:
            return {
                "measured": False,
                "reason": "no entity catalog: place entities.json beside the streams "
                "or pass --catalog <file>",
            }
        return {"measured": True, **self.join.to_dict()}


def _join(values: Mapping[str, object | None]) -> JoinInput:
    sequence = values.get("stop_sequence")
    return JoinInput(
        event_id=_text(values, "event_id"),
        event_at=_instant(values, "event_at"),
        available_at=_instant(values, "available_at"),
        route_id=_text(values, "route_id"),
        direction_id=_text(values, "direction_id"),
        stop_id=_text(values, "stop_id"),
        stop_sequence=sequence if isinstance(sequence, int) else None,
        vehicle_id=_text(values, "vehicle_id"),
    )


def _text(values: Mapping[str, object | None], name: str) -> str | None:
    value = values.get(name)
    return value if isinstance(value, str) else None


def _instant(values: Mapping[str, object | None], name: str) -> datetime | None:
    value = values.get(name)
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def _profile_stream(
    source: Path,
    profile: IntakeProfile,
    schema: StreamSchema,
    zone: ZoneInfo | None,
    context: _JoinContext,
) -> _StreamProfiler:
    adapter = profile.adapter
    profiler = _StreamProfiler(
        schema=schema,
        adapter=adapter,
        coerce=profile.source_format == "csv",
        context=context,
        accumulators={
            name: make_accumulator(name, adapter, zone, profile.source_format == "csv")
            for name in STREAM_FIELDS[schema.stream]
        },
        join=JoinCoverage(source_id=schema.stream),
    )
    path = source / schema.file
    decoder = (
        CsvLineDecoder(schema.header)
        if profile.source_format == "csv"
        else JsonLineDecoder()
    )
    for chunk in iter_chunks(path, schema.data_start, 0, DEFAULT_CHUNK_SIZE):
        for raw in chunk.rows:
            profiler.rows += 1
            decoded = decoder.decode(raw.raw)
            if isinstance(decoded, Rejection):
                profiler.reject(decoded.reason)
            else:
                profiler.observe(decoded)
    return profiler


def _zone(adapter: ColumnAdapter) -> ZoneInfo | None:
    if adapter.assume_timezone is None:
        return None
    try:
        return ZoneInfo(adapter.assume_timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise IntakeError(f"unknown assume_timezone {adapter.assume_timezone!r}") from error


def _catalog(request: IntakeRequest) -> CanonicalCatalog | None:
    path = request.catalog or request.source / ENTITIES_NAME
    if not path.is_file():
        if request.catalog is not None:
            raise IntakeError(f"{path} is not a readable entity catalog")
        return None
    try:
        return CanonicalCatalog.from_dict(json.loads(path.read_bytes()))
    except (ValueError, CatalogError) as error:
        raise IntakeError(f"{path.name}: not a data.v1 entity catalog ({error})") from error


def _source_section(
    profile: IntakeProfile,
    schemas: tuple[StreamSchema, ...],
    catalog: CanonicalCatalog | None,
    request: IntakeRequest,
) -> dict[str, object]:
    catalog_name = (request.catalog or request.source / ENTITIES_NAME).name
    return {
        "assume_timezone": profile.adapter.assume_timezone,
        "catalog": None
        if catalog is None
        else {"entity_version": catalog.entity_version, "file": catalog_name},
        "format": profile.source_format,
        "profile": profile.name,
        "streams": {schema.stream: schema.to_dict() for schema in schemas},
        "timestamp_format": profile.adapter.timestamp_format,
    }


def _supportability(profiler: _StreamProfiler) -> dict[str, object]:
    return {
        "horizons": horizon_support(profiler.span()),
        "span_stream": SPAN_STREAM,
        "targets": target_support(profiler.enumerated("target"), profiler.enumerated("unit")),
    }

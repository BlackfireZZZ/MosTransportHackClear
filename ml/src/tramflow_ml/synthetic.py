"""Deterministic synthetic fixtures; never a claim about observed Moscow demand."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import BinaryIO, NamedTuple, TypedDict
from zoneinfo import ZoneInfo

GENERATOR_VERSION = "synthetic.v1"
SOURCE_VERSION = "synthetic-source.v1"
ENTITY_VERSION = "synthetic-entities.v1"
MOSCOW = ZoneInfo("Europe/Moscow")
# Repetition is the 3:3:1:1 morning-peak/evening-peak/midday/night weighting the tests count on.
HOURS = (8, 8, 8, 18, 18, 18, 12, 3)
STOP_IDS: tuple[str, ...] = ("synthetic:stop:1", "synthetic:stop:2", "synthetic:stop:3")


class FileInventory(TypedDict):
    sha256: str
    bytes: int


class GenerationCounts(TypedDict):
    unique_validations: int
    duplicate_validations: int
    late_validations: int
    telemetry: int


class CellTotal(TypedDict):
    route_id: str
    direction_id: str
    stop_id: str
    date: str
    hour: int
    count: int


class GenerationReport(TypedDict):
    generator_version: str
    config: dict[str, object]
    counts: GenerationCounts
    synthetic: bool
    gap_dates: list[str]
    hour_totals: dict[str, int]
    cell_totals: list[CellTotal]


class DatasetManifest(TypedDict):
    schema_version: str
    dataset_id: str
    source_version: str
    source_hash: str
    date_from: str
    date_to: str
    timezone: str
    feature_version: str
    target: str
    unit: str
    entity_version: str
    calendar_version: str
    availability_policy: str
    synthetic: bool


class GenerationResult(TypedDict):
    manifest: DatasetManifest
    generation: GenerationReport
    files: dict[str, FileInventory]


class Stop(TypedDict):
    id: str
    name: str


class StopPattern(TypedDict):
    route_id: str
    direction_id: str
    stop_ids: list[str]


class Catalog(TypedDict):
    schema_version: str
    entity_version: str
    routes: list[str]
    stops: list[Stop]
    patterns: list[StopPattern]


_CellKey = tuple[str, str, str, str, int]


@dataclass(frozen=True)
class SyntheticConfig:
    start: date = date(2024, 1, 1)
    end: date = date(2026, 1, 1)
    events: int = 64
    seed: int = 42
    duplicate_every: int = 17
    late_every: int = 11
    telemetry_every: int = 5
    gap_every_days: int = 13

    def __post_init__(self) -> None:
        if type(self.start) is not date or type(self.end) is not date:
            raise ValueError("start and end must be dates")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        for name in (
            "events",
            "seed",
            "duplicate_every",
            "late_every",
            "telemetry_every",
            "gap_every_days",
        ):
            value = getattr(self, name)
            minimum = 1 if name in ("events", "telemetry_every") else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.gap_every_days == 1:
            raise ValueError("gap_every_days=1 leaves no source coverage")


def _encode(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


class _Writer:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, content: bytes) -> None:
        self.stream.write(content)
        self.digest.update(content)
        self.size += len(content)

    def inventory(self) -> FileInventory:
        return {"sha256": self.digest.hexdigest(), "bytes": self.size}


def _write_json(output: Path, name: str, value: object) -> FileInventory:
    with (output / name).open("xb") as stream:
        writer = _Writer(stream)
        writer.write(_encode(value))
        return writer.inventory()


def _catalog() -> Catalog:
    return {
        "schema_version": "data.v1",
        "entity_version": ENTITY_VERSION,
        "routes": ["synthetic:route:1", "synthetic:route:2"],
        "stops": [
            {"id": STOP_IDS[0], "name": "Тестовая площадь"},
            {"id": STOP_IDS[1], "name": "Тестовая площадь"},
            {"id": STOP_IDS[2], "name": "Тестовый парк"},
        ],
        "patterns": [
            StopPattern(
                route_id=f"synthetic:route:{route}",
                direction_id=f"synthetic:direction:{direction}",
                stop_ids=list(STOP_IDS if direction == 0 else reversed(STOP_IDS))
                + [STOP_IDS[0] if direction == 0 else STOP_IDS[-1]],
            )
            for route in (1, 2)
            for direction in (0, 1)
        ],
    }


def _salt(seed: int) -> int:
    return int.from_bytes(hashlib.sha256(str(seed).encode("ascii")).digest()[:8])


def _calendar_days(config: SyntheticConfig) -> tuple[list[date], list[str]]:
    covered_dates: list[date] = []
    gap_dates: list[str] = []
    for index in range((config.end - config.start).days):
        day = config.start + timedelta(days=index)
        if config.gap_every_days and (index + 1) % config.gap_every_days == 0:
            gap_dates.append(day.isoformat())
        else:
            covered_dates.append(day)
    return covered_dates, gap_dates


class _Event(NamedTuple):
    common: dict[str, object]
    cell: _CellKey
    hour: int
    late: bool
    stop_id: str


def _event(
    config: SyntheticConfig,
    catalog: Catalog,
    covered_dates: list[date],
    salt: int,
    index: int,
) -> _Event:
    day = covered_dates[index * (len(covered_dates) - 1) // max(config.events - 1, 1)]
    hour = HOURS[(index + salt) % len(HOURS)]
    event_at = datetime.combine(day, time(hour, (index + salt) % 60), MOSCOW)
    late = bool(config.late_every and (index + 1) % config.late_every == 0)
    available_at = event_at + (timedelta(days=1) if late else timedelta(minutes=1))
    pattern = catalog["patterns"][(index + salt) % len(catalog["patterns"])]
    sequence = (index // 4 + salt) % len(pattern["stop_ids"])
    stop_id = pattern["stop_ids"][sequence]
    common: dict[str, object] = {
        "schema_version": "data.v1",
        "entity_version": ENTITY_VERSION,
        "source_version": SOURCE_VERSION,
        "route_id": pattern["route_id"],
        "direction_id": pattern["direction_id"],
        "stop_id": stop_id,
        "stop_sequence": sequence,
        "synthetic": True,
        "vehicle_id": f"synthetic:vehicle:{(index + salt) % 8 + 1}",
        "event_at": event_at.isoformat(),
        "available_at": available_at.isoformat(),
    }
    cell = (pattern["route_id"], pattern["direction_id"], stop_id, day.isoformat(), hour)
    return _Event(common, cell, hour, late, stop_id)


def _validation_row(event: _Event, index: int) -> dict[str, object]:
    return {
        **event.common,
        "event_id": f"synthetic:validation:{index + 1}",
        "target": "synthetic_boardings",
        "unit": "event_count",
    }


def _telemetry_row(event: _Event, index: int) -> dict[str, object]:
    stop_index = STOP_IDS.index(event.stop_id)
    return {
        **event.common,
        "event_id": f"synthetic:telemetry:{index + 1}",
        "latitude": 55.75 + stop_index * 0.01,
        "longitude": 37.60 + stop_index * 0.01,
    }


class _EventStreams(NamedTuple):
    files: dict[str, FileInventory]
    counts: GenerationCounts
    cells: Counter[_CellKey]
    hour_totals: Counter[str]


def _write_events(
    output: Path,
    config: SyntheticConfig,
    catalog: Catalog,
    covered_dates: list[date],
    salt: int,
) -> _EventStreams:
    cells: Counter[_CellKey] = Counter()
    hour_totals: Counter[str] = Counter()
    duplicates = late = telemetry_rows = 0
    with (
        (output / "validations.jsonl").open("xb") as validation_stream,
        (output / "telemetry.jsonl").open("xb") as telemetry_stream,
    ):
        validations = _Writer(validation_stream)
        telemetry = _Writer(telemetry_stream)
        for index in range(config.events):
            event = _event(config, catalog, covered_dates, salt, index)
            encoded = _encode(_validation_row(event, index))
            validations.write(encoded)
            if config.duplicate_every and (index + 1) % config.duplicate_every == 0:
                validations.write(encoded)
                duplicates += 1
            late += int(event.late)
            if (index + 1) % config.telemetry_every == 0:
                telemetry.write(_encode(_telemetry_row(event, index)))
                telemetry_rows += 1
            hour_totals[f"{event.hour:02d}"] += 1
            cells[event.cell] += 1
    files = {
        "validations.jsonl": validations.inventory(),
        "telemetry.jsonl": telemetry.inventory(),
    }
    counts: GenerationCounts = {
        "unique_validations": config.events,
        "duplicate_validations": duplicates,
        "late_validations": late,
        "telemetry": telemetry_rows,
    }
    return _EventStreams(files, counts, cells, hour_totals)


def _build_generation(
    config: SyntheticConfig, gap_dates: list[str], streams: _EventStreams
) -> GenerationReport:
    return {
        "generator_version": GENERATOR_VERSION,
        "config": {
            **asdict(config),
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
        },
        "counts": streams.counts,
        "synthetic": True,
        "gap_dates": gap_dates,
        "hour_totals": dict(sorted(streams.hour_totals.items())),
        "cell_totals": [
            {
                "route_id": route,
                "direction_id": direction,
                "stop_id": stop,
                "date": day_string,
                "hour": hour,
                "count": count,
            }
            for (route, direction, stop, day_string, hour), count in sorted(streams.cells.items())
        ],
    }


def _build_manifest(config: SyntheticConfig, source_hash: str) -> DatasetManifest:
    return {
        "schema_version": "forecast.v1",
        "dataset_id": f"synthetic:{source_hash}",
        "source_version": SOURCE_VERSION,
        "source_hash": source_hash,
        "date_from": datetime.combine(config.start, time(), MOSCOW).isoformat(),
        "date_to": datetime.combine(config.end, time(), MOSCOW).isoformat(),
        "timezone": "Europe/Moscow",
        "feature_version": "raw.v1",
        "target": "synthetic_boardings",
        "unit": "event_count",
        "entity_version": ENTITY_VERSION,
        "calendar_version": "moscow-midnight.v1",
        "availability_policy": "event-and-availability.v1",
        "synthetic": True,
    }


def generate_dataset(config: SyntheticConfig, output: Path) -> GenerationResult:
    """Create a new directory; manifest.json exists only after all data files close.

    A directory holding a stray ``.manifest.json.tmp`` and no ``manifest.json`` is
    also a failed run. ``events`` counts unique validations. Duplicate and late
    injection periods are one-based. Summary counts exclude duplicates. Gap dates
    mean missing source coverage, while covered dates with no generated events have
    observed zero. Inventory hashes cover exact UTF-8 bytes, including terminating
    newlines.
    """
    covered_dates, gap_dates = _calendar_days(config)
    catalog = _catalog()
    output.mkdir(parents=True, exist_ok=False)
    entities = _write_json(output, "entities.json", catalog)
    streams = _write_events(output, config, catalog, covered_dates, _salt(config.seed))
    generation = _build_generation(config, gap_dates, streams)
    files = {
        "entities.json": entities,
        **streams.files,
        "generation.json": _write_json(output, "generation.json", generation),
    }
    source_hash = hashlib.sha256(_encode(files)).hexdigest()
    manifest = _build_manifest(config, source_hash)
    _write_json(output, ".manifest.json.tmp", manifest)
    (output / ".manifest.json.tmp").replace(output / "manifest.json")
    return {"manifest": manifest, "generation": generation, "files": files}

"""Types, constants and errors for the read-only intake profiler.

The field register, the identifier set and the value allowlists are imported from
``ingestion.normalize`` rather than restated: a second copy would drift from the
one the pipeline actually enforces.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict

from tramflow_ml.ingestion.normalize import (
    COORDINATE_LIMITS,
    IDENTIFIER_FIELDS,
    SCHEMA_VERSION,
    TARGETS,
    TIMESTAMP_FIELDS,
    UNIT,
)

REPORT_SCHEMA = "intake.v1"
PROFILE_SCHEMA = "intake-profile.v1"
IDENTITY_CROSSWALK = "identity"

MIN_COVERAGE_RATIO = 0.5
RATE_DECIMALS = 6
DIGEST_BYTES = 8
MAX_COLUMN_NAME_LENGTH = 64

Classification = Literal["identifier", "timestamp", "measure", "enumerated"]
FormatSignature = Literal["digits", "letters", "alnum", "alnum_punct", "other"]

ENUMERATED_FIELDS: Mapping[str, frozenset[str]] = {
    "schema_version": frozenset({SCHEMA_VERSION}),
    "target": TARGETS,
    "unit": frozenset({UNIT}),
    "synthetic": frozenset({"false", "true"}),
}
OTHER_VALUE = "other"
UNPARSED_KEY = "unparsed"
OUT_OF_CONTRACT_KEY = "out_of_contract"

TARGET_BUCKETS: Mapping[str, int] = {"day": 24, "month": 31, "year": 12}
"""Buckets one forecast period occupies. ``month`` uses the longest calendar month so
the verdict cannot flip on which month a sample happens to end in."""


@dataclass(frozen=True, slots=True)
class MeasureContract:
    """What ``ingestion.normalize`` will accept for a measure field.

    Intake reports a value outside these bounds instead of quarantining it, so the
    profile says in advance how many rows the pipeline is going to reject.
    """

    integral: bool
    minimum: float | None = None
    maximum: float | None = None

    def holds(self, value: float | int) -> bool:
        if self.integral and not isinstance(value, int):
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        return self.maximum is None or value <= self.maximum


MEASURE_CONTRACTS: Mapping[str, MeasureContract] = {
    "stop_sequence": MeasureContract(integral=True, minimum=0),
    "latitude": MeasureContract(
        integral=False,
        minimum=-COORDINATE_LIMITS["latitude"],
        maximum=COORDINATE_LIMITS["latitude"],
    ),
    "longitude": MeasureContract(
        integral=False,
        minimum=-COORDINATE_LIMITS["longitude"],
        maximum=COORDINATE_LIMITS["longitude"],
    ),
}

FIELD_CONTRACTS: Mapping[str, str] = {
    "schema_version": f"contract version of the row; must be {SCHEMA_VERSION}",
    "entity_version": "catalog version the route, direction and stop ids belong to",
    "source_version": "the source's own version of this extract",
    "event_id": "stable per-row key; repeated keys are counted as duplicates",
    "route_id": "route the vehicle was serving",
    "direction_id": "direction of travel; it is never inferred from the stop",
    "stop_id": "stop the event happened at",
    "stop_sequence": "zero-based visit index of that stop on the route pattern",
    "vehicle_id": "vehicle; used only to detect a mid-day route reassignment",
    "event_at": "when the event happened; needs an offset or an assume_timezone",
    "available_at": "when the row became visible to a forecaster; never before event_at",
    "synthetic": "whether the row is generated rather than observed",
    "target": f"what is counted; one of {sorted(TARGETS)}",
    "unit": f"unit of the count; only {UNIT!r} can be produced",
    "latitude": "WGS84 latitude of the telemetry fix",
    "longitude": "WGS84 longitude of the telemetry fix",
}


class IntakeError(ValueError):
    """Configuration or input problem that stops a profiling run."""


class SchemaError(IntakeError):
    """The sample cannot be read with the mapping in force.

    ``str`` is the operator-facing checklist: which columns exist, which canonical
    fields are required, which of them are unmapped and what to write down next.
    """


class InvariantError(IntakeError):
    """A reported count does not reconcile with the rows the profiler read."""


class StreamSection(TypedDict):
    rows: int
    readable: int
    unreadable: dict[str, int]
    fields: dict[str, dict[str, object]]
    duplicates: dict[str, object]
    date_span: dict[str, object]
    join_coverage: dict[str, object]


class IntakeReport(TypedDict):
    schema_version: str
    source: dict[str, object]
    streams: dict[str, StreamSection]
    supportability: dict[str, object]


def classify(field: str) -> Classification:
    """The role that decides what may be printed about a field."""
    if field in IDENTIFIER_FIELDS:
        return "identifier"
    if field in TIMESTAMP_FIELDS:
        return "timestamp"
    if field in ENUMERATED_FIELDS:
        return "enumerated"
    return "measure"


def rate(part: int, whole: int) -> float:
    return round(part / whole, RATE_DECIMALS) if whole else 0.0

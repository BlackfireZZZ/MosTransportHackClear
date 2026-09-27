"""Per-field accumulators. What a field's classification allows is what gets printed.

An identifier accumulator keeps truncated digests rather than values, so it is not
merely a promise that no identifier is emitted: the accumulator holds nothing that
could be.
"""

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from tramflow_ml.ingestion.records import MOSCOW, ColumnAdapter
from tramflow_ml.intake.records import (
    DIGEST_BYTES,
    ENUMERATED_FIELDS,
    MEASURE_CONTRACTS,
    OTHER_VALUE,
    OUT_OF_CONTRACT_KEY,
    UNPARSED_KEY,
    Classification,
    FormatSignature,
    MeasureContract,
    classify,
    rate,
)

MISSING: object = object()
_IDENTIFIER_PUNCTUATION = frozenset("_.:/-")
# Mirrors the private ``_TRUE_STRINGS``/``_FALSE_STRINGS`` of ``ingestion.normalize``.
# Importing a private name would couple the two modules harder than restating four
# literals, so these are deliberately a copy and are covered by the reconciliation test.
_TRUE_STRINGS = frozenset({"true", "1"})
_FALSE_STRINGS = frozenset({"false", "0"})


@dataclass(frozen=True, slots=True)
class Measurement:
    """A measure read from a row: canonical number, and whether the pipeline takes it."""

    value: float | int | None
    in_contract: bool


def lookup(adapter: ColumnAdapter, name: str, row: Mapping[str, object]) -> object:
    column = adapter.source_column(name)
    if column in row:
        return row[column]
    return adapter.constants.get(name, MISSING)


def is_missing(raw: object) -> bool:
    """One definition across encodings: absent, null, or a blank string.

    A CSV carries every column on every row, so without the blank case a CSV could
    never reconcile against a JSON Lines file that simply omits the key.
    """
    if raw is MISSING or raw is None:
        return True
    return isinstance(raw, str) and not raw.strip()


def canonical_text(raw: object) -> str:
    """One spelling per value across encodings; ``14.0`` and ``14`` are one identifier."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, bool):
        return str(raw).lower()
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw)


def parse_instant(raw: object, pattern: str | None, zone: ZoneInfo | None) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.strptime(raw, pattern) if pattern else datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if zone is None:
            return None
        parsed = parsed.replace(tzinfo=zone)
    try:
        return parsed.astimezone(MOSCOW)
    except OverflowError:
        return None


def measure(raw: object, coerce: bool, contract: MeasureContract) -> Measurement:
    """Canonical numeric form first, contract verdict second.

    An integer-contract field accepts ``0.0`` as ``0``: a CSV written from a
    float-typed column spells every integer that way, and the same fact must not
    read differently for having passed through pandas.
    """
    number = _number(raw, coerce)
    if number is None:
        return Measurement(None, False)
    if contract.integral and isinstance(number, float) and number.is_integer():
        number = int(number)
    return Measurement(number, contract.holds(number))


def _number(raw: object, coerce: bool) -> float | int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return raw
    if coerce and isinstance(raw, str):
        try:
            return int(raw) if raw.lstrip("-").isdigit() else float(raw)
        except ValueError:
            return None
    return None


def as_flag(raw: object, coerce: bool) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if coerce and isinstance(raw, str):
        lowered = raw.lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return None


def signature(text: str) -> FormatSignature:
    if text.isdigit():
        return "digits"
    if text.isalpha():
        return "letters"
    if text.isalnum():
        return "alnum"
    if all(character.isalnum() or character in _IDENTIFIER_PUNCTUATION for character in text):
        return "alnum_punct"
    return "other"


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()[:DIGEST_BYTES]


class Accumulator(Protocol):
    classification: Classification

    def observe(self, raw: object) -> None: ...

    def summary(self, rows: int) -> dict[str, object]: ...


@dataclass
class IdentifierProfile:
    """Cardinality, null rate and shape. Never a value, a sample or a frequency."""

    classification: Classification = "identifier"
    present: int = 0
    min_length: int | None = None
    max_length: int | None = None
    digests: set[bytes] = field(default_factory=set)
    signatures: Counter[str] = field(default_factory=Counter)

    def observe(self, raw: object) -> None:
        text = canonical_text(raw)
        self.present += 1
        length = len(text)
        self.min_length = length if self.min_length is None else min(self.min_length, length)
        self.max_length = length if self.max_length is None else max(self.max_length, length)
        self.digests.add(digest(text))
        self.signatures[signature(text)] += 1

    def summary(self, rows: int) -> dict[str, object]:
        return {
            **_presence(self.classification, self.present, rows),
            "distinct_values": len(self.digests),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "format_signatures": dict(sorted(self.signatures.items())),
        }


@dataclass
class TimestampProfile:
    classification: Classification = "timestamp"
    pattern: str | None = None
    zone: ZoneInfo | None = None
    present: int = 0
    unparsed: int = 0
    dates: set[date] = field(default_factory=set)
    first: datetime | None = None
    last: datetime | None = None

    def observe(self, raw: object) -> None:
        self.present += 1
        instant = parse_instant(raw, self.pattern, self.zone)
        if instant is None:
            self.unparsed += 1
            return
        self.dates.add(instant.date())
        self.first = instant if self.first is None else min(self.first, instant)
        self.last = instant if self.last is None else max(self.last, instant)

    def summary(self, rows: int) -> dict[str, object]:
        return {
            **_presence(self.classification, self.present, rows),
            UNPARSED_KEY: self.unparsed,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
        }


@dataclass
class MeasureProfile:
    """Range and sum, plus how many values ``ingest`` would refuse."""

    contract: MeasureContract
    coerce: bool = False
    classification: Classification = "measure"
    present: int = 0
    unparsed: int = 0
    out_of_contract: int = 0
    minimum: float | int | None = None
    maximum: float | int | None = None
    total: int = 0

    def observe(self, raw: object) -> None:
        self.present += 1
        reading = measure(raw, self.coerce, self.contract)
        if reading.value is None:
            self.unparsed += 1
            return
        value = reading.value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if not reading.in_contract:
            self.out_of_contract += 1
        elif isinstance(value, int):
            self.total += value

    def summary(self, rows: int) -> dict[str, object]:
        summable = self.contract.integral and not self.unparsed and not self.out_of_contract
        return {
            **_presence(self.classification, self.present, rows),
            UNPARSED_KEY: self.unparsed,
            OUT_OF_CONTRACT_KEY: self.out_of_contract,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sum": self.total if summable else None,
        }


@dataclass
class EnumeratedProfile:
    """Counts per allowlisted member; a value outside it is counted, never quoted."""

    allowed: frozenset[str] = frozenset()
    boolean: bool = False
    coerce: bool = False
    classification: Classification = "enumerated"
    present: int = 0
    counts: Counter[str] = field(default_factory=Counter)

    def observe(self, raw: object) -> None:
        self.present += 1
        self.counts[self._member(raw)] += 1

    def _member(self, raw: object) -> str:
        if self.boolean:
            flag = as_flag(raw, self.coerce)
            return OTHER_VALUE if flag is None else str(flag).lower()
        text = canonical_text(raw)
        return text if text in self.allowed else OTHER_VALUE

    def summary(self, rows: int) -> dict[str, object]:
        values = {member: self.counts.get(member, 0) for member in sorted(self.allowed)}
        return {
            **_presence(self.classification, self.present, rows),
            "values": {**values, OTHER_VALUE: self.counts.get(OTHER_VALUE, 0)},
        }


def make_accumulator(
    name: str, adapter: ColumnAdapter, zone: ZoneInfo | None, coerce: bool
) -> Accumulator:
    kind = classify(name)
    if kind == "identifier":
        return IdentifierProfile()
    if kind == "timestamp":
        return TimestampProfile(pattern=adapter.timestamp_format, zone=zone)
    if kind == "enumerated":
        return EnumeratedProfile(
            allowed=ENUMERATED_FIELDS[name], boolean=name == "synthetic", coerce=coerce
        )
    return MeasureProfile(contract=MEASURE_CONTRACTS[name], coerce=coerce)


def _presence(classification: Classification, present: int, rows: int) -> dict[str, object]:
    missing = rows - present
    return {
        "classification": classification,
        "present": present,
        "missing": missing,
        "missing_rate": rate(missing, rows),
    }

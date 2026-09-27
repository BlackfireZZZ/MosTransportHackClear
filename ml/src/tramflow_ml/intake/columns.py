"""Column names are untrusted input, because a headerless file puts data where a
header belongs.

The unit of trust is the **line**, not the cell. A per-cell test can only ever ask
"does this look like a name?", and a datum is perfectly capable of looking like one:
``MSK4276380155129043``, ``A4276380155129043`` and ``Ivanov`` are all valid
identifiers. No predicate can separate those from real column names, because the safe
set and the word-like set genuinely overlap.

What distinguishes a header is that *every* cell of it is word-like. So if any cell
fails, the whole line is data and every cell is rendered by position and shape —
including the ones that happen to look like names. Position is what keeps the checklist
actionable: the operator still learns which column needs mapping.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from tramflow_ml.intake.fields import signature
from tramflow_ml.intake.records import MAX_COLUMN_NAME_LENGTH

POSITIONAL_PREFIX = "column_"


@dataclass(frozen=True, slots=True)
class ColumnRef:
    """A source column: the real name for matching, a safe label for printing."""

    position: int
    name: str
    label: str

    @property
    def redacted(self) -> bool:
        return self.label != self.name


def printable(name: str) -> bool:
    """Whether this single name *could* be a column name.

    ``str.isidentifier`` is the conservative test: it accepts ``ticket_no`` and
    ``маршрут`` and rejects anything carrying a space, a separator or a leading digit.
    It is necessary but not sufficient — only :func:`trusted_line` decides.
    """
    return bool(name) and len(name) <= MAX_COLUMN_NAME_LENGTH and name.isidentifier()


def trusted_line(names: Sequence[str]) -> bool:
    """A header is a line whose every cell is word-like; one failure condemns the line."""
    return bool(names) and all(printable(name) for name in names)


def build(names: Sequence[str]) -> tuple[ColumnRef, ...]:
    """Refs for one stream's columns, judging the line as a whole."""
    trusted = trusted_line(names)
    return tuple(
        ColumnRef(index + 1, name, name if trusted else redaction(index + 1, name))
        for index, name in enumerate(names)
    )


def redaction(position: int, name: str) -> str:
    return f"<column {position}: {len(name)} {signature(name)}>"


def describe(name: str, trusted: bool = True) -> str:
    """A safe rendering for a name whose position is unknown, such as one a profile names."""
    if trusted and printable(name):
        return name
    return f"<declared column: {len(name)} {signature(name)}>"


def positional_names(count: int) -> tuple[str, ...]:
    """Names for a file declared to have no header; they are word-like by construction."""
    return tuple(f"{POSITIONAL_PREFIX}{index}" for index in range(1, count + 1))


def labels(columns: Sequence[ColumnRef]) -> list[str]:
    return [column.label for column in columns]

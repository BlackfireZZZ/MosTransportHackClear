"""Join coverage in the identity vocabulary, measured one row at a time.

No organizer crosswalk exists yet, so coverage is measured against the identity
crosswalk — every canonical id mapped to itself. The report says so, because a
coverage figure obtained that way is a statement about a sample already keyed by
catalog ids, not a certified join rate.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from tramflow_ml.identity import (
    AlignmentConfig,
    Ambiguous,
    CanonicalCatalog,
    Crosswalk,
    Matched,
    SourceEvent,
    Stale,
    Unmatched,
    align_event,
)
from tramflow_ml.intake.records import IDENTITY_CROSSWALK, rate

UNALIGNABLE_REASONS = ("event_at_unreadable", "event_id_missing")


@dataclass(frozen=True, slots=True)
class JoinInput:
    """The subset of a row that decides where it joins; nothing else is carried."""

    event_id: str | None
    event_at: datetime | None
    available_at: datetime | None
    route_id: str | None
    direction_id: str | None
    stop_id: str | None
    stop_sequence: int | None
    vehicle_id: str | None


@dataclass
class JoinCoverage:
    """Per-stream tally in the shape ``identity.StreamQuality`` reports."""

    source_id: str
    aligned: int = 0
    unalignable: Counter[str] = field(default_factory=Counter)
    kinds: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)
    ambiguous: int = 0
    stale: int = 0

    def observe(
        self,
        catalog: CanonicalCatalog,
        crosswalk: Crosswalk,
        config: AlignmentConfig,
        row: JoinInput,
    ) -> None:
        if row.event_id is None:
            self.unalignable["event_id_missing"] += 1
            return
        if row.event_at is None:
            self.unalignable["event_at_unreadable"] += 1
            return
        self.aligned += 1
        match align_event(catalog, crosswalk, config, _source_event(self.source_id, row)).match:
            case Matched(kind=kind):
                self.kinds[kind] += 1
            case Unmatched(reason=reason):
                self.reasons[reason] += 1
            case Ambiguous():
                self.ambiguous += 1
            case Stale():
                self.stale += 1

    def to_dict(self) -> dict[str, object]:
        matched = sum(self.kinds.values())
        unmatched = sum(self.reasons.values())
        return {
            "ambiguous": self.ambiguous,
            "crosswalk": IDENTITY_CROSSWALK,
            "matched": {"by_kind": dict(sorted(self.kinds.items())), "total": matched},
            "rates": {
                "ambiguous": rate(self.ambiguous, self.aligned),
                "matched": rate(matched, self.aligned),
                "stale": rate(self.stale, self.aligned),
                "unmatched": rate(unmatched, self.aligned),
            },
            "stale": self.stale,
            "total": self.aligned,
            "unalignable": {
                "by_reason": {
                    reason: self.unalignable.get(reason, 0) for reason in UNALIGNABLE_REASONS
                },
                "total": sum(self.unalignable.values()),
            },
            "unmatched": {"by_reason": dict(sorted(self.reasons.items())), "total": unmatched},
        }

    def reconciles(self, rows: int) -> bool:
        outcomes = sum(self.kinds.values()) + sum(self.reasons.values()) + self.ambiguous
        return (
            outcomes + self.stale == self.aligned
            and self.aligned + sum(self.unalignable.values()) == rows
        )


def _source_event(source_id: str, row: JoinInput) -> SourceEvent:
    assert row.event_id is not None and row.event_at is not None
    return SourceEvent(
        event_id=row.event_id,
        source_id=source_id,
        event_at=row.event_at,
        available_at=row.available_at,
        route_id=row.route_id,
        direction_id=row.direction_id,
        stop_id=row.stop_id,
        stop_sequence=row.stop_sequence,
        vehicle_id=row.vehicle_id,
    )

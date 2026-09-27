"""Per-stream join quality; counts by outcome, kind and reason always sum to the total."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tramflow_ml.identity.crosswalk import Crosswalk
from tramflow_ml.identity.types import (
    AlignedEvent,
    AlignedTime,
    Ambiguous,
    IdentityError,
    Matched,
    Stale,
    Unmatched,
)


@dataclass(frozen=True)
class StreamQuality:
    source_id: str
    total: int
    matched_by_kind: Mapping[str, int]
    unmatched_by_reason: Mapping[str, int]
    ambiguous: int
    stale: int
    service_day_shifted: int

    @property
    def matched(self) -> int:
        return sum(self.matched_by_kind.values())

    @property
    def unmatched(self) -> int:
        return sum(self.unmatched_by_reason.values())

    def rates(self) -> dict[str, float]:
        counts = {
            "ambiguous": self.ambiguous,
            "matched": self.matched,
            "stale": self.stale,
            "unmatched": self.unmatched,
        }
        return {outcome: count / self.total for outcome, count in counts.items()}

    def to_dict(self) -> dict[str, object]:
        """Top-level and nested keys are sorted so serialized output is byte-stable."""
        fields: dict[str, object] = {
            "source_id": self.source_id,
            "total": self.total,
            "matched": {
                "total": self.matched,
                "by_kind": dict(sorted(self.matched_by_kind.items())),
            },
            "unmatched": {
                "total": self.unmatched,
                "by_reason": dict(sorted(self.unmatched_by_reason.items())),
            },
            "ambiguous": self.ambiguous,
            "stale": self.stale,
            "service_day_shifted": self.service_day_shifted,
            "rates": self.rates(),
        }
        return dict(sorted(fields.items()))


@dataclass(frozen=True)
class QualityReport:
    entity_version: str
    crosswalk_version: str
    streams: tuple[StreamQuality, ...]

    def to_dict(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "entity_version": self.entity_version,
            "crosswalk_version": self.crosswalk_version,
            "streams": [stream.to_dict() for stream in self.streams],
        }
        return dict(sorted(fields.items()))


def quality_report(crosswalk: Crosswalk, events: Iterable[AlignedEvent]) -> QualityReport:
    """Streams are ordered by source id; events from another crosswalk version are rejected."""
    grouped: dict[str, list[AlignedEvent]] = {}
    for event in events:
        if (event.crosswalk_version, event.entity_version) != (
            crosswalk.crosswalk_version,
            crosswalk.entity_version,
        ):
            raise IdentityError(f"event {event.event_id!r} was aligned with another crosswalk")
        grouped.setdefault(event.source_id, []).append(event)
    return QualityReport(
        entity_version=crosswalk.entity_version,
        crosswalk_version=crosswalk.crosswalk_version,
        streams=tuple(_stream(source_id, items) for source_id, items in sorted(grouped.items())),
    )


def _stream(source_id: str, events: list[AlignedEvent]) -> StreamQuality:
    kinds: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    ambiguous = stale = 0
    for event in events:
        match event.match:
            case Matched(kind=kind):
                kinds[kind] += 1
            case Unmatched(reason=reason):
                reasons[reason] += 1
            case Ambiguous():
                ambiguous += 1
            case Stale():
                stale += 1
    return StreamQuality(
        source_id=source_id,
        total=len(events),
        matched_by_kind=MappingProxyType(dict(sorted(kinds.items()))),
        unmatched_by_reason=MappingProxyType(dict(sorted(reasons.items()))),
        ambiguous=ambiguous,
        stale=stale,
        service_day_shifted=sum(
            isinstance(event.time, AlignedTime) and event.time.service_day_shifted
            for event in events
        ),
    )

"""Which Moscow civil dates the source covers, and when each of them became available.

Absence of rows is not absence of demand, and a date the source has not published yet is
not a date with zero demand: both are missing. Coverage is an input — it is never derived
from the events, because the events are exactly what it has to qualify.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType

from tramflow_ml.features.periods import bucket_dates
from tramflow_ml.features.records import Bucket, FeatureError, require_aware


@dataclass(frozen=True, slots=True)
class CoverageCalendar:
    """Covered civil dates, each optionally carrying the instant its data arrived.

    A date in ``dates`` with no entry in ``published_at`` is treated as having always
    been available, which is what a source that states coverage without publication
    metadata means.
    """

    dates: frozenset[date]
    published_at: Mapping[date, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dates:
            raise FeatureError("coverage calendar must name at least one covered date")
        if any(type(day) is not date for day in self.dates):
            raise FeatureError("coverage calendar holds non-date entries")
        stamps: dict[date, datetime] = {}
        for day, instant in self.published_at.items():
            if type(day) is not date:
                raise FeatureError("publication instants must be keyed by civil date")
            if day not in self.dates:
                raise FeatureError(
                    f"{day.isoformat()} has a publication instant but is not a covered date"
                )
            stamps[day] = require_aware(instant, f"publication instant for {day.isoformat()}")
        object.__setattr__(self, "published_at", MappingProxyType(dict(sorted(stamps.items()))))

    @classmethod
    def from_dates(
        cls, days: Iterable[date], published_at: Mapping[date, datetime] | None = None
    ) -> "CoverageCalendar":
        return cls(frozenset(days), dict(published_at or {}))

    @classmethod
    def from_range(
        cls,
        start: date,
        end: date,
        gaps: Iterable[date] = (),
        published_at: Mapping[date, datetime] | None = None,
    ) -> "CoverageCalendar":
        """Half-open ``[start, end)`` of civil dates minus the source's stated gaps."""
        if end <= start:
            raise FeatureError("coverage range end must be after start")
        excluded = frozenset(gaps)
        span = (end - start).days
        dates = frozenset(start + timedelta(days=offset) for offset in range(span)) - excluded
        return cls(dates, dict(published_at or {}))

    def covers(self, day: date) -> bool:
        """Whether the source covers the date at all; publication is a separate question."""
        return day in self.dates

    def published_by(self, day: date, cutoff: datetime) -> bool:
        instant = self.published_at.get(day)
        return instant is None or instant.astimezone(UTC) <= cutoff.astimezone(UTC)

    def as_of(self, cutoff: datetime) -> "CoverageView":
        """The calendar a forecaster standing at ``cutoff`` actually has."""
        return CoverageView(self, require_aware(cutoff, "cutoff"))

    def complete(self) -> "CoverageView":
        """The calendar after everything arrived; the view labels are measured against."""
        return CoverageView(self, None)


@dataclass(frozen=True, slots=True)
class CoverageView:
    """One calendar seen either at a cutoff or in full.

    History and labels need different answers from the same statement of coverage, so
    they take different views of it rather than being handed different calendars.
    """

    calendar: CoverageCalendar
    cutoff: datetime | None

    def covers(self, day: date) -> bool:
        return self.calendar.covers(day)

    def available(self, day: date) -> bool:
        if not self.calendar.covers(day):
            return False
        return self.cutoff is None or self.calendar.published_by(day, self.cutoff)

    def units(self, bucket: Bucket) -> tuple[int, int]:
        """Available and total civil dates of a bucket; a month keeps its gap visible."""
        days = bucket_dates(bucket)
        return sum(1 for day in days if self.available(day)), len(days)

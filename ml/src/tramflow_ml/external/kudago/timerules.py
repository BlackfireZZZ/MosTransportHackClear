"""Resolution of a KudaGo ``dates`` record into normalised sessions.

Every rule in this module exists because the naive reading of the source is
wrong in a way that silently produces plausible numbers:

1. ``start``/``end`` are Unix seconds UTC and are converted to ``Europe/Moscow``,
   while ``start_date``/``start_time``/``end_date``/``end_time`` are already
   local. When the two disagree the record resolves to ``unresolved`` with
   ``time_conflict``; picking either side invents a fact.
2. ``end == start`` with no ``end_time`` is the source's encoding of *unknown
   duration*, not of a zero-length session.
3. ``end_date`` may be null while ``end_time`` is set. The end day is taken from
   a valid Unix end, which also settles midnight; without one the day is not
   invented and the session stays ``start_only``.
4. A calendar date with no clock time is ``date_only`` with both moments null.
   Substituting 00:00 would put a museum exhibition in the night buckets.
5. ``is_continuous``, ``is_startless``, ``is_endless``, ``schedules`` and
   ``use_place_schedule`` are preserved. A month-long range without hours does
   not become 24 hours a day, and unbounded repeats expand only inside the
   delivery window. A record lying entirely outside the window still yields one
   raw-backed row, flagged ``outside_collection_window``.
6. Repeats expand only when the schedule is unambiguous. A venue timetable that
   would have to be read out of free text leaves ``schedule_basis`` at
   ``unresolved``.
7. Intervals are half-open ``[start_at, end_at)``.
8. An unknown or open date is reported, never dropped; a negative interval is
   quarantined, never rounded to zero.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from tramflow_ml.external.kudago.records import MOSCOW, ScheduleBasis, TimePrecision

TIMERULES_VERSION = "kudago-timerules.v1"

PLAUSIBLE_FIRST_YEAR = 2000
PLAUSIBLE_LAST_YEAR = 2100
MAX_SINGLE_SESSION = timedelta(hours=24)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)

FLAG_TIME_CONFLICT = "time_conflict"
FLAG_OUTSIDE_WINDOW = "outside_collection_window"
FLAG_NO_IN_WINDOW_SESSION = "no_in_window_session"
FLAG_UNBOUNDED_START = "unbounded_start"
FLAG_UNBOUNDED_END = "unbounded_end"
FLAG_STARTLESS = "is_startless"
FLAG_ENDLESS = "is_endless"
FLAG_START_DAY_UNKNOWN = "start_day_unknown"
FLAG_START_TIME_UNKNOWN = "start_time_unknown"
FLAG_END_TIME_UNKNOWN = "end_time_unknown"
FLAG_END_DAY_UNKNOWN = "end_day_unknown"
FLAG_END_DAY_FROM_UNIX = "end_day_derived_from_unix"
FLAG_END_UNIX_UNUSED = "end_unix_unused"
FLAG_ZERO_LENGTH_SOURCE = "zero_length_source"
FLAG_ZERO_LENGTH_DROPPED = "zero_length_end_dropped"
FLAG_NO_CALENDAR_DATE = "no_calendar_date"
FLAG_AMBIGUOUS_MULTIDAY = "ambiguous_multiday_range"
FLAG_CONTINUOUS_SPAN = "continuous_span"
FLAG_PLACE_SCHEDULE_REQUIRED = "place_schedule_required"
FLAG_SCHEDULE_SHAPE_UNCONFIRMED = "schedule_shape_unconfirmed"
FLAG_SCHEDULE_WEEKDAYS_UNCONFIRMED = "schedule_weekdays_unconfirmed"
FLAG_SCHEDULE_CROSSES_MIDNIGHT = "schedule_crosses_midnight"

REASON_INVALID_DATE_RECORD = "invalid_date_record"
REASON_NEGATIVE_INTERVAL = "negative_interval"

_DATE_PATTERN = re.compile(r"\A(\d{4})-(\d{2})-(\d{2})\Z")
_TIME_PATTERN = re.compile(r"\A(\d{2}):(\d{2})(?::(\d{2}))?\Z")
_FULL_WEEK = frozenset(range(7))


class TimeRuleError(ValueError):
    """A date record contradicts the source's own format and cannot be read."""


@dataclass(frozen=True, slots=True)
class Window:
    """The delivery windows plus their buffers, each half-open ``[start, end)``.

    Disjoint months are separate spans rather than one hull. Collecting January
    and September fetches two months; treating them as ``[Jan, Oct)`` would put
    the eight unfetched months between them inside the window, so repeats would
    materialise there and the coverage denominator would count a period nobody
    collected.
    """

    spans: tuple[tuple[datetime, datetime], ...]

    def __post_init__(self) -> None:
        if not self.spans:
            raise TimeRuleError("a window needs at least one span")
        for start, end in self.spans:
            if start.tzinfo is None or end.tzinfo is None:
                raise TimeRuleError("window bounds must be timezone-aware")
            if end <= start:
                raise TimeRuleError("window end must be after window start")

    @property
    def start(self) -> datetime:
        """The earliest bound, for pinning an unbounded side outside every span."""
        return min(start for start, _ in self.spans)

    @property
    def end(self) -> datetime:
        return max(end for _, end in self.spans)

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return any(start < span_end and end > span_start for span_start, span_end in self.spans)

    def contains(self, moment: datetime) -> bool:
        return any(span_start <= moment < span_end for span_start, span_end in self.spans)


@dataclass(frozen=True, slots=True)
class ScheduleRule:
    """One repeat rule whose hours and day range are confirmed by the payload."""

    start_time: time
    end_time: time
    first_day: date | None
    last_day: date | None


@dataclass(frozen=True, slots=True)
class ParsedDate:
    """The raw date record after type and sentinel checks, before interpretation."""

    start_day: date | None
    end_day: date | None
    start_time: time | None
    end_time: time | None
    start_moment: datetime | None
    end_moment: datetime | None
    continuous: bool
    startless: bool
    endless: bool
    use_place_schedule: bool
    schedules: tuple[ScheduleRule, ...] | None
    flags: frozenset[str]


@dataclass(frozen=True, slots=True)
class Session:
    """One resolved session, or one row standing for what could not be resolved.

    ``start_date``/``end_date`` always carry the original record's bounds; the
    moment of this particular session lives in ``start_at``/``end_at``.
    """

    start_at: datetime | None
    end_at: datetime | None
    start_date: str | None
    end_date: str | None
    time_precision: TimePrecision
    schedule_basis: ScheduleBasis
    quality_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Rejection:
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class DateResolution:
    """Exactly one of the two is populated: a date record never vanishes."""

    sessions: tuple[Session, ...]
    rejection: Rejection | None


def parse_window(buffer_from: str, buffer_to: str) -> Window:
    """Build a single-span expansion window from two ISO bounds in ``Europe/Moscow``.

    A plain ``YYYY-MM-DD`` is that day's midnight, so ``buffer_to`` is exclusive:
    September 2025 with no buffer is ``2025-09-01`` to ``2025-10-01``.
    """
    return parse_windows(((buffer_from, buffer_to),))


def parse_windows(bounds: Sequence[tuple[str, str]]) -> Window:
    """Build a window from one ISO bound pair per collected period.

    Overlapping or touching spans are merged, so an accidental double-count is
    impossible and the span list is canonical for a given set of bounds.
    """
    parsed = sorted(
        (_parse_bound(start, "buffer_from"), _parse_bound(end, "buffer_to"))
        for start, end in bounds
    )
    if not parsed:
        raise TimeRuleError("a window needs at least one span")
    merged: list[tuple[datetime, datetime]] = [parsed[0]]
    for start, end in parsed[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return Window(tuple(merged))


def _parse_bound(text: str, key: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise TimeRuleError(f"{key}={text!r} is not an ISO 8601 date or datetime") from error
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=MOSCOW)


def resolve_date_record(raw: Mapping[str, object], window: Window) -> DateResolution:
    """Turn one ``dates`` entry into sessions, or into a single quarantine reason."""
    try:
        parsed = parse_date_record(raw)
    except TimeRuleError as error:
        return DateResolution((), Rejection(REASON_INVALID_DATE_RECORD, str(error)))
    flags = set(parsed.flags)
    basis = _schedule_basis(parsed, flags)
    if _conflicts(parsed):
        flags.add(FLAG_TIME_CONFLICT)
        return _finish(parsed, window, "unresolved", flags, ())
    if basis == "unresolved":
        return _finish(parsed, window, basis, flags, ())
    if basis == "event_schedule":
        return _expand_schedules(parsed, window, flags)
    return _resolve_explicit(parsed, window, flags)


def parse_date_record(raw: Mapping[str, object]) -> ParsedDate:
    """Read the documented fields, rejecting values the format cannot produce.

    Years outside 2000-2100 are the source's open-ended sentinels rather than
    real bounds, so they become an unbounded side with a flag, not a date.
    """
    start_day, start_open = _parse_day(_optional_text(raw, "start_date"), "start_date")
    end_day, end_open = _parse_day(_optional_text(raw, "end_date"), "end_date")
    start_moment, start_moment_open = _parse_moment(_optional_int(raw, "start"), "start")
    end_moment, end_moment_open = _parse_moment(_optional_int(raw, "end"), "end")
    startless = _optional_bool(raw, "is_startless")
    endless = _optional_bool(raw, "is_endless")
    flags: set[str] = set()
    if start_open or start_moment_open:
        flags.add(FLAG_UNBOUNDED_START)
    if end_open or end_moment_open:
        flags.add(FLAG_UNBOUNDED_END)
    if startless:
        flags.add(FLAG_STARTLESS)
    if endless:
        flags.add(FLAG_ENDLESS)
    schedules = _parse_schedules(raw, flags)
    return ParsedDate(
        start_day=start_day,
        end_day=end_day,
        start_time=_parse_clock(_optional_text(raw, "start_time"), "start_time"),
        end_time=_parse_clock(_optional_text(raw, "end_time"), "end_time"),
        start_moment=start_moment,
        end_moment=end_moment,
        continuous=_optional_bool(raw, "is_continuous"),
        startless=startless,
        endless=endless,
        use_place_schedule=_optional_bool(raw, "use_place_schedule"),
        schedules=schedules,
        flags=frozenset(flags),
    )


def _optional_text(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TimeRuleError(f"{key} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    return stripped or None


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimeRuleError(f"{key} must be an integer, got {type(value).__name__}")
    return value


def _optional_bool(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise TimeRuleError(f"{key} must be a boolean, got {type(value).__name__}")
    return value


def _parse_day(text: str | None, key: str) -> tuple[date | None, bool]:
    if text is None:
        return None, False
    match = _DATE_PATTERN.match(text)
    if match is None:
        raise TimeRuleError(f"{key}={text!r} is not YYYY-MM-DD")
    try:
        day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError as error:
        raise TimeRuleError(f"{key}={text!r} is not a real date") from error
    if not PLAUSIBLE_FIRST_YEAR <= day.year <= PLAUSIBLE_LAST_YEAR:
        return None, True
    return day, False


def _parse_clock(text: str | None, key: str) -> time | None:
    if text is None:
        return None
    match = _TIME_PATTERN.match(text)
    if match is None:
        raise TimeRuleError(f"{key}={text!r} is not HH:MM[:SS]")
    try:
        return time(int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
    except ValueError as error:
        raise TimeRuleError(f"{key}={text!r} is not a real time of day") from error


def _parse_moment(value: int | None, key: str) -> tuple[datetime | None, bool]:
    if value is None:
        return None, False
    try:
        moment = datetime.fromtimestamp(value, UTC).astimezone(MOSCOW)
    except (OSError, OverflowError, ValueError):
        return None, True
    if not PLAUSIBLE_FIRST_YEAR <= moment.year <= PLAUSIBLE_LAST_YEAR:
        return None, True
    return moment, False


def _parse_schedules(raw: Mapping[str, object], flags: set[str]) -> tuple[ScheduleRule, ...] | None:
    """Accept a repeat rule only when its weekday set cannot be misread.

    ``days_of_week`` is a list of integers whose week start the public
    documentation does not fix, so a partial week is refused rather than guessed:
    only an absent, empty or complete day set expands. Everything else leaves the
    record unresolved, which is rule 6.
    """
    value = raw.get("schedules")
    if value is None:
        return ()
    if not isinstance(value, list):
        flags.add(FLAG_SCHEDULE_SHAPE_UNCONFIRMED)
        return None
    rules: list[ScheduleRule] = []
    for entry in value:
        if not isinstance(entry, dict):
            flags.add(FLAG_SCHEDULE_SHAPE_UNCONFIRMED)
            return None
        rule = _parse_schedule_entry(entry, flags)
        if rule is None:
            return None
        rules.append(rule)
    return tuple(rules)


def _parse_schedule_entry(entry: Mapping[str, object], flags: set[str]) -> ScheduleRule | None:
    days = entry.get("days_of_week")
    if days is not None:
        if not isinstance(days, list) or not all(
            isinstance(day, int) and not isinstance(day, bool) for day in days
        ):
            flags.add(FLAG_SCHEDULE_SHAPE_UNCONFIRMED)
            return None
        if days and set(days) != _FULL_WEEK:
            flags.add(FLAG_SCHEDULE_WEEKDAYS_UNCONFIRMED)
            return None
    try:
        start_time = _parse_clock(_optional_text(entry, "start_time"), "schedules.start_time")
        end_time = _parse_clock(_optional_text(entry, "end_time"), "schedules.end_time")
        first_day, _ = _parse_day(_optional_text(entry, "start_date"), "schedules.start_date")
        last_day, _ = _parse_day(_optional_text(entry, "end_date"), "schedules.end_date")
    except TimeRuleError:
        flags.add(FLAG_SCHEDULE_SHAPE_UNCONFIRMED)
        return None
    if start_time is None or end_time is None or start_time == end_time:
        flags.add(FLAG_SCHEDULE_SHAPE_UNCONFIRMED)
        return None
    return ScheduleRule(start_time, end_time, first_day, last_day)


def _schedule_basis(parsed: ParsedDate, flags: set[str]) -> ScheduleBasis:
    if parsed.use_place_schedule:
        flags.add(FLAG_PLACE_SCHEDULE_REQUIRED)
        return "unresolved"
    if parsed.schedules is None:
        return "unresolved"
    if parsed.schedules:
        return "event_schedule"
    return "explicit"


def _conflicts(parsed: ParsedDate) -> bool:
    """Compare only the fields that would actually be used.

    The source encodes an unknown end as ``end == start`` and an absent
    ``end_time`` as midnight, so those are its format, not a disagreement.
    A Unix hour the local fields never mention is a disagreement, not a gap to
    fill: the card is either an all-day entry or a 19:00 one, and rule 1 refuses
    to pick. Live data never forces the question -- across 172644 date records an
    absent ``start_time`` always came with a midnight Unix start.
    """
    if parsed.start_moment is not None and parsed.start_day is not None:
        local = _combine(parsed.start_day, parsed.start_time or time())
        if local != parsed.start_moment:
            return True
    if parsed.end_time is None or parsed.end_moment is None:
        return False
    if parsed.end_day is not None:
        return _combine(parsed.end_day, parsed.end_time) != parsed.end_moment
    return parsed.end_moment.timetz().replace(tzinfo=None) != parsed.end_time


def _combine(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=MOSCOW)


def _ends_where_it_starts(parsed: ParsedDate) -> bool:
    return (
        parsed.end_time is None
        and parsed.end_moment is not None
        and parsed.start_moment is not None
        and parsed.end_moment == parsed.start_moment
    )


def _resolve_explicit(parsed: ParsedDate, window: Window, flags: set[str]) -> DateResolution:
    if parsed.start_time is None:
        if parsed.end_time is not None:
            flags.add(FLAG_START_TIME_UNKNOWN)
        return _finish(parsed, window, "explicit", flags, (_dateless(parsed, "explicit", flags),))
    if parsed.start_day is None:
        flags.add(FLAG_START_DAY_UNKNOWN)
        return _finish(parsed, window, "explicit", flags, ())
    start_at = _combine(parsed.start_day, parsed.start_time)
    end_at, rejection = _resolve_end(parsed, start_at, flags)
    if rejection is not None:
        return DateResolution((), rejection)
    if end_at is None:
        return _finish(
            parsed,
            window,
            "explicit",
            flags,
            (_session(parsed, start_at, None, "start_only", "explicit", flags),),
        )
    if end_at - start_at > MAX_SINGLE_SESSION and not parsed.continuous:
        flags.add(FLAG_AMBIGUOUS_MULTIDAY)
        return _finish(parsed, window, "unresolved", flags, ())
    if end_at - start_at > MAX_SINGLE_SESSION:
        flags.add(FLAG_CONTINUOUS_SPAN)
    return _finish(
        parsed,
        window,
        "explicit",
        flags,
        (_session(parsed, start_at, end_at, "exact_interval", "explicit", flags),),
    )


def _resolve_end(
    parsed: ParsedDate, start_at: datetime, flags: set[str]
) -> tuple[datetime | None, Rejection | None]:
    """Rules 2, 3 and 8: an end is used only when the source actually states one."""
    if parsed.end_time is None:
        flags.add(FLAG_END_TIME_UNKNOWN)
        if _ends_where_it_starts(parsed):
            flags.add(FLAG_ZERO_LENGTH_SOURCE)
        elif parsed.end_moment is not None and not _is_midnight_of(
            parsed.end_moment, parsed.end_day
        ):
            flags.add(FLAG_END_UNIX_UNUSED)
        return None, None
    if parsed.end_day is not None:
        end_at = _combine(parsed.end_day, parsed.end_time)
    elif parsed.end_moment is not None:
        end_at = parsed.end_moment
        flags.add(FLAG_END_DAY_FROM_UNIX)
    else:
        flags.add(FLAG_END_DAY_UNKNOWN)
        return None, None
    if end_at < start_at:
        return None, Rejection(
            REASON_NEGATIVE_INTERVAL,
            f"end {end_at.isoformat()} precedes start {start_at.isoformat()}",
        )
    if end_at == start_at:
        flags.add(FLAG_ZERO_LENGTH_DROPPED)
        return None, None
    return end_at, None


def _is_midnight_of(moment: datetime, day: date | None) -> bool:
    if day is None:
        return False
    return moment == _combine(day, time())


def _expand_schedules(parsed: ParsedDate, window: Window, flags: set[str]) -> DateResolution:
    """Rule 5: a repeat is materialised only inside the window plus buffer."""
    rules = parsed.schedules or ()
    moments: list[tuple[datetime, datetime]] = []
    for span_start, span_end in window.spans:
        first = max(parsed.start_day or span_start.date(), span_start.date())
        last = min(parsed.end_day or span_end.date(), span_end.date())
        for rule in rules:
            rule_first = max(first, rule.first_day) if rule.first_day is not None else first
            rule_last = min(last, rule.last_day) if rule.last_day is not None else last
            day = rule_first
            while day <= rule_last:
                start_at = _combine(day, rule.start_time)
                end_at = _combine(day, rule.end_time)
                if end_at <= start_at:
                    end_at += DAY
                    flags.add(FLAG_SCHEDULE_CROSSES_MIDNIGHT)
                moments.append((start_at, end_at))
                day += DAY
    unique = sorted(set(moments))
    sessions = tuple(
        _session(parsed, start_at, end_at, "exact_interval", "event_schedule", flags)
        for start_at, end_at in unique
    )
    return _finish(parsed, window, "event_schedule", flags, sessions)


def _dateless(parsed: ParsedDate, basis: ScheduleBasis, flags: set[str]) -> Session:
    """Rule 4: a known day with no clock time keeps the day and no moments."""
    if parsed.start_day is None and parsed.end_day is None:
        flags.add(FLAG_NO_CALENDAR_DATE)
        return _session(parsed, None, None, "unresolved", basis, flags)
    return _session(parsed, None, None, "date_only", basis, flags)


def _session(
    parsed: ParsedDate,
    start_at: datetime | None,
    end_at: datetime | None,
    precision: TimePrecision,
    basis: ScheduleBasis,
    flags: set[str],
) -> Session:
    return Session(
        start_at=start_at,
        end_at=end_at,
        start_date=parsed.start_day.isoformat() if parsed.start_day is not None else None,
        end_date=parsed.end_day.isoformat() if parsed.end_day is not None else None,
        time_precision=precision,
        schedule_basis=basis,
        quality_flags=tuple(sorted(flags)),
    )


def _finish(
    parsed: ParsedDate,
    window: Window,
    basis: ScheduleBasis,
    flags: set[str],
    sessions: tuple[Session, ...],
) -> DateResolution:
    """Keep the in-window sessions, or exactly one row saying why there are none."""
    kept = tuple(session for session in sessions if _in_window(parsed, session, window))
    if not kept:
        flags.add(
            FLAG_NO_IN_WINDOW_SESSION if _span_overlaps(parsed, window) else FLAG_OUTSIDE_WINDOW
        )
        kept = (_session(parsed, None, None, "unresolved", basis, flags),)
    else:
        kept = tuple(
            _session(
                parsed,
                session.start_at,
                session.end_at,
                session.time_precision,
                session.schedule_basis,
                flags,
            )
            for session in kept
        )
    for session in kept:
        problem = invariant_violation(session)
        if problem is not None:
            raise TimeRuleError(problem)
    return DateResolution(kept, None)


def _in_window(parsed: ParsedDate, session: Session, window: Window) -> bool:
    if session.start_at is not None and session.end_at is not None:
        return window.overlaps(session.start_at, session.end_at)
    if session.start_at is not None:
        return window.contains(session.start_at)
    return _span_overlaps(parsed, window)


def _span_overlaps(parsed: ParsedDate, window: Window) -> bool:
    start, end = _span(parsed, window)
    return window.overlaps(start, end)


def _span(parsed: ParsedDate, window: Window) -> tuple[datetime, datetime]:
    """The widest interval the record can occupy, for the window test only.

    An unbounded side is pinned just outside the window so that it always
    overlaps, and a degenerate span still occupies its own start instant.
    """
    before = window.start - DAY
    after = window.end + DAY
    if parsed.start_day is not None:
        start = _combine(parsed.start_day, parsed.start_time or time())
    elif parsed.start_moment is not None:
        start = parsed.start_moment
    else:
        start = before
    if parsed.endless:
        end = after
    elif parsed.end_day is not None:
        end = (
            _combine(parsed.end_day, parsed.end_time)
            if parsed.end_time is not None
            else _combine(parsed.end_day + DAY, time())
        )
    elif parsed.end_moment is not None:
        end = parsed.end_moment
    elif parsed.start_day is not None:
        end = _combine(parsed.start_day + DAY, time())
    else:
        end = after
    return start, max(end, start + timedelta(seconds=1))


def invariant_violation(session: Session) -> str | None:
    """The contract every emitted session must satisfy, or a description of the breach."""
    if session.time_precision == "exact_interval":
        if session.start_at is None or session.end_at is None:
            return "exact_interval requires both moments"
        if session.end_at <= session.start_at:
            return "exact_interval requires end after start"
        return None
    if session.time_precision == "start_only":
        if session.start_at is None or session.end_at is not None:
            return "start_only requires a start and a null end"
        return None
    if session.start_at is not None or session.end_at is not None:
        return f"{session.time_precision} requires both moments to be null"
    if session.time_precision == "date_only" and session.start_date is None:
        if session.end_date is None:
            return "date_only requires at least one calendar date"
    return None


def feeds_hourly_features(session: Session) -> bool:
    """Only a resolved interval may fill hour buckets; ``unresolved`` never does."""
    return session.time_precision == "exact_interval"


def overlaps_hour(session: Session, bucket_start: datetime) -> bool:
    """Rule 7: half-open overlap, so an event ending at 20:00 leaves 20:00-21:00 free."""
    if not feeds_hourly_features(session) or session.start_at is None or session.end_at is None:
        return False
    return session.start_at < bucket_start + HOUR and session.end_at > bucket_start


def occupied_hours(session: Session) -> tuple[datetime, ...]:
    if not feeds_hourly_features(session) or session.start_at is None or session.end_at is None:
        return ()
    bucket = session.start_at.replace(minute=0, second=0, microsecond=0)
    buckets: list[datetime] = []
    while bucket < session.end_at:
        buckets.append(bucket)
        bucket += HOUR
    return tuple(buckets)


def start_hour(session: Session) -> datetime | None:
    """The bucket a start-feature belongs to; ``start_only`` is usable here alone."""
    if session.start_at is None or session.time_precision not in ("exact_interval", "start_only"):
        return None
    return session.start_at.replace(minute=0, second=0, microsecond=0)

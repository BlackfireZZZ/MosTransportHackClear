"""Each test names the numbered time rule it defends, from the collector task."""

from datetime import datetime

import pytest

from tramflow_ml.external.kudago.records import MOSCOW
from tramflow_ml.external.kudago.timerules import (
    FLAG_AMBIGUOUS_MULTIDAY,
    FLAG_CONTINUOUS_SPAN,
    FLAG_END_DAY_FROM_UNIX,
    FLAG_END_DAY_UNKNOWN,
    FLAG_ENDLESS,
    FLAG_NO_CALENDAR_DATE,
    FLAG_NO_IN_WINDOW_SESSION,
    FLAG_OUTSIDE_WINDOW,
    FLAG_PLACE_SCHEDULE_REQUIRED,
    FLAG_SCHEDULE_CROSSES_MIDNIGHT,
    FLAG_SCHEDULE_WEEKDAYS_UNCONFIRMED,
    FLAG_STARTLESS,
    FLAG_TIME_CONFLICT,
    FLAG_UNBOUNDED_END,
    FLAG_ZERO_LENGTH_SOURCE,
    REASON_INVALID_DATE_RECORD,
    REASON_NEGATIVE_INTERVAL,
    Session,
    TimeRuleError,
    feeds_hourly_features,
    invariant_violation,
    occupied_hours,
    overlaps_hour,
    parse_window,
    parse_windows,
    resolve_date_record,
    start_hour,
)

JANUARY = parse_window("2025-01-01", "2025-02-01")


def moscow(text):
    return datetime.fromisoformat(text).replace(tzinfo=MOSCOW)


def unix(text):
    return int(moscow(text).timestamp())


def record(**overrides):
    base = {
        "start_date": None,
        "start_time": None,
        "start": None,
        "end_date": None,
        "end_time": None,
        "end": None,
        "is_continuous": False,
        "is_startless": False,
        "is_endless": False,
        "use_place_schedule": False,
        "schedules": [],
    }
    base.update(overrides)
    return base


def only(resolution):
    assert resolution.rejection is None
    assert len(resolution.sessions) == 1
    return resolution.sessions[0]


# Rule 1: Unix seconds are UTC and are cross-checked against the local fields.


def test_rule1_agreeing_unix_and_local_fields_resolve_to_an_exact_interval():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-15",
            start_time="19:00:00",
            start=unix("2025-01-15T19:00:00"),
            end_date="2025-01-15",
            end_time="21:00:00",
            end=unix("2025-01-15T21:00:00"),
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "exact_interval"
    assert session.start_at == moscow("2025-01-15T19:00:00")
    assert session.end_at == moscow("2025-01-15T21:00:00")
    assert session.schedule_basis == "explicit"


def test_rule1_disagreement_between_local_and_unix_is_unresolved_not_a_choice():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-15",
            start_time="19:00:00",
            start=unix("2025-01-15T20:00:00"),
            end_date="2025-01-15",
            end_time="21:00:00",
            end=unix("2025-01-15T21:00:00"),
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "unresolved"
    assert FLAG_TIME_CONFLICT in session.quality_flags
    assert session.start_at is None and session.end_at is None
    assert session.start_date == "2025-01-15"


def test_rule1_a_unix_hour_the_local_fields_do_not_mention_is_a_conflict():
    resolution = resolve_date_record(
        record(start_date="2025-01-15", start=unix("2025-01-15T19:00:00")),
        JANUARY,
    )

    assert FLAG_TIME_CONFLICT in only(resolution).quality_flags


# Rule 2: end == start with no explicit end means unknown duration.


def test_rule2_end_equal_to_start_never_becomes_a_zero_length_session():
    moment = unix("2025-01-15T19:00:00")

    resolution = resolve_date_record(
        record(
            start_date="2025-01-15",
            start_time="19:00:00",
            start=moment,
            end_date="2025-01-15",
            end=moment,
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "start_only"
    assert session.start_at == moscow("2025-01-15T19:00:00")
    assert session.end_at is None
    assert FLAG_ZERO_LENGTH_SOURCE in session.quality_flags


# Rule 3: a null end_date with an end_time is settled by the Unix end, or not at all.


def test_rule3_end_day_is_derived_from_a_valid_unix_end():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-15",
            start_time="22:00:00",
            start=unix("2025-01-15T22:00:00"),
            end_time="02:00:00",
            end=unix("2025-01-16T02:00:00"),
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "exact_interval"
    assert session.end_at == moscow("2025-01-16T02:00:00")
    assert FLAG_END_DAY_FROM_UNIX in session.quality_flags
    assert session.end_date is None


def test_rule3_midnight_end_lands_on_the_next_day_not_on_the_start_day():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-15",
            start_time="20:00:00",
            start=unix("2025-01-15T20:00:00"),
            end_time="00:00:00",
            end=unix("2025-01-16T00:00:00"),
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.end_at == moscow("2025-01-16T00:00:00")
    assert session.end_at > session.start_at


def test_rule3_without_a_unix_end_the_day_is_not_invented():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-15",
            start_time="22:00:00",
            start=unix("2025-01-15T22:00:00"),
            end_time="02:00:00",
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "start_only"
    assert session.end_at is None
    assert FLAG_END_DAY_UNKNOWN in session.quality_flags


# Rule 4: a calendar date with no clock time is date_only, never midnight.


def test_rule4_a_date_without_a_time_is_date_only_with_no_moments():
    resolution = resolve_date_record(
        record(start_date="2025-01-15", start=unix("2025-01-15T00:00:00")),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "date_only"
    assert session.start_at is None and session.end_at is None
    assert session.start_date == "2025-01-15"
    assert session.end_date is None


# Rule 5: preserved flags, no 24-hour expansion, window-bounded repeats.


def test_rule5_a_month_long_range_without_hours_stays_one_date_only_row():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-01",
            start=unix("2025-01-01T00:00:00"),
            end_date="2025-01-31",
            is_continuous=True,
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "date_only"
    assert (session.start_date, session.end_date) == ("2025-01-01", "2025-01-31")


def test_rule5_a_continuous_range_with_hours_is_one_span_not_a_daily_repeat():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-05",
            start_time="10:00:00",
            start=unix("2025-01-05T10:00:00"),
            end_date="2025-01-20",
            end_time="22:00:00",
            end=unix("2025-01-20T22:00:00"),
            is_continuous=True,
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "exact_interval"
    assert FLAG_CONTINUOUS_SPAN in session.quality_flags


def test_rule5_a_multiday_range_with_hours_and_no_continuity_is_ambiguous():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-05",
            start_time="10:00:00",
            start=unix("2025-01-05T10:00:00"),
            end_date="2025-01-20",
            end_time="22:00:00",
            end=unix("2025-01-20T22:00:00"),
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "unresolved"
    assert FLAG_AMBIGUOUS_MULTIDAY in session.quality_flags


def test_rule5_an_unbounded_repeat_expands_only_inside_the_window():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-01",
            start=unix("2025-01-01T00:00:00"),
            is_endless=True,
            schedules=[{"start_time": "10:00:00", "end_time": "20:00:00", "days_of_week": []}],
        ),
        JANUARY,
    )

    assert resolution.rejection is None
    assert len(resolution.sessions) == 31
    assert resolution.sessions[0].start_at == moscow("2025-01-01T10:00:00")
    assert resolution.sessions[-1].start_at == moscow("2025-01-31T10:00:00")
    assert all(session.schedule_basis == "event_schedule" for session in resolution.sessions)
    assert all(FLAG_ENDLESS in session.quality_flags for session in resolution.sessions)
    assert all(FLAG_UNBOUNDED_END not in session.quality_flags for session in resolution.sessions)


def test_rule5_expanded_sessions_keep_the_original_record_bounds():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-10",
            end_date="2025-01-12",
            schedules=[{"start_time": "18:00:00", "end_time": "23:00:00"}],
        ),
        JANUARY,
    )

    assert [session.start_at.day for session in resolution.sessions] == [10, 11, 12]
    assert {session.start_date for session in resolution.sessions} == {"2025-01-10"}
    assert {session.end_date for session in resolution.sessions} == {"2025-01-12"}


def test_rule5_a_schedule_running_past_midnight_ends_on_the_next_day():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-10",
            end_date="2025-01-10",
            schedules=[{"start_time": "22:00:00", "end_time": "02:00:00"}],
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.end_at == moscow("2025-01-11T02:00:00")
    assert FLAG_SCHEDULE_CROSSES_MIDNIGHT in session.quality_flags


def test_rule5_a_record_entirely_outside_the_window_is_one_flagged_row_not_quarantine():
    resolution = resolve_date_record(
        record(
            start_date="2024-06-01",
            start_time="19:00:00",
            start=unix("2024-06-01T19:00:00"),
            end_date="2024-06-01",
            end_time="21:00:00",
            end=unix("2024-06-01T21:00:00"),
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "unresolved"
    assert FLAG_OUTSIDE_WINDOW in session.quality_flags
    assert session.start_at is None and session.end_at is None
    assert session.start_date == "2024-06-01"
    assert not feeds_hourly_features(session)


def test_rule5_a_repeat_with_no_in_window_day_is_reported_separately():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-20",
            end_date="2025-02-20",
            schedules=[
                {
                    "start_time": "10:00:00",
                    "end_time": "20:00:00",
                    "start_date": "2025-02-10",
                    "end_date": "2025-02-20",
                }
            ],
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.time_precision == "unresolved"
    assert FLAG_NO_IN_WINDOW_SESSION in session.quality_flags
    assert FLAG_OUTSIDE_WINDOW not in session.quality_flags


def test_rule5_startless_and_endless_survive_into_the_flags():
    resolution = resolve_date_record(
        record(start_date="2025-01-10", is_startless=True, is_endless=True),
        JANUARY,
    )

    session = only(resolution)
    assert FLAG_STARTLESS in session.quality_flags
    assert FLAG_ENDLESS in session.quality_flags


def test_rule5_a_sentinel_year_is_an_open_bound_not_a_date():
    resolution = resolve_date_record(
        record(start_date="2025-01-10", end_date="2920-01-01"),
        JANUARY,
    )

    session = only(resolution)
    assert session.end_date is None
    assert FLAG_UNBOUNDED_END in session.quality_flags


# Rule 6: hours are never guessed.


def test_rule6_a_place_schedule_is_never_read_out_of_free_text():
    resolution = resolve_date_record(
        record(start_date="2025-01-10", end_date="2025-01-20", use_place_schedule=True),
        JANUARY,
    )

    session = only(resolution)
    assert session.schedule_basis == "unresolved"
    assert FLAG_PLACE_SCHEDULE_REQUIRED in session.quality_flags
    assert session.time_precision == "unresolved"


def test_rule6_a_partial_weekday_set_is_not_expanded():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-10",
            end_date="2025-01-20",
            schedules=[
                {"start_time": "10:00:00", "end_time": "20:00:00", "days_of_week": [0, 2]}
            ],
        ),
        JANUARY,
    )

    session = only(resolution)
    assert session.schedule_basis == "unresolved"
    assert FLAG_SCHEDULE_WEEKDAYS_UNCONFIRMED in session.quality_flags


def test_rule6_a_full_week_is_unambiguous_whatever_the_week_starts_on():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-10",
            end_date="2025-01-12",
            schedules=[
                {
                    "start_time": "10:00:00",
                    "end_time": "20:00:00",
                    "days_of_week": [0, 1, 2, 3, 4, 5, 6],
                }
            ],
        ),
        JANUARY,
    )

    assert len(resolution.sessions) == 3


# Rule 7: intervals are half-open.


def test_rule7_an_event_ending_at_20_does_not_occupy_the_20_to_21_bucket():
    session = only(
        resolve_date_record(
            record(
                start_date="2025-01-15",
                start_time="18:00:00",
                start=unix("2025-01-15T18:00:00"),
                end_date="2025-01-15",
                end_time="20:00:00",
                end=unix("2025-01-15T20:00:00"),
            ),
            JANUARY,
        )
    )

    assert occupied_hours(session) == (moscow("2025-01-15T18:00:00"), moscow("2025-01-15T19:00:00"))
    assert overlaps_hour(session, moscow("2025-01-15T19:00:00"))
    assert not overlaps_hour(session, moscow("2025-01-15T20:00:00"))
    assert not overlaps_hour(session, moscow("2025-01-15T17:00:00"))


def test_rule7_start_only_feeds_start_features_alone():
    session = only(
        resolve_date_record(
            record(
                start_date="2025-01-15",
                start_time="19:30:00",
                start=unix("2025-01-15T19:30:00"),
            ),
            JANUARY,
        )
    )

    assert session.time_precision == "start_only"
    assert not feeds_hourly_features(session)
    assert occupied_hours(session) == ()
    assert start_hour(session) == moscow("2025-01-15T19:00:00")


def test_rule7_unresolved_never_feeds_hourly_features():
    session = only(resolve_date_record(record(use_place_schedule=True), JANUARY))

    assert session.time_precision == "unresolved"
    assert not feeds_hourly_features(session)
    assert occupied_hours(session) == ()
    assert start_hour(session) is None


# Rule 8: nothing disappears, and a negative interval is quarantine.


def test_rule8_a_negative_interval_is_quarantined_not_silently_zeroed():
    resolution = resolve_date_record(
        record(
            start_date="2025-01-15",
            start_time="20:00:00",
            end_date="2025-01-15",
            end_time="18:00:00",
        ),
        JANUARY,
    )

    assert resolution.sessions == ()
    assert resolution.rejection is not None
    assert resolution.rejection.reason == REASON_NEGATIVE_INTERVAL


def test_rule8_an_empty_date_record_is_reported_as_unresolved():
    session = only(resolve_date_record(record(), JANUARY))

    assert session.time_precision == "unresolved"
    assert FLAG_NO_CALENDAR_DATE in session.quality_flags


def test_rule8_a_malformed_date_is_quarantined_with_its_reason():
    resolution = resolve_date_record(record(start_date="15.01.2025"), JANUARY)

    assert resolution.sessions == ()
    assert resolution.rejection is not None
    assert resolution.rejection.reason == REASON_INVALID_DATE_RECORD
    assert "start_date" in resolution.rejection.detail


# Invariants shared by every emitted session.


@pytest.mark.parametrize(
    "raw",
    [
        record(),
        record(start_date="2025-01-15"),
        record(start_date="2025-01-15", start_time="19:00:00", start=unix("2025-01-15T19:00:00")),
        record(
            start_date="2025-01-15",
            start_time="19:00:00",
            start=unix("2025-01-15T19:00:00"),
            end_date="2025-01-15",
            end_time="21:00:00",
            end=unix("2025-01-15T21:00:00"),
        ),
        record(start_date="2024-06-01"),
        record(start_date="2025-01-10", use_place_schedule=True),
        record(
            start_date="2025-01-10",
            end_date="2025-01-12",
            schedules=[{"start_time": "10:00:00", "end_time": "20:00:00"}],
        ),
    ],
)
def test_every_emitted_session_satisfies_the_precision_invariants(raw):
    resolution = resolve_date_record(raw, JANUARY)

    assert resolution.sessions
    for session in resolution.sessions:
        assert invariant_violation(session) is None


@pytest.mark.parametrize(
    ("session", "expected"),
    [
        (Session(None, None, "2025-01-01", None, "exact_interval", "explicit", ()), "both moments"),
        (
            Session(
                moscow("2025-01-01T10:00:00"),
                moscow("2025-01-01T10:00:00"),
                "2025-01-01",
                None,
                "exact_interval",
                "explicit",
                (),
            ),
            "end after start",
        ),
        (
            Session(
                moscow("2025-01-01T10:00:00"),
                moscow("2025-01-01T12:00:00"),
                "2025-01-01",
                None,
                "start_only",
                "explicit",
                (),
            ),
            "null end",
        ),
        (Session(None, None, None, None, "date_only", "explicit", ()), "calendar date"),
        (
            Session(
                moscow("2025-01-01T10:00:00"), None, None, None, "unresolved", "unresolved", ()
            ),
            "null",
        ),
    ],
)
def test_the_invariants_reject_sessions_that_break_them(session, expected):
    problem = invariant_violation(session)

    assert problem is not None
    assert expected in problem


# Disjoint collection periods stay disjoint.

JANUARY_AND_SEPTEMBER = parse_windows(
    (("2025-01-01", "2025-02-01"), ("2025-09-01", "2025-10-01"))
)


def test_a_gap_between_collected_months_is_not_inside_the_window():
    """The eight months nobody fetched must not count as collected."""
    resolution = resolve_date_record(
        record(
            start_date="2025-03-15",
            start_time="19:00:00",
            start=unix("2025-03-15T19:00:00"),
            end_date="2025-03-15",
            end_time="21:00:00",
            end=unix("2025-03-15T21:00:00"),
        ),
        JANUARY_AND_SEPTEMBER,
    )

    assert FLAG_OUTSIDE_WINDOW in only(resolution).quality_flags


def test_each_collected_month_is_inside_the_window():
    for day in ("2025-01-15", "2025-09-15"):
        resolution = resolve_date_record(
            record(
                start_date=day,
                start_time="19:00:00",
                start=unix(f"{day}T19:00:00"),
                end_date=day,
                end_time="21:00:00",
                end=unix(f"{day}T21:00:00"),
            ),
            JANUARY_AND_SEPTEMBER,
        )

        assert FLAG_OUTSIDE_WINDOW not in only(resolution).quality_flags


def test_a_repeat_does_not_materialise_in_the_gap_between_months():
    """Rule 5 clamps to each span, not to the hull they span together."""
    resolution = resolve_date_record(
        record(
            start_date="2025-01-01",
            end_date="2025-10-01",
            schedules=[
                {
                    "start_time": "19:00:00",
                    "end_time": "21:00:00",
                    "days_of_week": [],
                    "start_date": None,
                    "end_date": None,
                }
            ],
        ),
        JANUARY_AND_SEPTEMBER,
    )

    months = {session.start_at.strftime("%Y-%m") for session in resolution.sessions}
    assert months == {"2025-01", "2025-09"}


def test_touching_spans_are_merged_so_no_session_is_counted_twice():
    window = parse_windows((("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-03-01")))

    assert len(window.spans) == 1


def test_a_window_needs_at_least_one_span():
    with pytest.raises(TimeRuleError):
        parse_windows(())

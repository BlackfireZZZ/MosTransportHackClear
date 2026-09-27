"""A date the source has not published yet is missing, not an observed zero."""

from datetime import date, datetime, timedelta

import pytest

from tramflow_ml.features import (
    DAY_HOUR,
    MOSCOW,
    YEAR_MONTH,
    CoverageCalendar,
    EntityKey,
    FeatureError,
    FeatureRequest,
    Observation,
    aggregate,
    build_features,
)

TARGET = "synthetic_boardings"
UNIT = "event_count"
ORIGIN = "2025-06-10T00:00"
ENTITY = EntityKey("route-1", "dir-0", "stop-1")
OTHER = EntityKey("route-1", "dir-1", "stop-1")
HOURS_PER_DAY = 24
PUBLICATION_LAG_HOURS = 26


def moscow(text):
    return datetime.fromisoformat(text).replace(tzinfo=MOSCOW)


def observation(text, lag_hours=1, entity=ENTITY):
    event_at = moscow(text)
    return Observation(entity, event_at, event_at + timedelta(hours=lag_hours), TARGET, UNIT)


def a_full_day(day, lag_hours):
    """Twenty-four real events, one an hour, all published after the same lag."""
    return tuple(observation(f"{day}T{hour:02d}:30", lag_hours) for hour in range(HOURS_PER_DAY))


def calendar(published_at=None):
    return CoverageCalendar.from_range(
        date(2024, 1, 1), date(2026, 1, 1), published_at=published_at
    )


def build(policy, origin, observations, coverage):
    return build_features(
        FeatureRequest(
            policy=policy,
            origin=moscow(origin),
            entities=(ENTITY, OTHER),
            observations=tuple(observations),
            coverage=coverage,
            target=TARGET,
            unit=UNIT,
        )
    )


def row_at(built, start, entity=ENTITY):
    return next(
        row for row in built.rows if row.entity == entity and row.bucket.start == moscow(start)
    )


def test_a_day_whose_data_is_not_published_yet_is_missing_not_zero():
    """The reviewer's case: 24 real events, published 26 hours late, must not read as zero."""
    events = a_full_day("2025-06-09", PUBLICATION_LAG_HOURS)
    published = {date(2025, 6, 9): moscow("2025-06-10T12:00")}

    blind = build(DAY_HOUR, ORIGIN, events, calendar())
    honest = build(DAY_HOUR, ORIGIN, events, calendar(published))

    assert blind.rows[0].features["lag_24h"] == 0.0
    assert honest.rows[0].features["lag_24h"] is None
    assert honest.rows[0].features["roll_24h_sum"] is None
    assert honest.rows[0].features["roll_24h_coverage"] == 0.0
    assert honest.feature_digest != blind.feature_digest


def test_the_same_day_becomes_observed_at_a_later_cutoff():
    events = a_full_day("2025-06-09", PUBLICATION_LAG_HOURS)
    coverage = calendar({date(2025, 6, 9): moscow("2025-06-10T12:00")})

    early = build(DAY_HOUR, ORIGIN, events, coverage)
    late = build(DAY_HOUR, "2025-06-11T00:00", events, coverage)

    assert row_at(early, "2025-06-10T08:00").features["lag_24h"] is None
    assert row_at(late, "2025-06-11T08:00").features["lag_48h"] == 1.0
    assert row_at(late, "2025-06-11T08:00").features["roll_24h_coverage"] == 1.0


def test_an_unpublished_bucket_is_distinguishable_from_an_observed_zero():
    quiet = build(DAY_HOUR, ORIGIN, (), calendar())
    never = calendar({date(2025, 6, 9): moscow("2030-01-01T00:00")})
    unpublished = build(DAY_HOUR, ORIGIN, (), never)

    assert quiet.rows[0].features["lag_24h"] == 0.0
    assert unpublished.rows[0].features["lag_24h"] is None
    assert quiet.feature_digest != unpublished.feature_digest


def test_labels_still_see_a_date_the_history_cannot():
    """Labels are measured on the complete calendar; only history is clipped at the cutoff."""
    events = a_full_day("2025-06-09", PUBLICATION_LAG_HOURS)
    coverage = calendar({date(2025, 6, 9): moscow("2025-06-10T12:00")})

    labelled = build(DAY_HOUR, "2025-06-09T00:00", events, coverage)
    later = build(DAY_HOUR, ORIGIN, events, coverage)

    assert row_at(labelled, "2025-06-09T08:00").target_value == 1
    assert row_at(labelled, "2025-06-09T08:00").target_coverage == "observed"
    assert row_at(later, "2025-06-10T08:00").features["lag_24h"] is None


def test_a_calendar_with_publication_instants_is_not_a_dead_end_for_the_aggregator():
    """An as-of view must never force the caller to pass a mutilated calendar."""
    events = a_full_day("2025-06-09", PUBLICATION_LAG_HOURS)
    coverage = calendar({date(2025, 6, 9): moscow("2025-06-10T12:00")})

    index = aggregate(events, "daily", coverage.as_of(moscow(ORIGIN)), TARGET, UNIT)

    assert index.total == 0
    assert index.cell(ENTITY, moscow("2025-06-09T00:00")).coverage == "missing"


def test_an_event_on_a_date_the_calendar_never_covers_is_still_an_error():
    outside = CoverageCalendar.from_range(date(2025, 6, 1), date(2025, 6, 9))

    with pytest.raises(FeatureError, match="2025-06-09"):
        aggregate(a_full_day("2025-06-09", 1), "daily", outside.complete(), TARGET, UNIT)


def test_a_publication_instant_for_an_uncovered_date_is_refused():
    with pytest.raises(FeatureError, match="not a covered date"):
        CoverageCalendar.from_range(
            date(2025, 6, 1),
            date(2025, 6, 9),
            gaps=[date(2025, 6, 5)],
            published_at={date(2025, 6, 5): moscow("2025-06-06T00:00")},
        )


def test_a_naive_publication_instant_is_refused():
    with pytest.raises(FeatureError, match="timezone-aware"):
        CoverageCalendar.from_range(
            date(2025, 6, 1),
            date(2025, 6, 9),
            published_at={date(2025, 6, 5): datetime(2025, 6, 6)},
        )


@pytest.mark.parametrize(
    ("target", "unit"),
    [
        ("onboard_load", "passengers"),
        ("boarding_count", "passengers"),
        ("synthetic_boardings", "passengers"),
        ("onboard_load", "event_count"),
    ],
)
def test_a_unit_the_layer_cannot_produce_is_refused(target, unit):
    with pytest.raises(FeatureError):
        aggregate((), "daily", calendar().complete(), target, unit)
    with pytest.raises(FeatureError):
        FeatureRequest(
            policy=YEAR_MONTH,
            origin=moscow("2025-01-01T00:00"),
            entities=(ENTITY,),
            observations=(),
            coverage=calendar(),
            target=target,
            unit=unit,
        )


@pytest.mark.parametrize("target", ["synthetic_boardings", "validation_count"])
def test_the_counted_targets_are_accepted(target):
    index = aggregate((), "daily", calendar().complete(), target, UNIT)

    assert index.target == target
    assert index.unit == UNIT

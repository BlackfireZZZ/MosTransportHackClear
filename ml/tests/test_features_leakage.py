import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from tramflow_ml.features import (
    DAY_HOUR,
    MONTH_DAY,
    MOSCOW,
    YEAR_MONTH,
    CoverageCalendar,
    EntityKey,
    FeatureRequest,
    Observation,
    build_features,
    encode,
)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.data_v1 import EventRow  # noqa: E402, I001

TARGET = "synthetic_boardings"
UNIT = "event_count"
ENTITY = EntityKey("route-1", "dir-0", "stop-1")
OTHER = EntityKey("route-1", "dir-1", "stop-1")
PERTURBATION_COUNT = 40
MINUTES_PER_HOUR = 60
POLICY_ORIGINS = [
    (DAY_HOUR, "2025-06-10T00:00"),
    (MONTH_DAY, "2025-06-01T00:00"),
    (YEAR_MONTH, "2025-01-01T00:00"),
]


def moscow(text):
    return datetime.fromisoformat(text).replace(tzinfo=MOSCOW)


def observation(text, lag_minutes=1, entity=ENTITY):
    event_at = moscow(text)
    return Observation(entity, event_at, event_at + timedelta(minutes=lag_minutes), TARGET, UNIT)


def event_row(text, lag_minutes=1):
    event_at = moscow(text)
    return EventRow(
        schema_version="data.v1",
        entity_version="entities.v1",
        source_version="synthetic.v1",
        event_id="ev-1",
        route_id=ENTITY.route_id,
        direction_id=ENTITY.direction_id,
        stop_id=ENTITY.stop_id,
        stop_sequence=0,
        vehicle_id="veh-1",
        event_at=event_at,
        available_at=event_at + timedelta(minutes=lag_minutes),
        synthetic=True,
    )


PAST_EVENTS = (
    observation("2024-06-14T08:00"),
    observation("2024-06-14T09:00"),
    observation("2025-05-14T08:00"),
    observation("2025-06-09T08:00"),
    observation("2025-06-09T08:30", entity=OTHER),
)


@pytest.fixture
def coverage():
    return CoverageCalendar.from_range(date(2024, 1, 1), date(2026, 1, 1))


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


def post_cutoff_events(origin_text, count=PERTURBATION_COUNT):
    """Events strictly after the cutoff, spread over the horizon and beyond it."""
    origin = moscow(origin_text)
    return tuple(
        Observation(
            ENTITY if index % 2 else OTHER,
            origin + timedelta(hours=index + 1),
            origin + timedelta(hours=index + 1, minutes=1),
            TARGET,
            UNIT,
        )
        for index in range(count)
    )


@pytest.mark.parametrize(("policy", "origin"), POLICY_ORIGINS)
def test_appending_post_cutoff_events_changes_no_feature(policy, origin, coverage):
    base = build(policy, origin, PAST_EVENTS, coverage)

    perturbed = build(policy, origin, (*PAST_EVENTS, *post_cutoff_events(origin)), coverage)

    assert perturbed.feature_digest == base.feature_digest
    assert [encode(row.feature_dict()) for row in perturbed.rows] == [
        encode(row.feature_dict()) for row in base.rows
    ]


@pytest.mark.parametrize(("policy", "origin"), POLICY_ORIGINS)
def test_replacing_the_post_cutoff_values_changes_no_feature(policy, origin, coverage):
    few = build(policy, origin, (*PAST_EVENTS, *post_cutoff_events(origin, 3)), coverage)

    many = build(policy, origin, (*PAST_EVENTS, *post_cutoff_events(origin, 60)), coverage)

    assert many.feature_digest == few.feature_digest


def test_reordering_the_input_changes_nothing(coverage):
    forward = build(DAY_HOUR, "2025-06-10T00:00", PAST_EVENTS, coverage)

    backward = build(DAY_HOUR, "2025-06-10T00:00", tuple(reversed(PAST_EVENTS)), coverage)

    assert backward.digest == forward.digest


def test_events_after_the_horizon_change_not_even_the_labels(coverage):
    origin = "2025-06-01T00:00"
    later = tuple(
        observation(f"2025-07-{day:02d}T12:00") for day in range(1, 1 + PERTURBATION_COUNT // 4)
    )

    base = build(MONTH_DAY, origin, PAST_EVENTS, coverage)
    extended = build(MONTH_DAY, origin, (*PAST_EVENTS, *later), coverage)

    assert extended.digest == base.digest


def row_at(built, start, entity=ENTITY):
    return next(
        row for row in built.rows if row.entity == entity and row.bucket.start == moscow(start)
    )


def test_an_event_published_after_the_cutoff_is_not_a_feature(coverage):
    origin = "2025-06-10T00:00"
    on_time = observation("2025-06-09T08:00", lag_minutes=1)
    late = observation("2025-06-09T08:00", lag_minutes=MINUTES_PER_HOUR * 48)

    prompt = build(DAY_HOUR, origin, (on_time,), coverage)
    delayed = build(DAY_HOUR, origin, (late,), coverage)

    assert row_at(prompt, "2025-06-10T08:00").features["lag_24h"] == 1.0
    assert row_at(delayed, "2025-06-10T08:00").features["lag_24h"] == 0.0
    assert prompt.feature_digest != delayed.feature_digest


def test_moving_availability_across_the_cutoff_does_change_the_features(coverage):
    origin = "2025-06-10T00:00"
    visible = observation("2025-06-09T08:00", lag_minutes=1)
    hidden = observation("2025-06-09T08:00", lag_minutes=MINUTES_PER_HOUR * 40)

    assert visible.visible_at(moscow(origin))
    assert not hidden.visible_at(moscow(origin))
    assert (
        build(DAY_HOUR, origin, (visible,), coverage).feature_digest
        != build(DAY_HOUR, origin, (hidden,), coverage).feature_digest
    )


def test_the_cutoff_boundary_is_half_open_on_the_event_and_closed_on_availability():
    cutoff = moscow("2025-06-10T00:00")

    published_at_cutoff = Observation(ENTITY, moscow("2025-06-09T23:00"), cutoff, TARGET, UNIT)
    happening_at_cutoff = Observation(
        ENTITY, cutoff, cutoff + timedelta(minutes=1), TARGET, UNIT
    )

    assert published_at_cutoff.visible_at(cutoff)
    assert not happening_at_cutoff.visible_at(cutoff)


@pytest.mark.parametrize("event_text", ["2025-06-09T23:00", "2025-06-10T00:00", "2025-06-10T01:00"])
@pytest.mark.parametrize("lag_minutes", [1, 60, 2400])
def test_the_availability_rule_matches_the_data_contract(event_text, lag_minutes):
    cutoff = moscow("2025-06-10T00:00")

    mine = observation(event_text, lag_minutes).visible_at(cutoff)

    assert mine == event_row(event_text, lag_minutes).visible_at(cutoff)


def test_the_header_records_the_cutoff_that_was_applied(coverage):
    built = build(MONTH_DAY, "2025-06-01T00:00", PAST_EVENTS, coverage)

    assert built.header["cutoff"] == "2025-06-01T00:00:00+03:00"
    assert built.header["forecast_origin"] == "2025-06-01T00:00:00+03:00"
    assert all(row.cutoff == moscow("2025-06-01T00:00") for row in built.rows)


def coverage_with_gap(*gaps):
    return CoverageCalendar.from_range(date(2024, 1, 1), date(2026, 1, 1), gaps=gaps)


def gap_dates(start, days):
    return tuple(start + timedelta(days=offset) for offset in range(days))


@pytest.mark.parametrize(
    "hole",
    [
        gap_dates(date(2025, 7, 2), 20),
        gap_dates(date(2025, 11, 3), 25),
        gap_dates(date(2025, 12, 5), 20),
    ],
)
def test_a_coverage_hole_after_the_cutoff_cannot_move_a_feature(hole):
    """The leakage guarantee is about the coverage statement too, not only the events.

    A gap lying entirely inside the forecast horizon is a fact about the future. The
    ``*_units`` ratios ask the coverage calendar how much of a bucket exists, so without
    a cutoff gate they answer about buckets the forecaster has not reached yet.
    """
    baseline = build(YEAR_MONTH, "2025-01-01T00:00", PAST_EVENTS, coverage_with_gap())

    holed = build(YEAR_MONTH, "2025-01-01T00:00", PAST_EVENTS, coverage_with_gap(*hole))

    assert min(hole) > date(2025, 1, 1)
    assert holed.feature_digest == baseline.feature_digest
    assert [row.feature_dict() for row in holed.rows] == [
        row.feature_dict() for row in baseline.rows
    ]


def test_a_coverage_hole_before_the_cutoff_does_move_the_features():
    baseline = build(YEAR_MONTH, "2025-01-01T00:00", PAST_EVENTS, coverage_with_gap())

    holed = build(
        YEAR_MONTH,
        "2025-01-01T00:00",
        PAST_EVENTS,
        coverage_with_gap(*gap_dates(date(2024, 7, 3), 25)),
    )

    assert holed.feature_digest != baseline.feature_digest


def test_a_post_cutoff_coverage_hole_still_moves_the_labels():
    hole = gap_dates(date(2025, 7, 2), 20)
    baseline = build(YEAR_MONTH, "2025-01-01T00:00", PAST_EVENTS, coverage_with_gap())

    holed = build(YEAR_MONTH, "2025-01-01T00:00", PAST_EVENTS, coverage_with_gap(*hole))

    assert holed.digest != baseline.digest
    moved = [
        row.bucket.start.date()
        for holed_row, row in zip(holed.rows, baseline.rows, strict=True)
        if holed_row.target_covered_units != row.target_covered_units
    ]
    assert date(2025, 7, 1) in moved


@pytest.mark.parametrize("policy_origin", POLICY_ORIGINS)
def test_no_units_column_reports_a_bucket_that_has_not_finished(policy_origin):
    policy, origin = policy_origin
    built = build(policy, origin, PAST_EVENTS, coverage_with_gap())

    for row in built.rows:
        for name, value in row.features.items():
            if name.endswith("_units") and name.startswith("lag_"):
                assert value is None or 0.0 <= value <= 1.0

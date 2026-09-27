from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from tramflow_ml.features import (
    DAY_HOUR,
    MOSCOW,
    CapacityRecord,
    CoverageCalendar,
    EntityKey,
    FeatureError,
    FeatureRequest,
    Observation,
    build_features,
)

TARGET = "synthetic_boardings"
UNIT = "event_count"
ORIGIN = "2024-05-08T00:00"

BUSY = EntityKey("route-1", "dir-0", "stop-1")
QUIET = EntityKey("route-1", "dir-0", "stop-2")
CAPACITY = 180.0


def moscow(text):
    return datetime.fromisoformat(text).replace(tzinfo=MOSCOW)


def observation(entity, text):
    event_at = moscow(text)
    return Observation(entity, event_at, event_at + timedelta(minutes=1), TARGET, UNIT)


def table(coverage, observations=(), capacities=None, origin=ORIGIN):
    request = FeatureRequest(
        policy=DAY_HOUR,
        origin=moscow(origin),
        entities=(BUSY, QUIET),
        observations=tuple(observations),
        coverage=coverage,
        target=TARGET,
        unit=UNIT,
        capacities={} if capacities is None else capacities,
    )
    return build_features(request)


def row_for(built, entity, hour):
    start = moscow(f"2024-05-08T{hour:02d}:00")
    return next(row for row in built.rows if row.entity == entity and row.bucket.start == start)


@pytest.fixture
def full_coverage():
    return CoverageCalendar.from_range(date(2024, 5, 1), date(2024, 5, 10))


@pytest.fixture
def events():
    return (observation(BUSY, "2024-05-07T08:15"), observation(BUSY, "2024-05-06T08:20"))


def test_a_covered_bucket_without_events_is_an_observed_zero(full_coverage, events):
    built = table(full_coverage, events)

    assert row_for(built, BUSY, 8).features["lag_24h"] == 1.0
    assert row_for(built, QUIET, 8).features["lag_24h"] == 0.0
    assert row_for(built, QUIET, 8).features["lag_24h"] is not None


def test_one_row_can_hold_an_observed_zero_and_a_missing_lag_at_once():
    with_gap = CoverageCalendar.from_range(
        date(2024, 5, 1), date(2024, 5, 10), gaps=[date(2024, 5, 6)]
    )

    built = table(with_gap, [observation(BUSY, "2024-05-07T08:15")])
    features = row_for(built, BUSY, 8).features

    assert features["lag_24h"] == 1.0
    assert features["lag_48h"] is None
    assert row_for(built, QUIET, 8).features["lag_24h"] == 0.0
    assert row_for(built, QUIET, 8).features["lag_48h"] is None


def test_missing_and_observed_zero_produce_different_digests():
    covered = CoverageCalendar.from_range(date(2024, 5, 1), date(2024, 5, 10))
    with_gap = CoverageCalendar.from_range(
        date(2024, 5, 1), date(2024, 5, 10), gaps=[date(2024, 5, 7)]
    )

    zeroes = table(covered)
    gapped = table(with_gap)

    assert zeroes.rows[0].features["lag_24h"] == 0.0
    assert gapped.rows[0].features["lag_24h"] is None
    assert zeroes.feature_digest != gapped.feature_digest


def test_a_rolling_window_without_any_observation_is_missing_everywhere():
    single_day = CoverageCalendar.from_dates([date(2024, 5, 8)])

    built = table(single_day)
    features = built.rows[0].features

    assert features["roll_24h_sum"] is None
    assert features["roll_24h_mean"] is None
    assert features["roll_24h_max"] is None
    assert features["roll_24h_coverage"] == 0.0
    assert features["roll_168h_sum"] is None


def test_a_partly_covered_rolling_window_reports_its_coverage(events):
    partial = CoverageCalendar.from_range(date(2024, 5, 6), date(2024, 5, 10))

    built = table(partial, events)
    features = row_for(built, BUSY, 8).features

    assert features["roll_24h_coverage"] == 1.0
    assert 0.0 < features["roll_168h_coverage"] < 1.0
    assert features["roll_168h_sum"] == 2.0


def test_unknown_capacity_is_missing_and_a_known_one_is_a_float(full_coverage):
    known = CapacityRecord(CAPACITY, moscow("2024-01-01T00:00"))

    built = table(full_coverage, capacities={BUSY: known})

    assert row_for(built, BUSY, 0).features["entity_capacity"] == CAPACITY
    assert row_for(built, QUIET, 0).features["entity_capacity"] is None


def test_a_capacity_known_only_after_the_cutoff_is_missing(full_coverage):
    later = CapacityRecord(CAPACITY, moscow("2026-01-01T00:00"))
    earlier = CapacityRecord(CAPACITY, moscow("2024-05-08T00:00"))

    future = table(full_coverage, capacities={BUSY: later})
    present = table(full_coverage, capacities={BUSY: earlier})

    assert row_for(future, BUSY, 0).features["entity_capacity"] is None
    assert row_for(present, BUSY, 0).features["entity_capacity"] == CAPACITY
    assert future.feature_digest != present.feature_digest


def test_an_unstamped_capacity_is_refused_at_the_request_boundary(full_coverage):
    with pytest.raises(FeatureError, match="availability instant"):
        table(full_coverage, capacities={BUSY: CAPACITY})


def test_a_capacity_mapping_is_copied_so_a_caller_cannot_mutate_the_request(full_coverage):
    held = {BUSY: CapacityRecord(CAPACITY, moscow("2024-01-01T00:00"))}
    request = FeatureRequest(
        policy=DAY_HOUR,
        origin=moscow(ORIGIN),
        entities=(BUSY, QUIET),
        observations=(),
        coverage=full_coverage,
        target=TARGET,
        unit=UNIT,
        capacities=held,
    )

    held.clear()

    assert request.capacity_at(BUSY, moscow(ORIGIN)) == CAPACITY


def test_an_uncovered_target_bucket_has_no_label(events):
    without_the_eighth = CoverageCalendar.from_range(
        date(2024, 5, 1), date(2024, 5, 10), gaps=[date(2024, 5, 8)]
    )

    built = table(without_the_eighth, events)
    row = row_for(built, BUSY, 8)

    assert row.target_value is None
    assert row.target_coverage == "missing"
    assert row.target_covered_units == 0


def test_a_covered_target_bucket_without_events_has_a_zero_label(full_coverage, events):
    built = table(full_coverage, events)
    row = row_for(built, QUIET, 8)

    assert row.target_value == 0
    assert row.target_coverage == "observed"
    assert row.target_total_units == 1


def test_calendar_features_are_never_missing(full_coverage):
    built = table(full_coverage)

    for row in built.rows:
        assert row.features["hour_of_day"] is not None
        assert row.features["bucket_hours"] == 1.0


def test_a_feature_row_cannot_claim_a_label_it_does_not_have(full_coverage):
    built = table(full_coverage)
    row = built.rows[0]

    with pytest.raises(FeatureError, match="missing target"):
        replace(row, target_value=None, target_coverage="observed")
    with pytest.raises(FeatureError, match="missing target"):
        replace(row, target_value=5, target_coverage="missing")
    with pytest.raises(FeatureError, match="within"):
        replace(row, target_covered_units=2, target_total_units=1)

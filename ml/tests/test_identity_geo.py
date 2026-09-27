from datetime import timedelta

import pytest
from test_identity_fixtures import EVENT_AT, catalog, config, crosswalk, event

from tramflow_ml.identity import (
    Ambiguous,
    GeoPoint,
    Matched,
    Stale,
    Unmatched,
    align_event,
    haversine_metres,
)

METRES_PER_DEGREE_LATITUDE = 111_195.08
ONE_MILLIDEGREE_METRES = METRES_PER_DEGREE_LATITUDE / 1000


def align(source_event, **config_overrides):
    return align_event(catalog(), crosswalk(), config(**config_overrides), source_event).match


def telemetry(latitude, **overrides):
    return event(source_id="telemetry", latitude=latitude, longitude=37.60, **overrides)


def test_haversine_matches_the_meridian_arc_length():
    assert haversine_metres(GeoPoint(0, 0), GeoPoint(1, 0)) == pytest.approx(
        METRES_PER_DEGREE_LATITUDE, abs=0.01
    )
    assert haversine_metres(GeoPoint(55.75, 37.6), GeoPoint(55.75, 37.6)) == 0


def test_geo_point_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError, match="valid range"):
        GeoPoint(90.5, 0)


def test_position_just_inside_tolerance_matches_and_just_outside_is_unmatched():
    inside = 55.75 + 0.000895
    outside = 55.75 + 0.000905

    assert align(telemetry(inside), geo_tolerance_metres=100) == Matched(
        "geo_nearest", "route:1", "dir:0", "stop:A1", 0
    )
    assert align(telemetry(outside), geo_tolerance_metres=100) == Unmatched(
        "no_stop_within_tolerance"
    )


def test_distance_exactly_equal_to_tolerance_counts_as_within_tolerance():
    stop_a1 = GeoPoint(55.75, 37.60)
    one_millidegree_north = GeoPoint(55.751, 37.60)
    boundary = haversine_metres(stop_a1, one_millidegree_north)

    assert boundary == pytest.approx(ONE_MILLIDEGREE_METRES, abs=1e-6)
    assert align(telemetry(55.751), geo_tolerance_metres=boundary) == Matched(
        "geo_nearest", "route:1", "dir:0", "stop:A1", 0
    )
    assert align(telemetry(55.751), geo_tolerance_metres=boundary - 1e-9) == Unmatched(
        "no_stop_within_tolerance"
    )


def test_two_stops_within_tolerance_are_ambiguous_nearest_first():
    result = align(telemetry(55.75), geo_tolerance_metres=2000)

    assert result == Ambiguous("stop", ("stop:A1", "stop:A2"))


def test_geo_candidates_are_scoped_to_the_pattern_of_the_event():
    result = align(telemetry(55.78))

    assert result == Unmatched("no_stop_within_tolerance")


def test_stale_fix_is_flagged_and_boundary_lag_still_joins():
    stale = telemetry(55.75, fix_at=EVENT_AT - timedelta(seconds=61))
    fresh = telemetry(55.75, fix_at=EVENT_AT - timedelta(seconds=60))

    assert align(stale) == Stale(61.0)
    assert align(fresh).outcome == "matched"


def test_missing_positions_and_invalid_coordinates_are_unmatched():
    without_positions = crosswalk(stop_positions={})

    assert align_event(catalog(), without_positions, config(), telemetry(55.75)).match == Unmatched(
        "no_stop_positions"
    )
    assert align(telemetry(95.0)) == Unmatched("invalid_position")


def test_geo_match_of_a_repeated_stop_still_needs_a_sequence():
    assert align(telemetry(55.77)) == Ambiguous("stop_sequence", ("stop:B@1", "stop:B@3"))
    assert align(telemetry(55.77, stop_sequence=1)).stop_sequence == 1

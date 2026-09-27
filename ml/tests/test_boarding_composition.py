from datetime import date

import numpy as np
import pytest

from tramflow_ml.boarding.composition import (
    RouteEvidence,
    compose_route,
    known_service_warnings,
    monotone_stop_match,
    timetable_emissions,
)
from tramflow_ml.boarding.sequence import CycleTemplate


def test_geographic_mapping_is_ordered_and_does_not_reverse_direction() -> None:
    stops = [{"lat": 55.7, "lon": 37.6 + i * 0.01} for i in range(4)]
    match, _ = monotone_stop_match(stops, stops)
    assert match == {i: i for i in range(4)}
    reverse, _ = monotone_stop_match(stops, list(reversed(stops)))
    assert len(reverse) == 1


def test_actual_schedule_clock_and_boardability() -> None:
    template = CycleTemplate(
        "7", ("p", "p"), ("a", "b"), ("0", "0"), (100, 100), (False, True), "fixture"
    )
    evidence = RouteEvidence(
        template,
        ({"boardable": True}, {"boardable": False}),
        ((3600,), ()),
        ("historical_exact_day", "absent"),
        (),
        (),
        True,
    )
    seconds = np.array([3600.0 - 10800 + 15, 3600.0 - 10800 + 900])
    emissions = timetable_emissions(seconds, evidence)
    assert emissions[0, 0] == 0 and emissions[1, 0] < 0
    assert np.isneginf(emissions[:, 1]).all()
    assert timetable_emissions(seconds, evidence, use_schedule=False)[0, 0] == 0


def test_known_diversion_has_date_and_weekday_boundary() -> None:
    assert known_service_warnings("50", date(2025, 7, 29))
    assert not known_service_warnings("50", date(2025, 8, 11))
    assert known_service_warnings("17", date(2025, 4, 5))
    assert not known_service_warnings("17", date(2025, 4, 7))


def test_no_arbitrary_cycle_when_missing_direction() -> None:
    with pytest.raises(ValueError, match="two directed"):
        compose_route([], {"sources": []}, "1", date(2025, 1, 1))


def test_missing_schedule_does_not_become_a_perfect_match() -> None:
    template = CycleTemplate(
        "7", ("p", "p"), ("a", "b"), ("0", "0"), (100, 100), (False, True), "fixture"
    )
    evidence = RouteEvidence(
        template,
        ({"boardable": True}, {"boardable": True}),
        ((3600,), ()),
        ("historical_exact_day", "absent"),
        (),
        (),
        True,
    )
    emissions = timetable_emissions(np.array([0.0]), evidence)
    assert np.array_equal(emissions, np.zeros((1, 2)))


def test_local_midnight_selects_previous_utc_date() -> None:
    from tramflow_ml.boarding.composition import as_of

    assert as_of(date(2025, 1, 15)).isoformat() == "2025-01-14T21:00:00+00:00"

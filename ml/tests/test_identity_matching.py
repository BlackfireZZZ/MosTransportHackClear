import copy
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_identity_fixtures import CATALOG, MOSCOW, SQUARE, catalog, config, crosswalk, event

from tramflow_ml.identity import Ambiguous, Matched, Unmatched, align_event

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.data_v1 import EntityCatalog  # noqa: E402, I001


def align(source_event, **config_overrides):
    return align_event(catalog(), crosswalk(), config(**config_overrides), source_event).match


def test_duplicate_stop_name_is_ambiguous_without_a_qualified_entry():
    result = align(event(stop_name=SQUARE))

    assert result == Ambiguous("stop", ("stop:A1", "stop:A2"))


def test_duplicate_stop_name_resolves_through_qualified_route_direction_entry():
    result = align(event(stop_name=SQUARE, direction_id="B"))

    assert result == Matched("name_route_direction", "route:1", "dir:1", "stop:A2", 1)


def test_unique_name_on_pattern_matches_and_unknown_name_is_unmatched():
    assert align(event(stop_name="Парк", route_id="2")) == Ambiguous(
        "stop_sequence", ("stop:B@1", "stop:B@3")
    )
    assert align(event(stop_name="Депо", route_id="2", stop_sequence=2)) == Matched(
        "name_route_direction", "route:2", "dir:0", "stop:C", 2
    )
    assert align(event(stop_name="Никуда")) == Unmatched("unknown_stop_name")


def test_opposite_directions_are_resolved_only_by_direction_id():
    forward = align(event(stop_id="S-A1", direction_id="A"))
    backward = align(event(stop_id="S-A1", direction_id="B"))

    assert forward == Matched("exact_id", "route:1", "dir:0", "stop:A1", 0)
    assert backward == Matched("exact_id", "route:1", "dir:1", "stop:A1", 3)


def test_missing_or_unknown_direction_is_never_inferred_from_the_stop():
    assert align(event(stop_id="S-C", route_id="2", direction_id=None)) == Unmatched(
        "direction_missing"
    )
    assert align(event(stop_id="S-C", route_id="2", direction_id="Z")) == Unmatched(
        "unknown_direction"
    )
    assert align(event(stop_id="S-A1", route_id="2", direction_id="B")) == Unmatched(
        "unknown_pattern"
    )


def test_route_problems_and_stops_off_pattern_are_unmatched_with_reasons():
    assert align(event(stop_id="S-A1", route_id=None)) == Unmatched("route_missing")
    assert align(event(stop_id="S-A1", route_id="7")) == Unmatched("unknown_route")
    assert align(event(stop_id="S-C")) == Unmatched("stop_not_on_pattern")
    assert align(event(stop_id="S-ghost")) == Unmatched("unknown_stop")
    assert align(event()) == Unmatched("stop_missing")


def test_repeated_stop_on_loop_needs_sequence_or_previous_visit():
    assert align(event(stop_id="S-B")) == Ambiguous("stop_sequence", ("stop:B@1", "stop:B@3"))
    assert align(event(stop_id="S-B", stop_sequence=3)).stop_sequence == 3
    assert align(event(stop_id="S-B", stop_sequence=2)) == Unmatched("stop_sequence_mismatch")
    assert align(event(stop_id="S-B", previous_stop_id="S-A1")).stop_sequence == 1
    assert align(event(stop_id="S-B", previous_stop_id="S-A2")).stop_sequence == 3
    assert align(event(stop_id="S-B", previous_stop_id="S-C")) == Unmatched(
        "previous_stop_mismatch"
    )
    assert align(event(stop_id="S-B", previous_stop_id="S-ghost")) == Unmatched(
        "previous_stop_unknown"
    )


def test_previous_visit_hint_stays_ambiguous_when_both_visits_follow_the_same_stop():
    result = align(event(stop_id="S-B", route_id="2", previous_stop_id="S-C"))

    assert result == Ambiguous("stop_sequence", ("stop:B@1", "stop:B@3"))


def test_one_based_source_sequence_is_shifted_by_the_configured_base():
    result = align(event(stop_id="S-B", stop_sequence=4), stop_sequence_base=1)

    assert result == Matched("exact_id", "route:1", "dir:0", "stop:B", 3)


def test_vehicle_route_change_at_noon_vetoes_conflicting_routes_only():
    morning = datetime(2024, 3, 1, 8, 0, tzinfo=MOSCOW)
    noon = datetime(2024, 3, 1, 12, 0, tzinfo=MOSCOW)
    on_route_1 = {"stop_id": "S-A1", "route_id": "1", "vehicle_id": "V1"}
    on_route_2 = {"stop_id": "S-C", "route_id": "2", "stop_sequence": 0, "vehicle_id": "V1"}

    def at(instant, **fields):
        return event(event_at=instant, available_at=instant + timedelta(minutes=1), **fields)

    assert align(at(morning, **on_route_1)).outcome == "matched"
    assert align(at(morning, **on_route_2)) == Unmatched("vehicle_route_conflict")
    assert align(at(noon, **on_route_1)) == Unmatched("vehicle_route_conflict")
    assert align(at(noon, **on_route_2)).outcome == "matched"
    assert align(at(noon, **{**on_route_1, "vehicle_id": "V9"})).outcome == "matched"


@pytest.mark.parametrize(
    "overrides",
    [
        {"stop_id": "S-A1"},
        {"stop_id": "S-A1", "direction_id": "B"},
        {"stop_name": SQUARE, "direction_id": "B"},
        {"stop_id": "S-B", "stop_sequence": 3},
        {"stop_id": "S-B", "previous_stop_id": "S-A1"},
    ],
)
def test_every_match_passes_the_contract_location_oracle(overrides):
    oracle = EntityCatalog.model_validate(copy.deepcopy(CATALOG))

    result = align(event(**overrides))

    assert isinstance(result, Matched)
    oracle.validate_location(
        "entities.test",
        result.route_id,
        result.direction_id,
        result.stop_id,
        result.stop_sequence,
    )

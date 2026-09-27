import copy
from datetime import datetime

import pytest
from test_identity_fixtures import CATALOG, CROSSWALK, MOSCOW, catalog, crosswalk

from tramflow_ml.identity import (
    CanonicalCatalog,
    CatalogError,
    CrosswalkError,
    IdentityError,
    identity_crosswalk,
)


def test_catalog_from_dict_keeps_repeated_visits_and_same_name_stops():
    loaded = catalog()

    pattern = loaded.patterns[("route:1", "dir:0")]

    assert pattern.visits("stop:B") == (1, 3)
    assert loaded.stops["stop:A1"].name == loaded.stops["stop:A2"].name
    assert loaded.direction_ids == {"dir:0", "dir:1"}


def test_catalog_rejects_pattern_with_unknown_stop():
    payload = copy.deepcopy(CATALOG)
    payload["patterns"][0]["stop_ids"].append("stop:ghost")

    with pytest.raises(CatalogError, match="unknown stops"):
        CanonicalCatalog.from_dict(payload)


@pytest.mark.parametrize(
    ("table", "entry"),
    [
        ("stops", {"S-X": "stop:ghost"}),
        ("routes", {"9": "route:9"}),
        ("directions", {"Z": "dir:9"}),
    ],
)
def test_load_rejects_unknown_canonical_ids(table, entry):
    with pytest.raises(CrosswalkError, match=f"crosswalk.{table} maps to unknown"):
        crosswalk(**{table: entry})


def test_load_rejects_entity_version_mismatch():
    with pytest.raises(CrosswalkError, match="entity_version"):
        crosswalk(entity_version="entities.other")


def test_load_rejects_name_entry_whose_stop_is_not_on_the_pattern():
    entry = {
        "name": "Депо",
        "route_id": "route:1",
        "direction_id": "dir:0",
        "stop_id": "stop:C",
    }

    with pytest.raises(CrosswalkError, match="not on pattern"):
        crosswalk(stop_names=[entry])


def test_load_rejects_position_for_unknown_stop_and_out_of_range_coordinates():
    with pytest.raises(CrosswalkError, match="stop_positions maps to unknown"):
        crosswalk(stop_positions={"stop:ghost": {"latitude": 0, "longitude": 0}})
    with pytest.raises(CrosswalkError, match="valid range"):
        crosswalk(stop_positions={"stop:B": {"latitude": 91, "longitude": 0}})


def test_load_rejects_overlapping_vehicle_intervals():
    vehicles = copy.deepcopy(CROSSWALK["vehicles"])
    vehicles[1]["valid_from"] = "2024-03-01T11:59:59+03:00"

    with pytest.raises(CrosswalkError, match="overlap"):
        crosswalk(vehicles=vehicles)


def test_load_rejects_naive_and_inverted_vehicle_intervals():
    naive = {"vehicle_id": "V2", "route_id": "route:1", "valid_from": "2024-03-01T00:00:00"}
    inverted = {
        "vehicle_id": "V2",
        "route_id": "route:1",
        "valid_from": "2024-03-02T00:00:00+03:00",
        "valid_to": "2024-03-01T00:00:00+03:00",
    }

    with pytest.raises(CrosswalkError, match="timezone-aware"):
        crosswalk(vehicles=[naive])
    with pytest.raises(CrosswalkError, match="valid_from before valid_to"):
        crosswalk(vehicles=[inverted])


def test_load_rejects_malformed_tables_with_key_path():
    with pytest.raises(IdentityError, match=r"crosswalk.stops\['S-A1'\]"):
        crosswalk(stops={"S-A1": 7})
    with pytest.raises(IdentityError, match="crosswalk.crosswalk_version"):
        crosswalk(crosswalk_version="")


def test_assignment_at_uses_half_open_intervals():
    loaded = crosswalk()
    before_noon = datetime(2024, 3, 1, 11, 59, 59, tzinfo=MOSCOW)
    noon = datetime(2024, 3, 1, 12, 0, tzinfo=MOSCOW)
    before_service = datetime(2024, 2, 29, 23, 59, 59, tzinfo=MOSCOW)

    assert loaded.assignment_at("V1", before_noon).route_id == "route:1"
    assert loaded.assignment_at("V1", noon).route_id == "route:2"
    assert loaded.assignment_at("V1", before_service) is None
    assert loaded.assignment_at("V9", noon) is None


def test_identity_crosswalk_maps_every_catalog_id_to_itself():
    loaded = identity_crosswalk(catalog(), "identity.v1")

    assert loaded.routes == {"route:1": "route:1", "route:2": "route:2"}
    assert loaded.directions == {"dir:0": "dir:0", "dir:1": "dir:1"}
    assert set(loaded.stops) == {"stop:A1", "stop:A2", "stop:B", "stop:C"}
    assert all(source == target for source, target in loaded.stops.items())
    assert loaded.crosswalk_version == "identity.v1"

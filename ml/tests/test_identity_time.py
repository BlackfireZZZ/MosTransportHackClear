import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_identity_fixtures import MOSCOW, catalog, config, crosswalk, event

from tramflow_ml.identity import (
    AlignmentConfig,
    AlignmentError,
    ConfigError,
    SourceClock,
    Unmatched,
    UnresolvedLocalTime,
    align_event,
    align_time,
)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.calendar_v1 import service_date  # noqa: E402, I001
from contracts.data_v1 import ValidationEvent  # noqa: E402, I001


def aligned(source_event, **clock):
    clocks = {"validations": SourceClock(**clock)}
    return align_event(catalog(), crosswalk(), AlignmentConfig(clocks=clocks), source_event)


def test_naive_timestamp_is_read_in_the_source_timezone_and_emitted_in_moscow():
    naive = datetime(2024, 3, 1, 5, 0)

    result = aligned(event(stop_id="S-A1", event_at=naive, available_at=naive), timezone="UTC")

    assert result.time.event_at == datetime(2024, 3, 1, 8, 0, tzinfo=MOSCOW)
    assert result.time.event_at.tzinfo.key == "Europe/Moscow"
    assert result.time.source_event_at.tzinfo.key == "UTC"
    assert result.match.outcome == "matched"


def test_offset_is_applied_to_event_and_availability_and_recorded():
    at = datetime(2024, 3, 1, 8, 0, tzinfo=MOSCOW)
    result = aligned(
        event(stop_id="S-A1", event_at=at, available_at=at + timedelta(minutes=1)),
        offset_seconds=-3600,
    )

    assert result.time.event_at == datetime(2024, 3, 1, 7, 0, tzinfo=MOSCOW)
    assert result.time.available_at == datetime(2024, 3, 1, 7, 1, tzinfo=MOSCOW)
    assert result.time.source_event_at == at
    assert result.time.offset_seconds == -3600
    assert result.time.service_day_shifted is False


@pytest.mark.parametrize(
    ("source_at", "offset", "expected"),
    [
        (datetime(2024, 3, 1, 0, 30, tzinfo=MOSCOW), -3600, datetime(2024, 2, 29, 23, 30)),
        (datetime(2025, 1, 1, 0, 0, 10, tzinfo=MOSCOW), -20, datetime(2024, 12, 31, 23, 59, 50)),
        (datetime(2024, 4, 30, 23, 59, 30, tzinfo=MOSCOW), 60, datetime(2024, 5, 1, 0, 0, 30)),
    ],
)
def test_offset_crossing_moscow_midnight_is_recorded_not_silent(source_at, offset, expected):
    result = align_time(SourceClock(offset_seconds=offset), source_at, None)

    assert result.event_at == expected.replace(tzinfo=MOSCOW)
    assert result.service_day_shifted is True
    assert result.source_event_at == source_at
    assert service_date(result.event_at) != service_date(result.source_event_at)


def test_offset_is_applied_in_utc_across_a_source_dst_gap():
    clock = SourceClock(timezone="Europe/Berlin", offset_seconds=3600)

    result = align_time(clock, datetime(2024, 3, 31, 1, 30), None)

    assert result.event_at == datetime(2024, 3, 31, 4, 30, tzinfo=MOSCOW)


def test_missing_availability_needs_a_configured_lag():
    without_lag = aligned(event(stop_id="S-A1", available_at=None))
    with_lag = aligned(event(stop_id="S-A1", available_at=None), availability_lag_seconds=60)

    assert without_lag.match == Unmatched("availability_missing")
    assert without_lag.time.available_at is None
    assert with_lag.match.outcome == "matched"
    assert with_lag.time.available_at == with_lag.time.event_at + timedelta(seconds=60)


def test_availability_before_event_is_unmatched_not_clamped():
    at = datetime(2024, 3, 1, 8, 0, tzinfo=MOSCOW)

    result = aligned(event(stop_id="S-A1", event_at=at, available_at=at - timedelta(seconds=1)))

    assert result.match == Unmatched("availability_precedes_event")
    assert result.time.available_at == at - timedelta(seconds=1)


def test_missing_clock_and_bad_config_raise():
    with pytest.raises(AlignmentError, match="no clock configured"):
        align_event(catalog(), crosswalk(), AlignmentConfig(), event(stop_id="S-A1"))
    with pytest.raises(ConfigError, match="timezone"):
        SourceClock(timezone="Mars/Olympus")
    with pytest.raises(ConfigError, match="geo_tolerance_metres"):
        AlignmentConfig(geo_tolerance_metres=0)


def test_config_from_dict_reads_clocks_and_thresholds():
    payload = {
        "clocks": {"validations": {"timezone": "UTC", "offset_seconds": -5}},
        "geo_tolerance_metres": 25,
        "stop_sequence_base": 1,
    }

    loaded = AlignmentConfig.from_dict(payload)

    assert loaded.clocks == {"validations": SourceClock(timezone="UTC", offset_seconds=-5)}
    assert loaded.geo_tolerance_metres == 25.0
    assert loaded.gps_staleness_seconds == 60.0
    assert loaded.stop_sequence_base == 1


def test_aligned_match_builds_a_valid_contract_row_and_leaves_the_source_untouched():
    source = event(stop_id="S-A1", vehicle_id="V1")
    snapshot = event(stop_id="S-A1", vehicle_id="V1")

    result = align_event(catalog(), crosswalk(), config(), source)

    row = ValidationEvent(
        schema_version="data.v1",
        entity_version=result.entity_version,
        route_id=result.match.route_id,
        direction_id=result.match.direction_id,
        stop_id=result.match.stop_id,
        event_id=result.event_id,
        source_version=result.crosswalk_version,
        event_at=result.time.event_at,
        available_at=result.time.available_at,
        synthetic=True,
        vehicle_id="V1",
        stop_sequence=result.match.stop_sequence,
        target="synthetic_boardings",
        unit="event_count",
    )
    assert row.visible_at(result.time.available_at)
    assert source == snapshot
    with pytest.raises(AttributeError):
        source.stop_id = "S-B"


@pytest.mark.parametrize(
    ("wall_time", "reason"),
    [
        (datetime(2024, 3, 31, 2, 30), "nonexistent_local_time"),
        (datetime(2024, 10, 27, 2, 30), "ambiguous_local_time"),
    ],
)
def test_naive_time_in_source_dst_gap_or_overlap_is_never_guessed(wall_time, reason):
    clock = SourceClock(timezone="Europe/Berlin")

    unresolved = align_time(clock, wall_time, None)
    result = aligned(
        event(stop_id="S-A1", event_at=wall_time, available_at=wall_time), timezone="Europe/Berlin"
    )

    assert unresolved == UnresolvedLocalTime(wall_time, None, "event_at", reason)
    assert result.time == UnresolvedLocalTime(wall_time, wall_time, "event_at", reason)
    assert result.match == Unmatched(reason)


def test_unresolvable_availability_is_reported_on_its_own_field():
    clock = SourceClock(timezone="Europe/Berlin")
    event_at = datetime(2024, 3, 31, 1, 30)
    gap = datetime(2024, 3, 31, 2, 30)

    result = align_time(clock, event_at, gap)

    assert result == UnresolvedLocalTime(event_at, gap, "available_at", "nonexistent_local_time")


def test_unresolvable_fix_time_blocks_only_gps_joins():
    gap = datetime(2024, 3, 31, 2, 30)
    at = datetime(2024, 3, 31, 1, 30)
    by_gps = event(
        source_id="telemetry",
        event_at=at,
        available_at=at,
        latitude=55.75,
        longitude=37.6,
        fix_at=gap,
    )
    by_id = event(source_id="telemetry", event_at=at, available_at=at, stop_id="S-A1", fix_at=gap)
    berlin = AlignmentConfig(clocks={"telemetry": SourceClock(timezone="Europe/Berlin")})

    assert align_event(catalog(), crosswalk(), berlin, by_gps).match == Unmatched(
        "nonexistent_local_time"
    )
    assert align_event(catalog(), crosswalk(), berlin, by_id).match.kind == "exact_id"

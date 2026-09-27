import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from test_identity_fixtures import EVENT_AT, SQUARE, catalog, config, crosswalk, event

from tramflow_ml.identity import (
    AlignmentConfig,
    CanonicalCatalog,
    Crosswalk,
    GeoPoint,
    IdentityError,
    SourceClock,
    SourceEvent,
    align_events,
    identity_crosswalk,
    quality_report,
)
from tramflow_ml.synthetic import STOP_IDS, SyntheticConfig, generate_dataset

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.data_v1 import EntityCatalog  # noqa: E402, I001


def test_report_counts_and_rates_add_up_per_stream_in_deterministic_order():
    validations = [
        event(event_id="v1", stop_id="S-A1"),
        event(event_id="v2", stop_id="S-A1", direction_id="B"),
        event(event_id="v3", stop_id="S-B", stop_sequence=1),
        event(event_id="v4", stop_name=SQUARE, direction_id="B"),
        event(event_id="v5", route_id="7", stop_id="S-A1"),
        event(event_id="v6", stop_id="S-C", route_id="2", stop_sequence=0, vehicle_id="V1"),
        event(event_id="v7", stop_name=SQUARE),
        event(event_id="v8", stop_id="S-B"),
    ]
    telemetry = [
        event(event_id="t1", source_id="telemetry", latitude=55.75, longitude=37.6),
        event(
            event_id="t2",
            source_id="telemetry",
            latitude=55.75,
            longitude=37.6,
            fix_at=EVENT_AT - timedelta(minutes=5),
        ),
        event(event_id="t3", source_id="telemetry", latitude=55.78, longitude=37.6),
        event(event_id="t4", source_id="telemetry", stop_id="S-A1", available_at=None),
    ]
    aligned = align_events(catalog(), crosswalk(), config(), [*validations, *telemetry])

    report = quality_report(crosswalk(), aligned)

    assert report.to_dict() == {
        "entity_version": "entities.test",
        "crosswalk_version": "crosswalk.test.1",
        "streams": [
            {
                "source_id": "telemetry",
                "total": 4,
                "matched": {"total": 1, "by_kind": {"geo_nearest": 1}},
                "unmatched": {
                    "total": 2,
                    "by_reason": {"availability_missing": 1, "no_stop_within_tolerance": 1},
                },
                "ambiguous": 0,
                "stale": 1,
                "service_day_shifted": 0,
                "rates": {"ambiguous": 0.0, "matched": 0.25, "stale": 0.25, "unmatched": 0.5},
            },
            {
                "source_id": "validations",
                "total": 8,
                "matched": {"total": 4, "by_kind": {"exact_id": 3, "name_route_direction": 1}},
                "unmatched": {
                    "total": 2,
                    "by_reason": {"unknown_route": 1, "vehicle_route_conflict": 1},
                },
                "ambiguous": 2,
                "stale": 0,
                "service_day_shifted": 0,
                "rates": {"ambiguous": 0.25, "matched": 0.5, "stale": 0.0, "unmatched": 0.25},
            },
        ],
    }
    for stream in report.streams:
        assert stream.matched + stream.unmatched + stream.ambiguous + stream.stale == stream.total
        assert sum(stream.rates().values()) == pytest.approx(1.0)


def test_report_counts_service_day_shifts_and_rejects_foreign_crosswalk_versions():
    shifted_clock = AlignmentConfig(clocks={"validations": SourceClock(offset_seconds=-9 * 3600)})
    aligned = align_events(catalog(), crosswalk(), shifted_clock, [event(stop_id="S-A1")])

    report = quality_report(crosswalk(), aligned)

    assert report.streams[0].service_day_shifted == 1
    with pytest.raises(IdentityError, match="another crosswalk"):
        quality_report(crosswalk(crosswalk_version="crosswalk.test.2"), aligned)


def _source_events(rows, source_id, **overrides):
    return [
        SourceEvent(
            event_id=row["event_id"],
            source_id=source_id,
            event_at=datetime.fromisoformat(row["event_at"]),
            available_at=datetime.fromisoformat(row["available_at"]),
            route_id=row["route_id"],
            direction_id=row["direction_id"],
            stop_sequence=row["stop_sequence"],
            vehicle_id=row["vehicle_id"],
            **{key: value(row) for key, value in overrides.items()},
        )
        for row in rows
    ]


def test_tiny_synthetic_fixture_joins_completely_through_the_identity_crosswalk(tmp_path):
    output = tmp_path / "tiny"
    generate_dataset(
        SyntheticConfig(events=64, start=date(2024, 1, 1), end=date(2024, 2, 1)), output
    )
    entities = json.loads((output / "entities.json").read_text())
    rows = {
        name: [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
        for name in ("validations", "telemetry")
    }
    canonical = CanonicalCatalog.from_dict(entities)
    positions = {
        stop_id: GeoPoint(55.75 + index * 0.01, 37.60 + index * 0.01)
        for index, stop_id in enumerate(STOP_IDS)
    }
    identity = identity_crosswalk(canonical, "identity.v1")
    geo_aware = Crosswalk(**{**vars(identity), "stop_positions": positions})
    events = [
        *_source_events(rows["validations"], "validations", stop_id=lambda row: row["stop_id"]),
        *_source_events(
            rows["telemetry"],
            "telemetry",
            latitude=lambda row: row["latitude"],
            longitude=lambda row: row["longitude"],
        ),
    ]

    aligned = align_events(canonical, geo_aware, config(), events)
    report = quality_report(geo_aware, aligned).to_dict()

    assert [stream["total"] for stream in report["streams"]] == [
        len(rows["telemetry"]),
        len(rows["validations"]),
    ]
    assert report["streams"][0]["matched"] == {
        "total": len(rows["telemetry"]),
        "by_kind": {"geo_nearest": len(rows["telemetry"])},
    }
    assert report["streams"][1]["matched"] == {
        "total": len(rows["validations"]),
        "by_kind": {"exact_id": len(rows["validations"])},
    }
    assert all(stream["rates"]["matched"] == 1.0 for stream in report["streams"])
    assert all(stream["ambiguous"] == 0 and stream["stale"] == 0 for stream in report["streams"])
    oracle = EntityCatalog.model_validate(entities)
    by_id = {row["event_id"]: row for stream in rows.values() for row in stream}
    for item in aligned:
        row = by_id[item.event_id]
        assert item.match.stop_id == row["stop_id"]
        assert item.match.stop_sequence == row["stop_sequence"]
        assert item.time.event_at == datetime.fromisoformat(row["event_at"])
        assert item.time.service_day_shifted is False
        oracle.validate_location(
            item.entity_version,
            item.match.route_id,
            item.match.direction_id,
            item.match.stop_id,
            item.match.stop_sequence,
        )

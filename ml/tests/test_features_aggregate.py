import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from tramflow_ml.features import (
    MOSCOW,
    AggregateCell,
    Bucket,
    CoverageCalendar,
    EntityKey,
    FeatureError,
    Observation,
    aggregate,
)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from contracts.data_v1 import ObservedAggregate  # noqa: E402, I001

TARGET = "synthetic_boardings"
UNIT = "event_count"
DAYS_IN_MAY = 31

FIRST = EntityKey("route-1", "dir-0", "stop-1")
SECOND = EntityKey("route-1", "dir-1", "stop-1")

# (entity, Moscow wall time) laid out by hand so the expected totals can be counted by eye.
HAND_EVENTS = (
    (FIRST, "2024-05-06T08:05"),
    (FIRST, "2024-05-06T08:55"),
    (FIRST, "2024-05-06T09:00"),
    (SECOND, "2024-05-06T08:30"),
    (FIRST, "2024-05-07T18:10"),
)


def moscow(text):
    return datetime.fromisoformat(text).replace(tzinfo=MOSCOW)


def observation(entity, text, minutes=1):
    event_at = moscow(text)
    return Observation(entity, event_at, event_at + timedelta(minutes=minutes), TARGET, UNIT)


@pytest.fixture
def observations():
    return tuple(observation(entity, text) for entity, text in HAND_EVENTS)


@pytest.fixture
def coverage():
    return CoverageCalendar.from_range(date(2024, 5, 6), date(2024, 5, 9)).complete()


def oracle_totals(granularity):
    """Totals counted straight from HAND_EVENTS, without touching the feature code."""
    keys = {
        "hourly": lambda text: text[:13],
        "daily": lambda text: text[:10],
        "monthly": lambda text: text[:7],
    }
    return Counter((entity, keys[granularity](text)) for entity, text in HAND_EVENTS)


def test_hourly_cells_match_hand_counted_totals(observations, coverage):
    index = aggregate(observations, "hourly", coverage, TARGET, UNIT)

    counted = {
        (cell.entity, cell.bucket.start.isoformat()[:13]): cell.value for cell in index.cells()
    }

    assert counted == dict(oracle_totals("hourly"))
    assert counted[(FIRST, "2024-05-06T08")] == 2
    assert index.total == len(HAND_EVENTS)


@pytest.mark.parametrize("granularity", ["hourly", "daily", "monthly"])
def test_every_granularity_conserves_the_total(observations, coverage, granularity):
    index = aggregate(observations, granularity, coverage, TARGET, UNIT)

    assert sum(cell.value for cell in index.cells()) == sum(oracle_totals(granularity).values())
    assert index.total == len(HAND_EVENTS)


def test_daily_and_monthly_group_the_same_events(observations, coverage):
    daily = aggregate(observations, "daily", coverage, TARGET, UNIT)
    monthly = aggregate(observations, "monthly", coverage, TARGET, UNIT)

    assert daily.cell(FIRST, moscow("2024-05-06T00:00")).value == 3
    assert daily.cell(FIRST, moscow("2024-05-07T00:00")).value == 1
    assert daily.cell(SECOND, moscow("2024-05-06T00:00")).value == 1
    assert monthly.cell(FIRST, moscow("2024-05-01T00:00")).value == 4
    assert monthly.cell(SECOND, moscow("2024-05-01T00:00")).value == 1


def test_partial_month_coverage_stays_visible(observations, coverage):
    index = aggregate(observations, "monthly", coverage, TARGET, UNIT)

    cell = index.cell(FIRST, moscow("2024-05-01T00:00"))

    assert (cell.covered_units, cell.total_units) == (3, DAYS_IN_MAY)
    assert cell.coverage == "observed"


def test_cells_satisfy_the_observed_aggregate_contract(observations, coverage):
    index = aggregate(observations, "hourly", coverage, TARGET, UNIT)

    for cell in (*index.cells(), index.cell(FIRST, moscow("2024-05-20T08:00"))):
        model = ObservedAggregate(
            schema_version="data.v1",
            entity_version="entities.v1",
            source_version="synthetic.v1",
            route_id=cell.entity.route_id,
            direction_id=cell.entity.direction_id,
            stop_id=cell.entity.stop_id,
            target=TARGET,
            unit=UNIT,
            synthetic=True,
            bucket_start=cell.bucket.start,
            bucket_end=cell.bucket.end,
            coverage=cell.coverage,
            value=cell.value,
        )
        assert (model.coverage == "missing") == (model.value is None)


def test_cells_are_ordered_reproducibly(observations, coverage):
    index = aggregate(observations, "hourly", coverage, TARGET, UNIT)

    keys = [(cell.entity, cell.bucket.start) for cell in index.cells()]

    assert keys == sorted(keys)
    assert index.cells() == aggregate(observations, "hourly", coverage, TARGET, UNIT).cells()


def test_reversed_input_order_gives_the_same_cells(observations, coverage):
    forward = aggregate(observations, "hourly", coverage, TARGET, UNIT)
    backward = aggregate(tuple(reversed(observations)), "hourly", coverage, TARGET, UNIT)

    assert [cell.to_dict() for cell in forward.cells()] == [
        cell.to_dict() for cell in backward.cells()
    ]


def test_other_targets_and_units_are_not_counted(observations, coverage):
    foreign = Observation(
        FIRST, moscow("2024-05-06T08:10"), moscow("2024-05-06T08:11"), "validation_count", UNIT
    )

    index = aggregate((*observations, foreign), "hourly", coverage, TARGET, UNIT)

    assert index.total == len(HAND_EVENTS)
    assert index.cell(FIRST, moscow("2024-05-06T08:00")).value == 2


def test_events_on_an_uncovered_date_are_refused(observations):
    calendar = CoverageCalendar.from_range(
        date(2024, 5, 6), date(2024, 5, 9), gaps=[date(2024, 5, 7)]
    ).complete()

    with pytest.raises(FeatureError, match="2024-05-07"):
        aggregate(observations, "hourly", calendar, TARGET, UNIT)


def test_aggregation_does_not_mutate_its_input(observations, coverage):
    before = tuple(
        (obs.entity, obs.event_at, obs.available_at, obs.target, obs.unit) for obs in observations
    )

    aggregate(observations, "daily", coverage, TARGET, UNIT)

    after = tuple(
        (obs.entity, obs.event_at, obs.available_at, obs.target, obs.unit) for obs in observations
    )
    assert before == after


def test_availability_cannot_precede_the_event():
    with pytest.raises(FeatureError, match="available_at"):
        observation(FIRST, "2024-05-06T08:05", minutes=-1)


def test_a_cell_cannot_claim_a_value_it_does_not_have():
    """The invariant the layer is about, enforced rather than only documented."""
    bucket = Bucket(moscow("2024-05-06T08:00"), moscow("2024-05-06T09:00"))

    with pytest.raises(FeatureError, match="missing requires no value"):
        AggregateCell(FIRST, bucket, None, "observed", 1, 1)
    with pytest.raises(FeatureError, match="missing requires no value"):
        AggregateCell(FIRST, bucket, 3, "missing", 0, 1)
    with pytest.raises(FeatureError, match="no available date"):
        AggregateCell(FIRST, bucket, 3, "observed", 0, 1)
    with pytest.raises(FeatureError, match="within"):
        AggregateCell(FIRST, bucket, 3, "observed", 2, 1)
    with pytest.raises(FeatureError, match="at least one civil date"):
        AggregateCell(FIRST, bucket, None, "missing", 0, 0)
    with pytest.raises(FeatureError, match="cannot be negative"):
        AggregateCell(FIRST, bucket, -1, "observed", 1, 1)


def test_a_valid_cell_still_constructs():
    bucket = Bucket(moscow("2024-05-06T08:00"), moscow("2024-05-06T09:00"))

    assert AggregateCell(FIRST, bucket, 0, "observed", 1, 1).unit_ratio == 1.0
    assert AggregateCell(FIRST, bucket, None, "missing", 0, 1).unit_ratio == 0.0

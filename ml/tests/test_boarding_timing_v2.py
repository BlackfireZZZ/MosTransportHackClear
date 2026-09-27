from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.boarding.composition import RouteEvidence
from tramflow_ml.boarding.sequence import CycleTemplate, SequenceConfig, infer_sequence
from tramflow_ml.boarding.timing_v2 import (
    TimingConfig,
    clock_evidence,
    group_times,
    reconstruct_session,
)


def evidence() -> RouteEvidence:
    template = CycleTemplate(
        "1",
        ("p",) * 4,
        ("a", "b", "c", "d"),
        ("0",) * 4,
        (400, 600, 200, 800),
        (False,) * 4,
        "test",
    )
    return RouteEvidence(
        template,
        tuple({"stop_id": s, "direction": "0", "boardable": True} for s in template.stop_ids),
        ((36000, 36600),) * 4,
        ("historical_exact_day", "proxy_transfer") * 2,
        (),
        (),
        True,
    )


def test_first_validation_is_not_shifted_by_passenger_count() -> None:
    groups, times, counts = group_times(np.array([0.0, 10, 30, 60, 200, 210]), TimingConfig())
    assert times.tolist() == [0, 200]
    assert counts.tolist() == [4, 2]
    assert groups.tolist() == [0, 0, 0, 0, 1, 1]
    assert group_times(np.array([0.0, 10, 30, 60]), TimingConfig(timestamp="mean"))[1][0] == 25


def test_equal_departure_clocks_do_not_prefer_proxy_source() -> None:
    result = clock_evidence(np.array([36000 - 10800 + 180.0]), evidence(), TimingConfig())
    np.testing.assert_array_equal(result, np.repeat(result[:, :1], 4, axis=1))


def test_actual_local_times_do_not_accumulate_previous_residual() -> None:
    e = evidence()
    result = infer_sequence(
        [0, 140, 430],
        e.template,
        SequenceConfig(noise_seconds=25, relative_sigma=0, max_steps=3),
        edge_seconds=np.array([120.0, 300, 360, 240]),
        log_emissions=np.array(
            [
                [0, -np.inf, -np.inf, -np.inf],
                [-np.inf, 0, -np.inf, -np.inf],
                [-np.inf, -np.inf, 0, -np.inf],
            ]
        ),
    )
    assert result.path_steps.tolist() == [-1, 1, 1]
    assert result.residual_seconds.tolist() == [0, 20, -10]


def test_skips_sum_schedule_edges_without_inventing_bursts() -> None:
    e = evidence()
    result = infer_sequence(
        [0, 420],
        e.template,
        SequenceConfig(max_steps=4),
        edge_seconds=np.array([120.0, 300, 360, 240]),
        log_emissions=np.array([[0, -np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, 0, -np.inf]]),
    )
    assert result.path_steps.tolist() == [-1, 2]
    assert result.expected_seconds.tolist() == [0, 420]


@pytest.mark.parametrize("values", [[0, 1, 1, 1], [1, 2], [1, 2, np.nan, 3]])
def test_invalid_external_edge_durations_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError, match="edge seconds"):
        infer_sequence([0, 120], evidence().template, edge_seconds=np.array(values))


def test_original_hours_and_all_events_survive_first_burst_inference() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["a", "b", "c"],
            "event_at": ["2025-01-15 09:59:58", "2025-01-15 10:00:03", "2025-01-15 10:00:05"],
            "second": [35998.0, 36003, 36005],
            "route": ["1"] * 3,
            "group": ["g"] * 3,
            "vehicle_key": ["v"] * 3,
            "identity_conflict": [False] * 3,
        }
    )
    result = reconstruct_session(frame, evidence(), np.array([120.0, 300, 360, 240]))
    assert len(result["events"]) == 3
    mass = pd.DataFrame(result["soft"]).groupby("event_hour").expected_count.sum()
    assert mass.to_dict() == pytest.approx({"2025-01-15 09": 1, "2025-01-15 10": 2})
    assert all(not row["training_eligible"] for row in result["events"])
    assert all(a["trip_id"] is None for a in result["anchors"])


def test_incomplete_departure_coverage_disables_clocks_for_all_states() -> None:
    e = replace(evidence(), departures=((), (1,), (1,), (1,)))
    np.testing.assert_array_equal(clock_evidence(np.array([10.0]), e, TimingConfig()), [[0] * 4])

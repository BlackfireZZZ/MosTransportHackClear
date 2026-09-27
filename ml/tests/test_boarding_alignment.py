from itertools import product

import pytest

from tramflow_ml.boarding.alignment import (
    Anchor,
    DecodeConfig,
    DelayScenario,
    Pattern,
    decode,
    greedy_baseline,
    no_stop_baseline,
)


def pattern(*, verified: bool = True) -> Pattern:
    return Pattern(
        "p", "outbound", ("a", "b", "a", "c"), (10, 10, 10), historical_verified=verified
    )


def test_exact_frontier_matches_independent_brute_force() -> None:
    for timestamps in ((0, 0, 10), (0, 20, 30), (0, 11, 20), (7, 14, 24)):
        p = pattern()
        scenario = DelayScenario("interval", 0, 2)
        actual = decode(timestamps, (p,), scenarios=(scenario,), config=DecodeConfig(top_k=1000))
        expected = {}
        for visits in product((None, 0, 1, 2, 3), repeat=len(timestamps)):
            pairs = [(t, v) for t, v in zip(timestamps, visits, strict=True) if v is not None]
            sequence = [v for _, v in pairs]
            if sequence != sorted(sequence):
                continue
            lower = max((t - 2 - v * 10 for t, v in pairs), default=float("-inf"))
            upper = min((t - v * 10 for t, v in pairs), default=float("inf"))
            if lower > upper:
                continue
            score = 5 * visits.count(None) + sum(
                max(0, b - a - 1) for a, b in zip(sequence, sequence[1:], strict=False)
            )
            expected[visits] = (score, (lower, upper) if pairs else None)
        assert {c.visits: (c.score, c.start_interval) for c in actual.candidates} == expected
        assert actual.feasible_paths == len(expected)
        assert not actual.approximation


def test_repeated_stop_ids_are_distinct_and_skipped_visits_are_latent() -> None:
    result = decode((0, 20, 30), (pattern(),), anchors=(Anchor("p", 0, 0, "GPS", True),))
    best = result.candidates[0]
    assert best.visits == (0, 2, 3)
    assert best.latent_visits == (1,)
    assert result.stop_ids == ("a", "a", "c")
    assert result.status == "anchored_weak_labels"
    assert not result.calibrated


def test_first_event_is_not_forced_to_first_stop() -> None:
    result = decode((200, 210), (pattern(),), config=DecodeConfig(top_k=1))
    assert result.stop_ids == (None, None)
    assert "ambiguous_alignment" in result.reasons
    assert result.alternatives_omitted > 0
    assert result.candidates[0].open_left and result.candidates[0].open_right
    all_paths = decode((200, 210), (pattern(),), config=DecodeConfig(top_k=100))
    assert (2, 3) in {c.visits for c in all_paths.candidates if c.score == 0}


def test_many_to_one_and_split_are_event_level() -> None:
    result = decode((0, 0, 10, 10), (pattern(),), anchors=(Anchor("p", 0, 1, "GPS", True),))
    assert result.candidates[0].visits == (1, 1, 2, 2)
    assert result.stop_ids == ("b", "b", "a", "a")
    assert len(result.candidates[0].visits) == 4


def test_competing_directions_preserved_before_top_k() -> None:
    p = pattern()
    reverse = Pattern(
        "reverse", "inbound", tuple(reversed(p.stop_ids)), (10, 10, 10), historical_verified=True
    )
    result = decode((0, 10, 20, 30), (p, reverse), config=DecodeConfig(top_k=1))
    assert "ambiguous_alignment" in result.reasons
    assert result.stop_ids == (None,) * 4
    assert result.alternatives_omitted > 0


@pytest.mark.parametrize("verified,independent", [(False, True), (True, False), (False, False)])
def test_verification_and_anchor_both_required(verified: bool, independent: bool) -> None:
    result = decode(
        (0, 10),
        (pattern(verified=verified),),
        anchors=(Anchor("p", 0, 1, "observed_or_inferred", independent),),
    )
    assert result.status == "relative_only"
    assert result.stop_ids == (None, None)


def test_delay_interval_and_clock_scenarios_do_not_change_original_events() -> None:
    timestamps = (100, 115)
    p = Pattern("p", "out", ("a", "b"), (10,), historical_verified=True)
    anchors = (Anchor("p", 0, 0, "GPS", True), Anchor("p", 1, 1, "GPS", True))
    result = decode(
        timestamps, (p,), anchors=anchors, scenarios=(DelayScenario("delayed", 0, 5, 7),)
    )
    assert result.candidates[0].start_interval == (93, 93)
    assert timestamps == (100, 115)
    impossible = decode(timestamps, (p,), anchors=anchors)
    assert impossible.status == "abstained"
    assert impossible.reasons == ("pattern_mismatch",)


def test_dwell_and_midnight_relative_timestamps() -> None:
    p = Pattern("p", "out", ("a", "b"), (10,), dwell_seconds=5, historical_verified=True)
    result = decode((86398, 86402, 86413), (p,), anchors=(Anchor("p", 0, 0, "GPS", True),))
    assert result.candidates[0].visits == (0, 0, 1)
    assert result.stop_ids == ("a", "a", "b")


@pytest.mark.parametrize(
    "config",
    [DecodeConfig(max_events=1), DecodeConfig(max_states=1), DecodeConfig(max_transitions=1)],
)
def test_budget_exhaustion_never_returns_partial_best(config: DecodeConfig) -> None:
    result = decode((0, 10), (pattern(),), config=config)
    assert result.status == "undecoded"
    assert result.candidates == ()
    assert result.stop_ids == (None, None)
    assert result.reasons == ("budget_exhausted",)


def test_outlier_preserves_event_mass_and_does_not_reset_order() -> None:
    result = decode((0, 7, 10), (pattern(),), anchors=(Anchor("p", 0, 0, "GPS", True),))
    assert result.candidates[0].visits == (0, None, 1)
    assert result.stop_ids == ("a", None, "b")


def test_contradictory_anchors_abstain() -> None:
    result = decode(
        (0, 10),
        (pattern(),),
        anchors=(Anchor("p", 0, 0, "GPS", True), Anchor("p", 0, 1, "GPS", True)),
    )
    assert result.reasons == ("pattern_mismatch",)
    assert not result.candidates


def test_b0_b2_do_not_invent_anchors() -> None:
    assert no_stop_baseline(3).stop_ids == (None,) * 3
    assert greedy_baseline((0, 10), pattern()).reasons == ("no_anchor",)
    result = greedy_baseline((0, 10, 20), pattern(), anchor=Anchor("p", 1, 2, "GPS", True))
    assert result.candidates[0].visits == (1, 2, 3)
    assert result.stop_ids == ("b", "a", "c")


def test_top_k_does_not_hide_ties_under_an_anchor() -> None:
    p = Pattern("p", "out", ("a", "b"), (0,), historical_verified=True)
    result = decode(
        (0, 0), (p,), anchors=(Anchor("p", 0, 0, "GPS", True),), config=DecodeConfig(top_k=1)
    )
    assert "ambiguous_alignment" in result.reasons
    assert result.stop_ids == (None, None)


@pytest.mark.parametrize("timestamps", [(1, 0), (float("nan"),), (float("inf"),)])
def test_invalid_timestamps_rejected(timestamps: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        decode(timestamps, (pattern(),))


def test_invalid_config_pattern_and_anchor_rejected() -> None:
    with pytest.raises(ValueError):
        DecodeConfig(top_k=0)
    with pytest.raises(ValueError):
        DecodeConfig(skip_cost=float("nan"))
    with pytest.raises(ValueError):
        Pattern("p", "out", ("a", "b"), ())
    with pytest.raises(ValueError):
        DelayScenario("bad", 3, 1)
    with pytest.raises(ValueError):
        decode((0,), (pattern(),), anchors=(Anchor("p", 2, 0, "GPS"),))
    with pytest.raises(ValueError):
        decode((0,), (pattern(), pattern()))
    assert decode((), (pattern(),)).status == "empty"
    assert decode((0,), ()).reasons == ("pattern_mismatch",)

from datetime import date, timedelta

import pandas as pd
import pytest

from tramflow_ml.competition_shift import detect_level_shifts
from tramflow_ml.decision_guard import Evaluation, ShiftSignal, review_switch


def history(origin: date, *, persistent: bool = True) -> pd.DataFrame:
    start = origin - timedelta(days=42)
    rows = []
    for day in range(42):
        current = start + timedelta(days=day)
        elevated = day >= 28 and (persistent or day < 35)
        for route in (1, 5):
            for hour in range(24):
                rows.append((route, current, hour, 2 if route == 1 and elevated else
                             1 if route == 1 else 0))
    return pd.DataFrame(rows, columns=["route", "date", "hour", "boardings"])


def test_detector_requires_persistent_complete_pre_origin_shift():
    origin = date(2025, 9, 1)
    source = history(origin)
    signals = detect_level_shifts(source, origin)
    assert len(signals) == 1
    assert signals[0].entity == "1" and signals[0].value == 2
    changed = pd.concat([source, source.assign(date=origin, boardings=1000)])
    assert detect_level_shifts(changed, origin) == signals
    assert detect_level_shifts(history(origin, persistent=False), origin) == ()
    missing = source.drop(source[(source.route == 1) & (source.hour == 3)].index[0])
    assert detect_level_shifts(missing, origin) == ()


def test_guard_does_not_evaluate_without_a_long_window_or_shift():
    evaluated = []

    def evaluate(model: str, origin: date) -> Evaluation:
        evaluated.append((model, origin))
        return Evaluation(.8, {"1": .8})

    kwargs = dict(origin=date(2025, 10, 1), first_review_origin=date(2025, 1, 1),
                  incumbent="long", candidates=("short", "catboost"),
                  evaluate=evaluate)
    assert review_switch(**kwargs, incumbent_window_days=7,
                         detect=lambda _: (ShiftSignal("1", "up", 1.4, 140, 100),)
                         ).status == "not_long_window"
    assert review_switch(**kwargs, incumbent_window_days=28,
                         detect=lambda _: ()).status == "no_shift"
    assert evaluated == []


def test_guard_requires_completed_comparable_windows_and_both_score_views():
    origin = date(2025, 11, 1)
    previous = origin - timedelta(days=61)
    older = previous - timedelta(days=61)
    calls = []

    def detect(day: date) -> tuple[ShiftSignal, ...]:
        if day in (origin, previous, older):
            return (ShiftSignal("1", "up", 1.5, 150, 100),)
        return ()

    def evaluate(model: str, day: date) -> Evaluation:
        calls.append((model, day))
        if model == "long":
            return Evaluation(.8, {"1": .7})
        if model == "short":
            return Evaluation(.81, {"1": .72})
        return Evaluation(.83, {"1": .68})

    report = review_switch(origin=origin, first_review_origin=date(2025, 4, 1),
                           incumbent="long", candidates=("short", "catboost"),
                           incumbent_window_days=28, detect=detect, evaluate=evaluate)
    assert report.status == "switch_supported" and report.selected_model == "short"
    assert report.comparable_origins == (previous.isoformat(), older.isoformat())
    assert {day for _, day in calls} == {previous, older, older - timedelta(days=61)}
    assert all(day + timedelta(days=60) < origin for _, day in calls)
    assert report.reviews[1].eligible is False
    insufficient = review_switch(origin=origin, first_review_origin=previous,
                                 incumbent="long", candidates=("short",),
                                 incumbent_window_days=28, detect=detect, evaluate=evaluate)
    assert insufficient.status == "insufficient_evidence"
    rejected = review_switch(origin=origin, first_review_origin=older,
                             incumbent="long", candidates=("catboost",),
                             incumbent_window_days=28, detect=detect, evaluate=evaluate)
    assert rejected.status == "candidate_rejected"
    assert rejected.selected_model == "long"
    with pytest.raises(ValueError, match="distinct"):
        review_switch(origin=origin, first_review_origin=older, incumbent="long",
                      candidates=("long",), incumbent_window_days=28,
                      detect=detect, evaluate=evaluate)


def test_guard_does_not_match_hour_shape_to_level_shift():
    origin = date(2025, 11, 1)

    def detect(day: date) -> tuple[ShiftSignal, ...]:
        if day == origin:
            return (ShiftSignal("1", "changed", .1, 100, 100,
                                "hour_shape", "total_variation"),)
        return (ShiftSignal("1", "up", 1.5, 150, 100),)

    report = review_switch(
        origin=origin, first_review_origin=date(2025, 5, 1),
        incumbent="long", candidates=("catboost",), incumbent_window_days=28,
        detect=detect, evaluate=lambda _model, _origin: Evaluation(.8, {"1": .8}),
    )
    assert report.status == "insufficient_evidence"
    assert report.comparable_origins == ()
    assert report.selected_model == "long"

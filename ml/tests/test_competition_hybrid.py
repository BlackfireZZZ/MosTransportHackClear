import sys
import types
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.competition import LABEL_ROUTES, feature_frame
from tramflow_ml.competition_hybrid import bounded_hybrid, hybrid_predict
from tramflow_ml.competition_limits import assess_limits
from tramflow_ml.competition_shift import (
    ShapeShiftPolicy,
    detect_hour_shape_shifts,
    detect_level_shifts,
)
from tramflow_ml.decision_guard import ShiftSignal
from tramflow_ml.route_models.features import OFF


def dense_days(start: date, days: int) -> pd.DataFrame:
    return pd.DataFrame(
        (
            (route, start + timedelta(days=offset), hour,
             0 if route == 5 else route + hour + offset)
            for route in LABEL_ROUTES for offset in range(days) for hour in range(24)
        ),
        columns=["route", "date", "hour", "boardings"],
    )


def test_bounded_hybrid_limits_the_ensemble_adjustment():
    anchor = np.array([100.0, 100.0, 0.0])
    actual = bounded_hybrid(anchor, np.array([10.0, -10.0, 4.0]),
                            np.array([10.0, -10.0, -4.0]))
    assert actual.tolist() == pytest.approx([102.5, 97.5, 0])
    assert bounded_hybrid(anchor, np.array([10.0, -10.0, 4.0])).tolist() == pytest.approx(
        [102.5, 97.5, 0]
    )
    with pytest.raises(ValueError, match="align"):
        bounded_hybrid(anchor, np.zeros(2))


def test_hybrid_uses_only_pre_origin_labels_and_recency_weights(monkeypatch):
    fits = []

    class FakeCatBoost:
        def __init__(self, **_config):
            pass

        def fit(self, features, target, sample_weight=None):
            fits.append((len(features), len(target), sample_weight))

        def predict(self, features):
            return np.zeros(len(features))

    monkeypatch.setitem(sys.modules, "catboost", types.SimpleNamespace(
        CatBoostRegressor=FakeCatBoost
    ))
    source = dense_days(date(2025, 1, 1), 150)
    origin = date(2025, 5, 1)
    past = source.loc[source.date < origin]
    first = hybrid_predict(past, feature_frame(source, origin),
                           "catboost_daily_recency_ensemble_50")
    changed = source.copy()
    changed.loc[changed.date >= origin, "boardings"] += 10000
    second = hybrid_predict(past, feature_frame(changed, origin),
                            "catboost_daily_recency_ensemble_50")
    np.testing.assert_array_equal(first, second)
    assert fits[0][0] == fits[1][0] == 610
    assert fits[0][2] is None
    assert fits[1][2] is not None
    assert np.isfinite(first).all() and (first >= 0).all()
    with pytest.raises(ValueError, match="future labels"):
        hybrid_predict(source, feature_frame(source, origin), "catboost_daily_bounded_50")


def test_hour_shape_requires_persistent_change_and_complete_history():
    origin = date(2025, 10, 1)
    source = dense_days(origin - timedelta(days=84), 84)
    source["boardings"] = np.where(source.route == 5, 0, 10)
    changed = source.copy()
    recent = changed.date >= origin - timedelta(days=28)
    changed.loc[recent & (changed.hour == 8) & (changed.route == 1), "boardings"] = 80
    changed.loc[recent & (changed.hour != 8) & (changed.route == 1), "boardings"] = 7
    signals = detect_hour_shape_shifts(changed, origin)
    assert len(signals) == 1 and signals[0].entity == "1"
    assert signals[0].kind == "hour_shape" and signals[0].value > .2
    transient = changed.copy()
    second_half = transient.date >= origin - timedelta(days=14)
    transient.loc[second_half & (transient.route == 1), "boardings"] = 10
    assert detect_hour_shape_shifts(transient, origin) == ()
    missing = changed.drop(changed[(changed.route == 1) & (changed.hour == 3)].index[0])
    assert detect_hour_shape_shifts(missing, origin) == ()
    with pytest.raises(ValueError, match="policy"):
        detect_hour_shape_shifts(changed, origin, ShapeShiftPolicy(min_reference_days=0))


def test_known_holidays_do_not_look_like_persistent_demand_change():
    origin = date(2025, 5, 15)
    source = dense_days(origin - timedelta(days=42), 42)
    source["boardings"] = 10
    source.loc[source.date.isin(OFF), "boardings"] = 2
    assert detect_level_shifts(source, origin) == ()


def test_limit_report_separates_data_gaps_from_model_capabilities():
    origin = date(2025, 11, 1)
    history = pd.DataFrame(
        [(route, date(2025, 10, day), 0, 100 if route == 1 else 0)
         for route in (1, 5) for day in range(1, 32)],
        columns=["route", "date", "hour", "boardings"],
    )
    signals = (ShiftSignal("1", "up", 1.7, 170, 100),)
    flags = assess_limits(history, origin, 61, signals, {
        "trend_input_ratio_clip": [.75, 1.25], "max_ml_relative_change": .025,
    })
    assert {(flag.code, flag.entity) for flag in flags} == {
        ("unseen_target_month", "11"), ("unseen_target_month", "12"),
        ("no_observed_route_demand", "5"), ("trend_input_clipped", "1"),
        ("ml_bound_below_observed_change", "1"),
    }

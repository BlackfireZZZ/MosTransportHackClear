import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.stop_models import StopData, calendar_stop_profile


@pytest.fixture
def experiment(monkeypatch):
    scripts = Path(__file__).parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "experiment_stop_source_gain.py"))


def test_calendar_free_anchor_matches_civil_weekdays_and_ignores_future(experiment):
    history = np.arange(304 * 24, dtype=float).reshape(304, 1, 24)
    origin = 120
    days = pd.date_range("2025-01-01", periods=304)
    past = days[:origin].dayofweek
    expected = .5 * history[:origin][past == days[origin].dayofweek].mean(axis=0)
    expected += .5 * history[:origin][past < 5].mean(axis=0)
    before = experiment["anchor"](history, origin, 1, False)
    np.testing.assert_allclose(before[0], expected)
    history[origin:] = 1e12
    np.testing.assert_array_equal(before, experiment["anchor"](history, origin, 1, False))


def test_calendar_anchor_is_the_exact_incumbent(experiment):
    history = np.random.default_rng(4).uniform(size=(304, 2, 24))
    np.testing.assert_array_equal(
        experiment["anchor"](history, 120, 61, True),
        calendar_stop_profile(history, 120, 61, blend=True),
    )


def test_disabled_sources_and_future_labels_cannot_change_features(experiment):
    values = np.arange(304 * 24, dtype=float).reshape(304, 1, 24)
    ids = pd.DataFrame({"route": [1], "stop_id": ["a"], "direction": [0],
                        "lat": [55.75], "lon": [37.6]})
    data = StopData(values, values.copy(), ids, {})
    before, anchor = experiment["features"](data, 120, 2, 0, None, None, None)
    data.train[120:] = 1e12
    after, changed = experiment["features"](data, 120, 2, 0, object(), object(), object())
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_array_equal(anchor, changed)
    assert not any(c.startswith(("calendar", "weather", "traffic", "events")) for c in before)


def test_residual_caps_preserve_zero_support_and_bound_changes(experiment):
    base = np.array([0., 10., 100., 2.])
    residual = np.array([999., -999., 999., .01])
    np.testing.assert_array_equal(experiment["bounded"](base, residual, 0.), base)
    np.testing.assert_allclose(experiment["bounded"](base, residual, .1), [0., 9., 110., 2.01])


def test_acceptance_requires_each_window_and_both_controls(experiment):
    rows = [{"origin": origin, "mask": mask, "cap": cap,
             "pseudo_wape_known": .49 if mask == 15 else .5,
             "route_score": .85}
            for origin in (120, 181, 243)
            for mask, cap in ((0, .1), (1, 0.), (15, .1))]
    assess = experiment["assess"]
    assert assess(rows, .1, (120, 181, 243))["pass"]
    rows[-1]["pseudo_wape_known"] = .501
    assert not assess(rows, .1, (120, 181, 243))["pass"]
    assert assess(rows, .1, (120, 181))["pass"]
    rows[-1]["pseudo_wape_known"] = .49
    rows[-1]["route_score"] = .847
    assert not assess(rows, .1, (120, 181, 243))["pass"]
    rows[-1]["route_score"] = .85
    rows[0]["pseudo_wape_known"] = .48
    assert not assess(rows, .1, (120, 181, 243))["pass"]


def test_all_source_features_ignore_values_at_and_after_origin(experiment):
    from datetime import date

    from tramflow_ml.external_factors import WEATHER_FIELDS

    values = np.ones((304, 1, 24))
    ids = pd.DataFrame({"route": [1], "stop_id": ["a"], "direction": [0],
                        "lat": [55.75], "lon": [37.6]})
    data = StopData(values, values.copy(), ids, {})
    weather = pd.DataFrame({"date": [date(2025, 4, 30), date(2025, 5, 1)],
                            "route": [1, 1], "hour": [12, 12],
                            **{c: [2., 999.] for c in WEATHER_FIELDS}})
    posts = pd.DataFrame({"published_at": ["2025-04-30T12:00:00+03:00",
                                           "2025-05-01T12:00:00+03:00"],
                          "edited": [False, False], "congestion_score": [3., 10.],
                          "mean_speed_kmh": [30., 999.]})
    events = np.zeros((365, 1, 24))
    events[100, 0, 12] = 1.
    before, baseline = experiment["features"](data, 120, 7, 15, weather, posts, events)
    weather.loc[1, WEATHER_FIELDS] = 1e12
    posts.loc[1, ["congestion_score", "mean_speed_kmh"]] = 1e12
    events[120:] = 1e12
    data.train[120:] = 1e12
    after, anchor = experiment["features"](data, 120, 7, 15, weather, posts, events)
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_array_equal(baseline, anchor)
    assert before.weather_available.sum() > 0
    assert before.traffic_congestion_score_support.sum() > 0
    assert before.events_retrospective_recent.max() > 0

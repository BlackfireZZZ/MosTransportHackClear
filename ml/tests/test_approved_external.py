from datetime import date

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.approved_external import _choose, _segments, blend, source_inputs


def _daily() -> pd.DataFrame:
    return pd.DataFrame({"route": [1], "date": [date(2025, 7, 1)],
                         "anchor_day": [100.0], "lead_day": [1]})


def test_weather_never_uses_target_date_actuals() -> None:
    weather = pd.DataFrame({"route": [1, 1, 1],
                            "date": [date(2025, 6, 29), date(2025, 6, 30),
                                     date(2025, 7, 1)],
                            "temperature_2m": [10.0, 999.0, 999.0],
                            "precipitation": [2.0, 999.0, 999.0]})
    result = source_inputs(_daily(), date(2025, 7, 1), "weather",
                           weather, pd.DataFrame())
    assert result.temperature_28.iloc[0] == 10
    assert result.precipitation_28.iloc[0] == 2
    assert result.weather_support.iloc[0] == 1


def test_posts_strictly_before_origin() -> None:
    posts = pd.DataFrame({"date": [date(2025, 6, 30), date(2025, 7, 1)],
                          "congestion_score": [4.0, 10.0],
                          "incident_notice": [True, True],
                          "closure_notice": [False, True]})
    traffic = source_inputs(_daily(), date(2025, 7, 1), "traffic",
                            pd.DataFrame(), posts)
    events = source_inputs(_daily(), date(2025, 7, 1), "events",
                           pd.DataFrame(), posts)
    assert traffic.congestion_56.iloc[0] == 4
    assert traffic.traffic_support.iloc[0] == 1
    assert events.incident_28.iloc[0] == 1
    assert events.closure_28.iloc[0] == 0


def test_calendar_only_adds_fields_absent_from_incumbent() -> None:
    result = source_inputs(_daily(), date(2025, 7, 1), "calendar",
                           pd.DataFrame(), pd.DataFrame())
    assert list(result.columns) == ["route", "anchor_day", "lead_day",
                                    "days_to_off", "days_since_off", "adjacent_off"]
    assert not {"month", "day_of_week", "calendar_day_type", "calendar_exception"} & set(result)


def test_reject_target_dates_before_origin_and_nonconvex_weights() -> None:
    with pytest.raises(ValueError, match="origin"):
        source_inputs(_daily(), date(2025, 7, 2), "calendar",
                      pd.DataFrame(), pd.DataFrame())
    with pytest.raises(ValueError, match="convex"):
        blend(np.array([1.0]), {"calendar": np.array([2.0])}, {"calendar": 1.1})
    with pytest.raises(ValueError, match="misaligned"):
        blend(np.array([1.0]), {"calendar": np.array([2.0, 3.0])}, {"calendar": 0.1})


def test_calibration_cannot_select_ineligible_sources() -> None:
    calibration = {"truth": np.array([20.0]), "incumbent": np.array([10.0]),
                   "branches": {"calendar": np.array([10.0]),
                                "weather": np.array([20.0]),
                                "traffic": np.array([20.0]),
                                "events": np.array([20.0])}}
    assert _choose(calibration) == {}


def test_zero_mass_route_has_undefined_segment_score() -> None:
    keys = pd.DataFrame({"route": [1, 5], "hour": [0, 0]})
    result = _segments(np.array([10.0, 0.0]), np.array([9.0, 0.0]), keys)
    assert result["route:1"] == 0.9
    assert result["route:5"] is None

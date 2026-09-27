from datetime import date

import numpy as np
import pytest

from tramflow_ml.competition import hour_grid, validate_submission
from tramflow_ml.statistical_sweep import (
    Config,
    configurations,
    extended_configurations,
    monthly_configurations,
    predict,
)


@pytest.mark.parametrize(
    "config", configurations() + monthly_configurations() + extended_configurations(),
    ids=lambda config: config.name,
)
def test_constant_zero_and_future_leakage(config: Config) -> None:
    labels = hour_grid(date(2025, 1, 1), date(2025, 5, 31))
    labels["boardings"] = np.where(labels.route == 5, 0, labels.route + labels.hour)
    origin = date(2025, 4, 1)
    before = predict(labels, origin, config)
    expected = hour_grid(origin, date(2025, 5, 31))
    np.testing.assert_allclose(before, np.where(expected.route == 5, 0,
                                               expected.route + expected.hour))
    labels.loc[labels.date >= origin, "boardings"] = 10000000
    np.testing.assert_array_equal(before, predict(labels, origin, config))


def test_grid_and_missing_history() -> None:
    labels = hour_grid(date(2025, 10, 1), date(2025, 10, 31))
    labels["boardings"] = 7
    result = hour_grid(date(2025, 11, 1), date(2025, 12, 31))
    result["prediction"] = predict(labels, date(2025, 11, 1), Config("mean"))
    validate_submission(result)
    with pytest.raises(ValueError, match="missing"):
        predict(labels.iloc[1:], date(2025, 11, 1), Config("mean"))
    with pytest.raises(ValueError, match="no history"):
        predict(labels, date(2025, 10, 1), Config("mean"))
    with pytest.raises(ValueError, match="positive"):
        predict(labels, date(2025, 11, 1), Config("mean"), horizon=0)


def test_frozen_grid() -> None:
    grid = configurations()
    assert len(grid) == 114
    assert len({item.name for item in grid}) == len(grid)


def test_weekday_and_calendar_exception_are_distinct() -> None:
    labels = hour_grid(date(2025, 1, 1), date(2025, 3, 31))
    labels["boardings"] = [100 + day.weekday() * 10 for day in labels.date]
    output = predict(labels, date(2025, 4, 1), Config("weekday"), horizon=30)
    expected = hour_grid(date(2025, 4, 1), date(2025, 4, 30))
    np.testing.assert_allclose(output, [100 + day.weekday() * 10 for day in expected.date])


def test_season_fallback_uses_only_available_seasons() -> None:
    labels = hour_grid(date(2025, 5, 1), date(2025, 8, 31))
    labels["boardings"] = [10 if day.month == 5 else 100 for day in labels.date]
    output = predict(labels, date(2025, 9, 1), Config("seasonal", seasonal=True))
    np.testing.assert_allclose(output, 10)


def test_month_profiles_weight_months_equally() -> None:
    labels = hour_grid(date(2025, 1, 1), date(2025, 3, 31))
    labels["boardings"] = [day.month * 100 for day in labels.date]
    origin = date(2025, 4, 1)
    np.testing.assert_allclose(predict(labels, origin, Config(
        "previous", calendar_months=1)), 300)
    np.testing.assert_allclose(predict(labels, origin, Config(
        "equal", "month_equal", calendar_months=2)), 250)
    weights = np.exp2(-np.array([1, 2, 3]))
    expected = np.average([300, 200, 100], weights=weights)
    np.testing.assert_allclose(predict(labels, origin, Config(
        "exponential", "month_exp", half_life=1)), expected)


def test_month_profile_discards_partial_current_month() -> None:
    labels = hour_grid(date(2025, 1, 1), date(2025, 3, 14))
    labels["boardings"] = [9999 if day.month == 3 else 5 for day in labels.date]
    for config in monthly_configurations()[1:]:
        np.testing.assert_allclose(predict(labels, date(2025, 3, 15), config), 5)
    assert len(monthly_configurations()) == 13


def test_extended_grid_frozen() -> None:
    grid = extended_configurations()
    assert len(grid) == 125
    assert len({config.name for config in grid}) == len(grid)

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.competition import hour_grid
from tramflow_ml.complete_features import (
    calendar_features,
    catalog_features,
    enrich_stops,
    route_fold,
)


def test_calendar_midnight_and_transferred_workday() -> None:
    frame = pd.DataFrame({"date": ["2025-11-01", "2025-11-04"], "hour": [0, 23]})
    result = calendar_features(frame)
    assert result.calendar_day_type.tolist() == [0, 2]
    assert result.calendar_exception.tolist() == [1, 1]
    assert np.isclose(result.hour_sin.iloc[0], 0)
    assert "boardings" not in result


@pytest.mark.parametrize("hour", [-1, 24, 0.5, np.nan])
def test_calendar_rejects_bad_hours(hour: float) -> None:
    with pytest.raises(ValueError):
        calendar_features(pd.DataFrame({"date": ["2025-01-01"], "hour": [hour]}))


def test_fold_future_mutation_cannot_change_features() -> None:
    labels = hour_grid(date(2025, 1, 1), date(2025, 6, 30))
    labels["boardings"] = 5
    origin = date(2025, 5, 1)
    first = route_fold(labels, origin)
    labels.loc[labels.date >= origin, "boardings"] = 9999
    second = route_fold(labels, origin)
    pd.testing.assert_frame_equal(first.drop(columns="boardings"), second.drop(columns="boardings"))
    assert len(first) == 14640
    assert first.feature_cutoff.eq("2025-04-30").all()
    assert not first.boardings.equals(second.boardings)


def source() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame({
        "day": ["2025-01-01"], "event_hour": ["2025-01-01T00:00:00+03:00"],
        "route": [7], "hour": [0], "direction": ["0"], "stop_id": ["gtfs:1"],
        "expected_count": [2.5], "training_eligible": [False],
    })
    catalog = pd.DataFrame({
        "route": [7, 7], "direction": ["0", "0"], "stop_id": ["gtfs:1", "gtfs:1"],
        "lat": [55.1, 55.1], "lon": [37.1, 37.1], "sequence": [0, 24],
    })
    return frame, catalog


def test_stop_join_does_not_duplicate_or_erase_exclusion() -> None:
    frame, catalog = source()
    result = enrich_stops(frame, catalog)
    assert len(result) == 1
    assert result.expected_count.sum() == 2.5
    assert not result.training_eligible.iloc[0]
    assert result.sequence_ambiguous.iloc[0]
    assert result.sequence_min.iloc[0] == 0
    assert result.sequence_max.iloc[0] == 24


def test_conflicting_geometry_rejected() -> None:
    _, catalog = source()
    catalog.loc[1, "lat"] = 56
    with pytest.raises(ValueError, match="conflicting geometry"):
        catalog_features(catalog)


def test_original_moscow_hour_checked() -> None:
    frame, catalog = source()
    frame["event_hour"] = "2025-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="Moscow hour"):
        enrich_stops(frame, catalog)


def test_missing_geometry_remains_explicit() -> None:
    frame, catalog = source()
    frame["stop_id"] = "unassigned"
    result = enrich_stops(frame, catalog)
    assert not result.geometry_available.iloc[0]
    assert result.expected_count.sum() == 2.5


def test_fold_rejects_wrong_truth_keys() -> None:
    labels = hour_grid(date(2025, 1, 1), date(2025, 6, 30))
    labels["boardings"] = 5
    labels.loc[(labels.date == date(2025, 5, 1)) & labels.route.eq(1)
               & labels.hour.eq(0), "route"] = 999
    with pytest.raises(ValueError, match="keys"):
        route_fold(labels, date(2025, 5, 1))


@pytest.mark.parametrize("lat,lon", [(np.inf, 37), (55, 999), (91, 37)])
def test_geometry_range_rejected(lat: float, lon: float) -> None:
    _, catalog = source()
    catalog["lat"], catalog["lon"] = lat, lon
    with pytest.raises(ValueError, match="geometry"):
        catalog_features(catalog)


def test_fractional_hour_rejected() -> None:
    frame, catalog = source()
    frame["event_hour"] = "2025-01-01T00:00:00.1+03:00"
    with pytest.raises(ValueError, match="Moscow hour"):
        enrich_stops(frame, catalog)


def test_full_stop_feature_matrix_has_cutoff_and_no_forecast_target(tmp_path) -> None:
    from tramflow_ml.complete_features import export_stop_model_features
    from tramflow_ml.stop_models import StopData

    identities = pd.DataFrame({"route": [1], "stop_id": ["gtfs:1"], "direction": [0],
                               "lat": [55.7], "lon": [37.6]})
    values = np.ones((60, 1, 24))
    data = StopData(values, values.copy(), identities, {"source": "test"})
    result = export_stop_model_features(data, tmp_path / "matrix")
    train = pd.read_csv(tmp_path / "matrix/training-059.csv.gz")
    future = pd.read_csv(tmp_path / "matrix/forecast-060.csv.gz")
    assert result["files"]["training-059.csv.gz"]["rows"] == 24
    assert train.expected_count_target.sum() == 24
    assert "expected_count_target" not in future
    assert len(future) == 61 * 24
    assert (train.feature_cutoff < train.target_date).all()
    assert (future.feature_cutoff < future.target_date).all()

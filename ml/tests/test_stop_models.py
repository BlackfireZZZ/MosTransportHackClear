from datetime import date

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.competition import hour_grid
from tramflow_ml.stop_models import aggregate, features, prepare, profile


def identities():
    return pd.DataFrame({"route": [1, 1], "stop_id": [10, 11], "direction": [0, 0],
                         "lat": [55., 55.1], "lon": [37., 37.1]})


def test_future_perturbation_invariant():
    history = np.arange(90 * 2 * 24).reshape(90, 2, 24).astype(float)
    original = features(history, identities(), 60, 30)
    history[60:] = 999999
    pd.testing.assert_frame_equal(original, features(history, identities(), 60, 30))


def test_aggregate_preserves_continuous_mass():
    result = aggregate(np.full((2, 2, 24), .4), identities(), 0)
    assert result.prediction.sum() == pytest.approx(38.4)
    assert result.loc[result.route == 1, "prediction"].eq(.8).all()
    assert result.loc[result.route == 5, "prediction"].eq(0).all()


def fixture_rows():
    return pd.DataFrame({"day": ["2025-01-01"] * 2, "route": [1, 1],
                         "stop_id": [10, 11], "direction": [0, 0], "hour": [0, 0],
                         "expected_count": [2., 3.], "training_eligible": [True, False]})


def fixture_labels():
    labels = hour_grid(date(2025, 1, 1), date(2025, 1, 1))
    labels["boardings"] = 0.
    labels.loc[(labels.route == 1) & (labels.hour == 0), "boardings"] = 5.
    return labels


def test_block_flags_do_not_remove_mass_or_create_training_zeros():
    data = prepare(fixture_rows(), identities(), fixture_labels(), complete_decode=True,
                   raw_reconciled=True, source_version="test", days=1)
    assert data.values.sum() == 5
    assert np.isnan(data.train[0, data.identities.route == 1, 0]).all()
    assert data.train[0, :, 1:].sum() == 0
    assert data.audit["row_eligible_mass"] == 2


def test_zero_policy_requires_proof_and_mass_reconciliation():
    with pytest.raises(ValueError, match="complete decoding"):
        prepare(fixture_rows(), identities(), fixture_labels(), complete_decode=False,
                raw_reconciled=True, source_version="test", days=1)
    labels = fixture_labels()
    labels.loc[0, "boardings"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        prepare(fixture_rows(), identities(), labels, complete_decode=True,
                raw_reconciled=True, source_version="test", days=1)


def test_no_future_date_features_or_excluded_zero_bias():
    with pytest.raises(ValueError, match="2025"):
        features(np.zeros((304, 2, 24)), identities(), 304, 62)
    history = np.full((60, 2, 24), 5.)
    history[::2] = np.nan
    assert np.allclose(profile(history, 60, 30, 28), 5.)


def test_unknown_bucket_is_kept_separate_and_all_routes_present():
    rows = fixture_rows()
    rows["stop_id"] = ["10", "unallocated:1"]
    rows["direction"] = [0, -1]
    rows["training_eligible"] = True
    data = prepare(rows, identities(), fixture_labels(), complete_decode=True,
                   raw_reconciled=True, source_version="test", days=1)
    unknown = data.identities.stop_id.eq("unallocated:1").to_numpy()
    assert data.values[:, unknown].sum() == 3
    assert data.identities.route.nunique() == 10
    assert data.audit["block_eligible_mass"] == 5


def test_training_examples_are_frozen_at_outer_origin():
    from tramflow_ml.stop_models import StopData, training_frame

    history = np.arange(90 * 2 * 24).reshape(90, 2, 24).astype(float)
    data = StopData(history.copy(), history.copy(), identities(), {})
    x, y = training_frame(data, 70, 100)
    data.train[70:] = 999999
    xx, yy = training_frame(data, 70, 100)
    pd.testing.assert_frame_equal(x, xx)
    np.testing.assert_array_equal(y, yy)


def test_known_and_unallocated_error_are_reported_separately():
    from tramflow_ml.stop_models import StopData, evaluate

    ids = identities()
    ids["stop_id"] = ["10", "unallocated:1"]
    values = np.zeros((61, 2, 24))
    values[:, 0] = 2
    values[:, 1] = 3
    data = StopData(values, values, ids, {})
    predictions = np.zeros((1, 2, 24))
    predictions[:, 0] = 3
    predictions[:, 1] = 2
    result = evaluate(data, predictions, 60)
    assert result["route_score"] == 1
    assert result["pseudo_wape_known"] == .5
    assert result["pseudo_wape_all"] == .4
    assert result["unallocated_mass"] == 72


@pytest.mark.parametrize("column,value", [("hour", .5), ("day", "2025-01-01T00:30:00")])
def test_non_hourly_keys_rejected(column, value):
    rows = fixture_rows().astype({column: object})
    rows.loc[0, column] = value
    with pytest.raises(ValueError):
        prepare(rows, identities(), fixture_labels(), complete_decode=True,
                raw_reconciled=True, source_version="test", days=1)


@pytest.mark.parametrize("column,value", [("lat", 91.), ("lon", np.inf)])
def test_invalid_geometry_rejected(column, value):
    coordinates = identities()
    coordinates.loc[0, column] = value
    with pytest.raises(ValueError, match="coordinate"):
        prepare(fixture_rows(), coordinates, fixture_labels(), complete_decode=True,
                raw_reconciled=True, source_version="test", days=1)


def test_future_identity_presence_cannot_change_historical_grid():
    coords = identities()
    rows = fixture_rows().iloc[:1].copy()
    rows["training_eligible"] = True
    labels = hour_grid(date(2025, 1, 1), date(2025, 3, 1))
    labels["boardings"] = 0.
    labels.loc[(labels.route == 1) & (labels.date == date(2025, 1, 1))
               & (labels.hour == 0), "boardings"] = 2.
    first = prepare(rows, coords, labels, complete_decode=True, raw_reconciled=True,
                    source_version="test", days=60)
    future = fixture_rows().iloc[1:].copy()
    future["day"] = "2025-03-01"
    future["training_eligible"] = True
    labels.loc[(labels.route == 1) & (labels.date == date(2025, 3, 1))
               & (labels.hour == 0), "boardings"] = 3.
    second = prepare(pd.concat([rows, future]), coords, labels, complete_decode=True,
                     raw_reconciled=True, source_version="test", days=60)
    pd.testing.assert_frame_equal(first.identities, second.identities)
    pd.testing.assert_frame_equal(features(first.train, first.identities, 59, 1),
                                  features(second.train, second.identities, 59, 1))


@pytest.mark.parametrize("blend", [False, True])
def test_stop_calendar_baseline_matches_route_baseline_without_forcing_totals(blend):
    from tramflow_ml.competition import (
        calendar_profile_predict,
        calendar_weekday_blend_predict,
        feature_frame,
    )
    from tramflow_ml.stop_models import calendar_stop_profile

    history = np.random.default_rng(4).poisson(20, (243, 2, 24)).astype(float)
    route_history = aggregate(history, identities(), 0).rename(columns={"prediction": "boardings"})
    for origin in (120, 181, 243):
        route_features = feature_frame(route_history, date(2025, 1, 1)
                                       + pd.Timedelta(days=origin), 61)
        route_predict = calendar_weekday_blend_predict if blend else calendar_profile_predict
        expected = route_predict(route_features)
        independent = calendar_stop_profile(history, origin, 61, blend=blend)
        actual = aggregate(independent, identities(), origin).prediction.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


def test_reuse_requires_explicit_compatible_code_and_score_hashes(tmp_path):
    import json

    from tramflow_ml.stop_models import StopData, run_experiment

    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "run.json").write_text(json.dumps({"code_sha256": "old"}))
    (prior / "scores.json").write_text("[]")
    data = StopData(np.zeros((1, 2, 24)), np.zeros((1, 2, 24)), identities(), {})
    with pytest.raises(ValueError, match="checksums"):
        run_experiment(data, tmp_path / "out", reuse=prior)

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.stop_models import StopData
from tramflow_ml.stop_refinement import bounded_prediction, refinement_features


def test_bounded_independent_correction_keeps_zero_and_cap():
    np.testing.assert_allclose(bounded_prediction(np.array([0., 100., 100.]),
                                                np.array([2., 2., -2.])), [0, 105, 95])
    with pytest.raises(ValueError):
        bounded_prediction(np.array([1.]), np.array([float("nan")]))


def test_future_targets_do_not_change_profiles_shares_or_coverage():
    identities = pd.DataFrame({"route": [1, 1], "stop_id": ["gtfs:1", "unallocated:1"],
                               "direction": [0, -1], "lat": [55.7, np.nan],
                               "lon": [37.6, np.nan]})
    values = np.ones((150, 2, 24))
    values[:, 1] *= 3
    data = StopData(values.copy(), values.copy(), identities, {})
    before = refinement_features(data, 90, 3)
    data.train[90:] = 999999
    after = refinement_features(data, 90, 3)
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_allclose(before.mapping_coverage_7, .25)
    np.testing.assert_allclose(before.groupby("stop").recent_share_7.first(), [.25, .75])

"""Invariant checks for replacing inferred allocations in the fixed linear model."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tramflow_ml.stop_models import StopData

SPEC = importlib.util.spec_from_file_location(
    "refresh_anchored_stop_model",
    Path(__file__).resolve().parents[2] / "scripts/refresh_anchored_stop_model.py",
)
assert SPEC and SPEC.loader
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)


def dataset(share: float) -> StopData:
    identities = pd.DataFrame({"route": [1, 1], "stop_id": ["known", "unallocated:1"],
                               "direction": [0, -1]})
    total = np.arange(304 * 24).reshape(304, 1, 24) % 17
    values = np.concatenate([total * share, total * (1 - share)], axis=1)
    return StopData(values, values.copy(), identities, {})


def test_redistribution_preserves_linear_route_predictions_without_future_counts():
    old, new = dataset(.2), dataset(.85)
    refresh.verify_pair(old, new)
    for origin, days in ((120, 61), (243, 1), (304, 61)):
        before = refresh.forecast_checked(old, origin, days)
        after = refresh.forecast_checked(new, origin, days)
        assert refresh.compare_routes(old, new, before, after, origin) < 1e-8
        assert after[:, 1].sum() / after.sum() == pytest.approx(.15)


def test_missingness_and_identity_changes_cannot_masquerade_as_redistribution():
    old, new = dataset(.2), dataset(.85)
    new.train[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="eligibility"):
        refresh.verify_pair(old, new)
    new = dataset(.85)
    new.identities.loc[0, "stop_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        refresh.verify_pair(old, new)


def test_lost_mass_is_rejected():
    old, new = dataset(.2), dataset(.85)
    new.values[0, 0, 0] += 1
    with pytest.raises(ValueError, match="counts differ"):
        refresh.verify_pair(old, new)

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/experiment_external_residual.py"
spec = importlib.util.spec_from_file_location("external_residual", PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_bound_preserves_zero_and_limits_both_directions():
    assert np.array_equal(module.bounded([0., 100., 200.], [99., -50., 50.], .025),
                          [0., 97.5, 205.])


@pytest.mark.parametrize("anchor,residual,bound", [([1], [1, 2], .1), ([-1], [1], .1),
                                                 ([1], [float("nan")], .1), ([1], [1], 2)])
def test_invalid_correction_rejected(anchor, residual, bound):
    with pytest.raises(ValueError):
        module.bounded(anchor, residual, bound)


def test_zero_mass_is_unavailable_and_rounding_is_once():
    assert module.metrics(np.array([0.]), np.array([2.])) == {
        "actual": 0., "absolute_error": 2., "score": None}
    assert module.metrics(np.array([2.]), np.array([1.6]))["score"] == 1.

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from contracts.forecast_v1 import validate_forecast_json  # noqa: E402, I001


FIXTURE = ROOT / "contracts/fixtures/forecast_v1.valid.json"


def test_ml_accepts_same_forecast_fixture_as_backend() -> None:
    artifact = validate_forecast_json(FIXTURE.read_text(encoding="utf-8"))

    assert artifact.target.value == "synthetic_boardings"
    assert artifact.points[0].bucket_end > artifact.points[0].bucket_start


def test_ml_rejects_nonfinite_prediction_in_shared_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["points"][0]["predicted"] = "NaN"

    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(payload))

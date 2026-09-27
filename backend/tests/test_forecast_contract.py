import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from contracts.forecast_v1 import validate_forecast_json  # noqa: E402, I001


FIXTURE = ROOT / "contracts/fixtures/forecast_v1.valid.json"


def test_backend_accepts_shared_forecast_fixture() -> None:
    artifact = validate_forecast_json(FIXTURE.read_text(encoding="utf-8"))

    assert artifact.run_id == "run-2026-01"
    assert artifact.points[0].stop_id == "stop-1"


def test_backend_rejects_shared_fixture_with_unknown_version() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["schema_version"] = "forecast.v2"

    with pytest.raises(ValidationError):
        validate_forecast_json(json.dumps(payload))

import json
from datetime import date
from pathlib import Path

import pytest

from tramflow_ml.issued_weather import END, START, load_issued_weather

ROOT = Path(__file__).resolve().parents[2]
WEATHER = ROOT / "data/weather/issued-2025/hourly.csv.gz"


def test_archived_issued_weather_grid_and_cutoff() -> None:
    rows = load_issued_weather(WEATHER)
    assert len(rows) == 72960
    assert rows.date.min() == START == date(2025, 1, 1)
    assert rows.date.max() == END == date(2025, 10, 31)
    assert rows.groupby("route").size().nunique() == 1


def test_issued_weather_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / WEATHER.name
    output.write_bytes(WEATHER.read_bytes())
    manifest = json.loads((WEATHER.parent / "manifest.json").read_text())
    manifest["csv_gz_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_issued_weather(output)

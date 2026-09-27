import json

import pandas as pd
import pytest

from tramflow_ml.demand_map import render_map


def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame({
        "route": [11, 11], "direction": ["0", "-1"],
        "stop_id": ["gtfs:1", "unallocated:11"], "date": ["2025-11-01"] * 2,
        "hour": [1, 1], "prediction": [20., 7.],
    }), pd.DataFrame({
        "route": [11], "direction": ["0"], "stop_id": ["gtfs:1"],
        "lat": [55.7], "lon": [37.6], "name": ["</script><script>unsafe</script>"],
    })


def test_map_preserves_unknown_mass_and_escapes_script() -> None:
    forecast, catalog = data()
    html = render_map(forecast, catalog, "test")
    assert "<script>unsafe</script>" not in html
    encoded = html.split('id="data">')[1].split('</script>')[0]
    payload = json.loads(encoded)
    assert sum(row[-1] for row in payload["values"]) == 27
    assert any(point["lat"] is None for point in payload["points"])
    assert "не заполненность салона" in html


def test_duplicate_map_keys_fail() -> None:
    forecast, catalog = data()
    with pytest.raises(ValueError, match="duplicate"):
        render_map(pd.concat([forecast, forecast]), catalog, "test")


def test_incomplete_map_grid_fails() -> None:
    forecast, catalog = data()
    partial = pd.concat([forecast, forecast.iloc[:1].assign(date="2025-11-02")])
    with pytest.raises(ValueError, match="complete identity/date/hour grid"):
        render_map(partial, catalog, "test")

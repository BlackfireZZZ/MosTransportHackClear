from datetime import date

import pandas as pd
import pytest

from tramflow_ml.external_factors import parse_traffic_page, source_features


def test_parser_current_observation_and_timezone():
    page = ('<div class="tgme_widget_message " data-post="DtOperativno/123">'
            '<div class="tgme_widget_message_text js-message_text">'
            'ЦОДД: на дорогах сейчас 5 баллов. Средняя скорость движения — 32 км/ч.</div>'
            '<time datetime="2025-07-01T10:00:00+00:00"></time>')
    row = parse_traffic_page(page)[0]
    assert row["congestion_score"] == 5
    assert row["mean_speed_kmh"] == 32
    assert row["published_at"] == "2025-07-01T10:00:00+00:00"
    future = page.replace("на дорогах сейчас", "ожидается вечером")
    assert parse_traffic_page(future)[0]["congestion_score"] is None
    with pytest.raises(ValueError, match="timezone"):
        parse_traffic_page(page.replace("+00:00", ""))


def fixtures():
    frame = pd.DataFrame({"route": [1], "hour": [0], "day_of_week": [1], "lead_day": [1],
                          "route_hour_28": [3.], "route_hour_56": [4.],
                          "route_hour_all": [5.], "route_day_28": [6.], "route_day_56": [7.],
                          "date": [date(2025, 7, 1)]})
    weather = pd.DataFrame({"route": [1, 1], "hour": [0, 0],
                            "date": [date(2025, 6, 30), date(2025, 7, 1)],
                            "temperature_2m": [20., 999.], "precipitation": [0., 999.],
                            "snowfall": [0., 999.], "wind_speed_10m": [2., 999.]})
    posts = pd.DataFrame({"published_at": ["2025-06-23T21:00:00Z", "2025-06-30T21:00:00Z"],
                          "edited": [False, False], "congestion_score": [3., 999.],
                          "mean_speed_kmh": [40., 999.], "closure_notice": [True, True],
                          "incident_notice": [False, True]})
    return frame, weather, posts


def test_future_source_perturbation_and_moscow_midnight():
    frame, weather, posts = fixtures()
    enabled = frozenset({"weather", "traffic", "news"})
    original = source_features(frame, date(2025, 7, 1), enabled=enabled,
                               weather=weather, posts=posts)
    weather.loc[1, "temperature_2m"] = -999
    posts.loc[1, "congestion_score"] = -999
    changed = source_features(frame, date(2025, 7, 1), enabled=enabled,
                              weather=weather, posts=posts)
    pd.testing.assert_frame_equal(original, changed)
    assert original.traffic_congestion_score.iloc[0] == 3
    assert original.weather_temperature_2m.iloc[0] == 20


def test_disabled_sources_do_not_read_input_and_unknown_rejected():
    frame, weather, posts = fixtures()
    baseline = source_features(frame, date(2025, 7, 1), enabled=frozenset(),
                               weather=weather, posts=posts)
    empty = source_features(frame, date(2025, 7, 1), enabled=frozenset(),
                            weather=pd.DataFrame(), posts=pd.DataFrame())
    pd.testing.assert_frame_equal(baseline, empty)
    assert not any(name.startswith(("traffic_", "news_", "weather_")) for name in baseline)
    with pytest.raises(ValueError, match="unknown"):
        source_features(frame, date(2025, 7, 1), enabled=frozenset({"fake"}),
                        weather=weather, posts=posts)


def test_missing_and_edited_traffic_never_become_zero_congestion():
    frame, weather, posts = fixtures()
    posts["edited"] = True
    values = source_features(frame, date(2025, 7, 1), enabled=frozenset({"traffic"}),
                             weather=weather, posts=posts)
    assert values.traffic_congestion_score.iloc[0] == -1
    assert values.traffic_congestion_score_support.iloc[0] == 0


def test_publication_date_moscow_midnight_and_year_boundary():
    from tramflow_ml.external_factors import moscow_publication_day

    assert moscow_publication_day("2024-12-31T21:00:00Z") == date(2025, 1, 1)
    assert moscow_publication_day("2025-10-31T21:00:00Z") == date(2025, 11, 1)
    assert moscow_publication_day("2025-06-30T20:59:59Z") == date(2025, 6, 30)


@pytest.mark.parametrize("defect", ["route", "mass", "provenance"])
def test_label_export_rejects_self_consistent_hash_with_wrong_contract(tmp_path, defect):
    import hashlib
    import json

    from tramflow_ml.competition import hour_grid
    from tramflow_ml.external_factors import load_labels

    labels = hour_grid(date(2025, 1, 1), date(2025, 10, 31))
    labels["boardings"] = 0
    labels.loc[0, "boardings"] = 59667191
    if defect == "route":
        labels.loc[0, "route"] = 999
    elif defect == "mass":
        labels.loc[0, "boardings"] -= 1
    filename = "route-hour-labels.csv.gz"
    labels.to_csv(tmp_path / filename, index=False)
    checksum = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
    manifest = {
        "complete": defect != "provenance", "timezone": "Europe/Moscow",
        "target": "successful_validation_count_route_date_hour",
        "date_range": ["2025-01-01", "2025-10-31"],
        "archive_sha256": "7e34e56b93f7379cb5abe9aca8e87967cb986111b9939caead5a3fea6c296e5a",
        "reconciliation_sha256": "b" * 64, "target_mass": 59667191,
        "files": {filename: {"sha256": checksum}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance|grid or target mass"):
        load_labels(tmp_path)

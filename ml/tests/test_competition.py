import hashlib
import json
import sys
import types
import zipfile
from datetime import date, timedelta

import pandas as pd
import pytest

from tramflow_ml.competition import (
    LABEL_ROUTES,
    calendar_profile_predict,
    calendar_weekday_blend_predict,
    complete_labels,
    feature_frame,
    make_training_frame,
    profile_predict,
    score,
    validate_submission,
)
from tramflow_ml.competition_cli import _evaluate_one, _training_examples, _verified_proof
from tramflow_ml.competition_graph import COLUMNS, build_osm_structure
from tramflow_ml.competition_models import _daily_relative_predict
from tramflow_ml.competition_profile import robust_hour_shape_predict, robust_profile_predict
from tramflow_ml.competition_reconcile import reconcile_archive
from tramflow_ml.competition_ts import time_series_predict
from tramflow_ml.route_models.features import OFF, WORK


def labels(start: date, days: int) -> pd.DataFrame:
    return pd.DataFrame(
        (
            (route, start + timedelta(days=offset), hour, route + hour + offset)
            for route in LABEL_ROUTES
            for offset in range(days)
            for hour in range(24)
        ),
        columns=["route", "date", "hour", "boardings"],
    )


def test_missing_labels_require_explicit_zero_policy():
    source = labels(date(2025, 1, 1), 2).iloc[:-1]
    with pytest.raises(ValueError, match="missing"):
        complete_labels(source, date(2025, 1, 1), date(2025, 1, 2))
    dense = complete_labels(source, date(2025, 1, 1), date(2025, 1, 2), missing_as_zero=True)
    assert len(dense) == 480
    assert dense.iloc[-1].boardings == 0


def test_future_labels_do_not_change_features_or_training_rows():
    source = labels(date(2025, 1, 1), 150)
    origin = date(2025, 4, 1)
    first = feature_frame(source, origin, 61)
    changed = source.copy()
    changed.loc[changed.date >= origin, "boardings"] += 10_000
    pd.testing.assert_frame_equal(first, feature_frame(changed, origin, 61))
    assert (profile_predict(first) == profile_predict(feature_frame(changed, origin, 61))).all()
    assert (
        calendar_weekday_blend_predict(first)
        == calendar_weekday_blend_predict(feature_frame(changed, origin, 61))
    ).all()
    original_train = make_training_frame(source, date(2025, 5, 31), 61)
    perturbed_train = make_training_frame(changed, date(2025, 5, 31), 61)
    assert not original_train.equals(perturbed_train)
    assert original_train.date.max() <= date(2025, 5, 31)


def test_robust_profile_ignores_future_labels_and_retains_cutoff():
    source = labels(date(2025, 1, 1), 150)
    origin = date(2025, 4, 1)
    future = feature_frame(source, origin)
    past = source.loc[source.date < origin]
    first = robust_profile_predict(past, future, "calendar_robust_short")
    changed = source.copy()
    changed.loc[changed.date >= origin, "boardings"] += 10_000
    second = robust_profile_predict(
        changed.loc[changed.date < origin], feature_frame(changed, origin),
        "calendar_robust_short",
    )
    assert (first == second).all()
    assert (first >= 0).all()


def test_robust_profile_rejects_future_history_and_unknown_variant():
    source = labels(date(2025, 1, 1), 91)
    future = feature_frame(source, date(2025, 4, 1))
    with pytest.raises(ValueError, match="future labels"):
        robust_profile_predict(source, future, "calendar_robust_short")
    with pytest.raises(ValueError, match="unknown robust profile"):
        robust_profile_predict(source.loc[source.date < date(2025, 4, 1)], future, "wrong")


def test_robust_blend_uses_unrounded_component_predictions():
    source = labels(date(2025, 1, 1), 90)
    future = feature_frame(source, date(2025, 4, 1))
    short = robust_profile_predict(source, future, "calendar_robust_short")
    long = robust_profile_predict(source, future, "calendar_robust_28")
    blend = robust_profile_predict(source, future, "calendar_robust_blend")
    assert (blend == 0.5 * short + 0.5 * long).all()


@pytest.mark.parametrize("base_model", ["calendar_robust_blend", "calendar_robust_28"])
@pytest.mark.parametrize("group", ["day_type", "route"])
def test_hour_shape_preserves_daily_totals_and_ignores_future_labels(
    base_model: str, group: str
):
    source = labels(date(2025, 1, 1), 150)
    source.loc[source.route == 5, "boardings"] = 0
    origin = date(2025, 4, 1)
    future = feature_frame(source, origin)
    past = source.loc[source.date < origin]
    baseline = robust_profile_predict(past, future, base_model)
    shaped = robust_hour_shape_predict(past, future, base_model, group)
    assert not (baseline == shaped).all()
    assert pd.notna(shaped).all()
    assert (shaped[future.route == 5] == 0).all()
    keys = future[["route", "date"]]
    baseline_totals = pd.Series(baseline).groupby([keys.route, keys.date]).sum()
    shaped_totals = pd.Series(shaped).groupby([keys.route, keys.date]).sum()
    pd.testing.assert_series_equal(baseline_totals, shaped_totals, atol=1e-6)
    changed = source.copy()
    changed.loc[changed.date >= origin, "boardings"] += 10_000
    same = robust_hour_shape_predict(
        changed.loc[changed.date < origin], feature_frame(changed, origin), base_model,
        group,
    )
    assert (shaped == same).all()


def test_score_pools_absolute_error_and_uses_rounded_predictions():
    assert score([100, 0], [89.6, 0.4]) == pytest.approx(0.9)
    assert score([0], [0]) is None


def test_published_calendar_corrects_holiday_and_working_saturday():
    source = labels(date(2025, 1, 15), 290)
    work = source.date.map(lambda day: day in WORK or (day.weekday() < 5 and day not in OFF))
    source["boardings"] = work.map({True: 100, False: 20})
    for origin, target_day, base, corrected in (
        (date(2025, 4, 1), date(2025, 5, 1), 100, 20),
        (date(2025, 11, 1), date(2025, 11, 1), 20, 100),
    ):
        future = feature_frame(source, origin)
        row = future.loc[
            (future.route == 1) & (future.date == target_day) & (future.hour == 8)
        ]
        assert profile_predict(row)[0] == pytest.approx(base)
        assert calendar_profile_predict(row)[0] == pytest.approx(corrected)


def test_calendar_weekday_blend_respects_support_and_exception():
    rows = pd.DataFrame({
        "same_season_support": [4, 4, 4],
        "route_dow_hour_same_season": [100, 100, 100],
        "route_dow_hour_all": [80, 80, 80],
        "calendar_exception": [0, 0, 1],
        "day_type_support": [3, 0, 3],
        "route_day_type_hour_same_season": [60, 60, 60],
    })
    assert calendar_weekday_blend_predict(rows).tolist() == [80, 100, 60]


def test_submission_rejects_missing_duplicate_extra_and_bad_values():
    start = date(2025, 11, 1)
    valid = feature_frame(labels(date(2025, 1, 1), 31), start, 61)[["route", "date", "hour"]].copy()
    valid["prediction"] = 1.0
    validate_submission(valid)
    with pytest.raises(ValueError, match="grid"):
        validate_submission(valid.iloc[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission(pd.concat([valid, valid.iloc[:1]], ignore_index=True))
    bad = valid.copy()
    bad.loc[0, "prediction"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_submission(bad)


def test_direction_graph_features_require_past_available_version():
    source = labels(date(2025, 1, 1), 90)
    origin = date(2025, 4, 1)
    graph = pd.DataFrame(
        [
            (1, "out", date(2025, 3, 1), date(2025, 1, 1), date(2025, 12, 31), 30, 29, 8000, "v1"),
            (
                1,
                "back",
                date(2025, 3, 1),
                date(2025, 1, 1),
                date(2025, 12, 31),
                35,
                34,
                10000,
                "v1",
            ),
            (
                7,
                "out",
                date(2026, 9, 18),
                date(2025, 1, 1),
                date(2025, 12, 31),
                40,
                39,
                9000,
                "future",
            ),
        ],
        columns=[
            "route",
            "direction_id",
            "available_at",
            "valid_from",
            "valid_to",
            "stop_count",
            "edge_count",
            "length_m",
            "source_version",
        ],
    )
    with_graph = feature_frame(source, origin, 61, direction_graph=graph)
    route1 = with_graph.loc[with_graph.route == 1].iloc[0]
    route7 = with_graph.loc[with_graph.route == 7].iloc[0]
    assert route1.graph_direction_count == 2
    assert route1.graph_length_m_spread == 2000
    assert route7.graph_available == 0
    changed = graph.copy()
    changed.loc[changed.route == 7, "length_m"] = 100_000
    pd.testing.assert_frame_equal(
        with_graph, feature_frame(source, origin, 61, direction_graph=changed)
    )


def test_structural_direction_features_allow_unknown_length_and_ignore_future_edit():
    source = labels(date(2025, 1, 1), 90)
    origin = date(2025, 4, 1)
    graph = pd.DataFrame(
        [
            (1, "osm:11", "2025-02-20", "2025-02-20", "9999-12-31", 21, 20, "", "v1"),
            (1, "osm:12", "2025-02-20", "2025-02-20", "9999-12-31", 25, 24, "", "v1"),
            (1, "osm:11", "2025-06-01", "2025-06-01", "9999-12-31", 40, 39, "", "v2"),
        ],
        columns=[
            "route", "direction_id", "available_at", "valid_from", "valid_to",
            "stop_count", "edge_count", "length_m", "source_version",
        ],
    )
    current = feature_frame(source, origin, direction_graph=graph)
    row = current.loc[current.route == 1].iloc[0]
    assert row.graph_direction_count == 2
    assert row.graph_stop_count_mean == 23
    assert row.graph_length_available == 0
    graph.loc[2, "stop_count"] = 500
    pd.testing.assert_frame_equal(current, feature_frame(source, origin, direction_graph=graph))


def test_osm_structure_uses_only_dated_relation_edits(tmp_path):
    name = "relation-11-history.json"
    index = tmp_path / "sources.json"
    history = tmp_path / name
    payload = {
        "elements": [
            {
                "type": "relation", "id": 11, "version": version,
                "timestamp": timestamp,
                "tags": {"type": "route", "route": "tram", "ref": "1"},
                "members": [
                    {"type": "node", "role": "stop", "ref": i}
                    for i in range(stops)
                ],
            }
            for version, timestamp, stops in (
                (1, "2025-02-20T12:00:00Z", 2),
                (2, "2025-09-01T12:00:00Z", 5),
            )
        ]
    }
    content = json.dumps(payload).encode()
    history.write_bytes(content)
    index.write_text(json.dumps({"osm": [{
        "file": name,
        "url": "https://api.openstreetmap.org/api/0.6/relation/11/history.json",
        "sha256": hashlib.sha256(content).hexdigest(),
    }]}))
    rows, report = build_osm_structure(tmp_path, index)
    assert report["service_history_verified"] is False
    graph = pd.DataFrame(rows, columns=COLUMNS)
    earlier = feature_frame(labels(date(2025, 1, 1), 180), date(2025, 7, 1), direction_graph=graph)
    later = feature_frame(labels(date(2025, 1, 1), 300), date(2025, 11, 1), direction_graph=graph)
    assert earlier.loc[earlier.route == 1].iloc[0].graph_stop_count_mean == 2
    assert later.loc[later.route == 1].iloc[0].graph_stop_count_mean == 5


def test_osm_relation_route_change_and_deactivation_end_prior_state(tmp_path):
    name = "relation-11-history.json"
    index = tmp_path / "sources.json"
    history = tmp_path / name
    versions = []
    for version, day, route, ref in (
        (1, "2025-02-20", "tram", "1"),
        (2, "2025-04-01", "tram", "7"),
        (3, "2025-06-01", "tram_disused", "7"),
    ):
        versions.append({
            "type": "relation", "id": 11, "version": version,
            "timestamp": f"{day}T12:00:00Z",
            "tags": {"type": "route", "route": route, "ref": ref},
            "members": [
                {"type": "node", "role": "stop", "ref": 1},
                {"type": "node", "role": "stop", "ref": 2},
            ],
        })
    content = json.dumps({"elements": versions}).encode()
    history.write_bytes(content)
    index.write_text(json.dumps({"osm": [{
        "file": name,
        "url": "https://api.openstreetmap.org/api/0.6/relation/11/history.json",
        "sha256": hashlib.sha256(content).hexdigest(),
    }]}))
    rows, _ = build_osm_structure(tmp_path, index)
    graph = pd.DataFrame(rows, columns=COLUMNS)
    source = labels(date(2025, 1, 1), 181)
    april = feature_frame(source, date(2025, 4, 15), direction_graph=graph)
    assert april.loc[april.route == 1].iloc[0].graph_available == 0
    assert april.loc[april.route == 7].iloc[0].graph_available == 1
    july = feature_frame(source, date(2025, 7, 1), direction_graph=graph)
    assert july.loc[july.route == 7].iloc[0].graph_available == 0


def test_daily_model_reconciles_its_route_day_prediction_to_hours(monkeypatch):
    class FixedDailyModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _features, _target):
            return self

        def predict(self, features):
            return pd.Series(0.2, index=features.index).to_numpy()

    monkeypatch.setitem(
        sys.modules, "catboost", types.SimpleNamespace(CatBoostRegressor=FixedDailyModel)
    )
    source = labels(date(2025, 1, 1), 150)
    train = make_training_frame(source, date(2025, 5, 30))
    future = feature_frame(source, date(2025, 5, 31))
    baseline = profile_predict(future)
    prediction = _daily_relative_predict(train, future)
    mask = (future.route == 1) & (future.date == date(2025, 5, 31))
    base_day = baseline[mask].sum()
    expected = base_day + 0.25 * 0.2 * (base_day + 300)
    assert prediction[mask].sum() == pytest.approx(expected)


def test_daily_model_full_fold_is_unchanged_by_future_labels(monkeypatch):
    source = labels(date(2025, 1, 1), 181)
    origin = date(2025, 5, 1)
    captured = []

    class CapturingDailyModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, features, target):
            captured.append((features.to_json(), tuple(target)))
            return self

        def predict(self, features):
            captured.append(features.to_json())
            return [0.0] * len(features)

    monkeypatch.setitem(
        sys.modules, "catboost", types.SimpleNamespace(CatBoostRegressor=CapturingDailyModel)
    )
    monkeypatch.setattr("tramflow_ml.competition_cli.model_spec", lambda _name: {})
    first = _evaluate_one(source, origin, "catboost_daily_relative", None)
    changed = source.copy()
    changed.loc[changed.date >= origin, "boardings"] += 100
    second = _evaluate_one(changed, origin, "catboost_daily_relative", None)
    assert captured[0] == captured[2]
    assert captured[1] == captured[3]
    assert first["score"] != second["score"]


@pytest.mark.parametrize("model", ["fourier_ridge_daily", "direct_extratrees_daily"])
def test_time_series_fold_ignores_future_labels(model):
    source = labels(date(2025, 1, 1), 181)
    origin = date(2025, 5, 1)
    future = feature_frame(source, origin)
    history = source.loc[source.date < origin]
    first_prediction = time_series_predict(model, history, future)
    first = _evaluate_one(source, origin, model, None)
    changed = source.copy()
    changed.loc[changed.date >= origin, "boardings"] += 100
    changed_future = feature_frame(changed, origin)
    second_prediction = time_series_predict(
        model, changed.loc[changed.date < origin], changed_future
    )
    second = _evaluate_one(changed, origin, model, None)
    assert (first_prediction == second_prediction).all()
    assert first["score"] != second["score"]


def test_time_series_rejects_incomplete_or_future_history():
    source = labels(date(2025, 1, 1), 120)
    origin = date(2025, 5, 1)
    future = feature_frame(source, origin)
    with pytest.raises(ValueError, match="end the day before"):
        time_series_predict(
            "fourier_ridge_daily", source.loc[source.date < origin - timedelta(days=1)], future
        )
    bad = source.iloc[:-1]
    with pytest.raises(ValueError, match="cover every route/day/hour"):
        time_series_predict("fourier_ridge_daily", bad, future)


def test_time_series_report_distinguishes_history_rows_from_training_examples():
    history = labels(date(2025, 1, 1), 120)
    assert len(history) == 28_800
    assert _training_examples("fourier_ridge_daily", history) == 1_200
    assert _training_examples("direct_extratrees_daily", history) == 110


def test_reconciliation_binds_zero_policy_to_exact_archive(tmp_path):
    archive = tmp_path / "small.zip"
    header = (
        "tran_no;device_no;tran_date_time;begin_date_time;input_date_time;"
        "crd_hashcode;validation_result;tran_type_id;place_id;good_type;"
        "pass_route;ngpt_route;bus_exit_no;garage_number\n"
    )
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr(
            "labels/labels_day_train.csv", "route;date;hour;boardings\n1;2025-08-31;12;1\n"
        )
        out.writestr("labels/labels_day_test.csv", "route;date;hour;boardings\n1;2025-09-01;0;1\n")
        out.writestr(
            "train.csv",
            header
            + "a;d;2025-08-31 12:00:00;;;x;1;;;;;1 трамвай;;\n"
            + "b;d;2025-09-01 00:00:00;;;x;1;;;;;1 трамвай;;\n",
        )
        out.writestr("test.csv", header + "c;d;2025-09-01 01:00:00;;;x;30;;;;;5 трамвай;;\n")
    result = reconcile_archive(archive)
    assert result["raw_keys"] == result["label_keys"] == 2
    result["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps(result), encoding="utf-8")
    assert _verified_proof(archive, proof) == result["archive_sha256"]
    result["archive_sha256"] = "wrong"
    proof.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _verified_proof(archive, proof)


def test_full_fold_model_inputs_do_not_change_with_future_labels(monkeypatch):
    source = labels(date(2025, 1, 1), 181)
    origin = date(2025, 5, 1)
    captured = []

    def fake_predict(model, train, future):
        assert train.date.max() < origin
        captured.append((train.to_json(), future.to_json()))
        return profile_predict(future)

    monkeypatch.setattr("tramflow_ml.competition_cli.predict", fake_predict)
    first = _evaluate_one(source, origin, "hgb", None)
    changed = source.copy()
    changed.loc[changed.date >= origin, "boardings"] += 100
    second = _evaluate_one(changed, origin, "hgb", None)
    assert captured[0] == captured[1]
    assert first["score"] != second["score"]

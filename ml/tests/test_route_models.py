import zipfile
from datetime import date
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest

from tramflow_ml.route_models.data import History, day_index, read_labels
from tramflow_ml.route_models.experiment import environment, metric
from tramflow_ml.route_models.features import DAY_TYPE, profiles, training_snapshots
from tramflow_ml.route_models.models import Candidate, Trained, fit


@pytest.fixture
def history() -> History:
    values = np.arange(304 * 10 * 24, dtype=np.float64).reshape(304, 10, 24) % 97
    values[:, 1, :] = 0
    return History(values, "a" * 64)


def test_post_origin_changes_cannot_change_features_or_training(history: History) -> None:
    changed = history.values.copy()
    changed[151:] = 1e9
    other = History(changed, "b" * 64)
    for left, right in zip(profiles(history, 151, 61), profiles(other, 151, 61), strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(
        training_snapshots(history, 151), training_snapshots(other, 151), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    before = history.values.copy()
    before[150] += 100
    assert not np.array_equal(
        profiles(history, 151, 61)[0], profiles(History(before, "c" * 64), 151, 61)[0]
    )


def test_working_saturday_and_transferred_holidays() -> None:
    assert DAY_TYPE[day_index(date(2025, 11, 1))] == 0
    assert DAY_TYPE[day_index(date(2025, 11, 3))] == 2
    assert DAY_TYPE[day_index(date(2025, 12, 31))] == 2
    assert DAY_TYPE[day_index(date(2025, 5, 2))] == 2


@pytest.mark.parametrize(("origin", "days"), [(27, 61), (305, 61), (304, 62), (304, 0)])
def test_unsupported_history_or_horizon_rejected(history: History, origin: int, days: int) -> None:
    with pytest.raises(ValueError):
        profiles(history, origin, days)


def test_baseline_bundle_roundtrip_and_no_future_saved(history: History, tmp_path: Path) -> None:
    model = fit(history, 151, Candidate("week4", "baseline", "week4"))
    target = tmp_path / "model"
    model.save(target)
    restored = Trained.load(target)
    np.testing.assert_array_equal(model.predict(), restored.predict())
    assert not restored.history.values[151:].any()
    assert not restored.predict()[:, 1].any()


def test_catboost_prediction_and_reload_are_origin_safe(history: History, tmp_path: Path) -> None:
    pytest.importorskip("catboost")
    config = Candidate("test-small", iterations=8, depth=3, native_categories=True)
    changed = history.values.copy()
    changed[60:] = 9999
    first = fit(history, 60, config)
    second = fit(History(changed, "b" * 64), 60, config)
    np.testing.assert_array_equal(first.predict(), second.predict())
    first.save(tmp_path / "model")
    np.testing.assert_array_equal(first.predict(), Trained.load(tmp_path / "model").predict())


def test_metadata_does_not_require_optional_catboost(monkeypatch: pytest.MonkeyPatch) -> None:
    from tramflow_ml.route_models import experiment

    real_version = experiment.version

    def installed(name: str) -> str:
        if name == "catboost":
            raise PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(experiment, "version", installed)
    assert environment()["catboost"] == "not-installed"


def archive(tmp_path: Path, train: str, test: str = "") -> Path:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as stream:
        for part, contents in (("train", train), ("test", test)):
            stream.writestr(
                f"labels/labels_day_{part}.csv", "route;date;hour;boardings\n" + contents
            )
    return path


def test_label_adapter_sums_counts_not_rows(tmp_path: Path) -> None:
    data = read_labels(archive(tmp_path, "17;2025-01-01;8;42\n", "17;2025-09-01;8;7\n"))
    assert data.values.sum() == 49
    assert data.values[0, 5, 8] == 42
    assert len(data.source_hash) == 64
    assert not data.values.flags.writeable


@pytest.mark.parametrize(
    "bad",
    [
        "17;2025-01-01;8;1\n17;2025-01-01;8;2\n",
        "17;2025-11-01;8;1\n",
        "99;2025-01-01;8;1\n",
        "17;2025-01-01;24;1\n",
        "17;2025-01-01;8;-1\n",
        "17;2025-01-01;8;nan\n",
        "17;2025-01-01;8;1.2\n",
    ],
)
def test_bad_labels_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        read_labels(archive(tmp_path, bad))


def test_wape_is_global_and_zero_total_is_undefined() -> None:
    actual = np.array([100.0, 1.0])
    predicted = np.array([90.0, 0.0])
    assert metric(actual, predicted)["wape"] == pytest.approx(11 / 101)
    assert metric(np.zeros(2), np.ones(2))["wape"] is None
    with pytest.raises(ValueError):
        metric(actual, np.array([np.nan, 1.0]))

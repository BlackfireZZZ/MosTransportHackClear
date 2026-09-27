"""Fixed-budget candidates; evaluated future labels never tune fitting or iteration count."""

import json
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from tramflow_ml.route_models.data import FloatArray, History
from tramflow_ml.route_models.features import (
    CALENDAR_NAMES,
    PROFILE_NAMES,
    calendar_features,
    profiles,
    training_snapshots,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    engine: str = "cat"
    features: str = "calendar"
    iterations: int = 500
    depth: int = 6
    loss: str = "MAE"
    decay_days: int | None = None
    native_categories: bool = False
    one_hot_max_size: int = 24
    calibrated: bool = False
    normalized: bool = False
    blend_type4: float = 0.0

    def __post_init__(self) -> None:
        if self.engine not in {"cat", "hgb", "baseline"}:
            raise ValueError("unknown estimator")
        if self.features not in {
            "calendar",
            "weekday",
            "nosummer",
            "snapshot_calendar",
            "history",
            "full",
            "week1",
            "week4",
            "type4",
            "median4",
        }:
            raise ValueError("unknown feature schema")
        if not 1 <= self.iterations <= 2000 or not 2 <= self.depth <= 10:
            raise ValueError("invalid training budget")
        if not 0 <= self.blend_type4 <= 1:
            raise ValueError("invalid blend coefficient")
        if self.decay_days is not None and self.decay_days <= 0:
            raise ValueError("decay must be positive")


CANDIDATES = [
    *[Candidate(name, "baseline", name) for name in ("week1", "week4", "type4", "median4")],
    *[
        Candidate(
            "hgb_" + mode,
            "hgb",
            "full" if mode in {"ratio", "poisson"} else mode,
            iterations=220,
            normalized=mode == "ratio",
            loss="poisson" if mode == "poisson" else "absolute_error",
        )
        for mode in ("snapshot_calendar", "history", "full", "ratio", "poisson")
    ],
    Candidate("cat_history", features="history"),
    Candidate("cat_full", features="full"),
    Candidate("cat_ratio", features="full", normalized=True),
    Candidate("cat_calendar"),
    Candidate("cat_calendar_recent", decay_days=60),
    Candidate("cat_calendar_nosummer", features="nosummer"),
    Candidate("cat_calendar_calibrated", calibrated=True),
    Candidate("cat_deep", iterations=800, depth=8),
    Candidate("cat_native", iterations=800, depth=8, native_categories=True),
    Candidate("cat_weekday", features="weekday", iterations=800, depth=8),
    Candidate("cat_recent120", iterations=800, depth=8, decay_days=120),
    Candidate("cat_recent30", iterations=800, depth=8, decay_days=30),
    Candidate("cat_blend", blend_type4=0.5),
    Candidate("cat_native_rmse", iterations=800, depth=8, native_categories=True, loss="RMSE"),
    Candidate("cat_ctr", iterations=800, depth=8, native_categories=True, one_hot_max_size=2),
]


def columns(candidate: Candidate) -> list[int]:
    if candidate.features in {"calendar", "nosummer", "weekday"}:
        return {"calendar": [0, 1, 2, 3, 4, 5], "nosummer": [0, 1, 2, 3, 5], "weekday": [0, 1, 2]}[
            candidate.features
        ]
    if candidate.features == "snapshot_calendar":
        return list(range(8))
    return list(range(12 if candidate.features == "history" else len(PROFILE_NAMES)))


def native(values: FloatArray, candidate: Candidate) -> Any:
    if not candidate.native_categories:
        return values
    result = values.astype(object)
    for col in range(min(4, result.shape[1])):
        result[:, col] = result[:, col].astype(int).astype(str)
    return result


@dataclass
class Trained:
    candidate: Candidate
    origin: int
    history: History
    estimator: Any = None
    calibration: FloatArray = field(default_factory=lambda: np.ones(10))

    def predict(self, days: int = 61) -> FloatArray:
        candidate = self.candidate
        features, scale = profiles(self.history, self.origin, days)
        if candidate.engine == "baseline":
            index = {"week1": 8, "week4": 10, "type4": 12, "median4": 13}[candidate.features]
            prediction = features[:, index]
        else:
            if candidate.features in {"calendar", "weekday", "nosummer"}:
                features = calendar_features(self.origin, days)
            prediction = np.asarray(
                self.estimator.predict(native(features[:, columns(candidate)], candidate)),
                dtype=np.float64,
            )
            if candidate.normalized:
                prediction = prediction * scale
        prediction = prediction.reshape(days, 10, 24) * self.calibration[None, :, None]
        if candidate.blend_type4:
            reference, _ = profiles(self.history, self.origin, days)
            prediction = (
                1 - candidate.blend_type4
            ) * prediction + candidate.blend_type4 * reference[:, 12].reshape(days, 10, 24)
        cold_routes = self.history.values[: self.origin].sum(axis=(0, 2)) == 0
        prediction[:, cold_routes, :] = 0
        if not np.isfinite(prediction).all():
            raise ValueError("estimator emitted nonfinite predictions")
        return np.rint(np.maximum(0.0, prediction))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        values = self.history.values.copy()
        values[self.origin :] = 0
        np.savez_compressed(path / "history.npz", values=values)
        metadata = {
            "candidate": asdict(self.candidate),
            "origin": self.origin,
            "source_hash": self.history.source_hash,
            "calibration": self.calibration.tolist(),
        }
        (path / "model.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if self.candidate.engine == "cat":
            self.estimator.save_model(str(path / "model.cbm"))
        elif self.candidate.engine == "hgb":
            import_module("joblib").dump(self.estimator, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> "Trained":
        """Load a locally trusted model bundle; sklearn joblib is executable serialization."""
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        candidate = Candidate(**metadata["candidate"])
        with np.load(path / "history.npz", allow_pickle=False) as archive:
            history = History(archive["values"], metadata["source_hash"])
        estimator = None
        if candidate.engine == "cat":
            estimator = import_module("catboost").CatBoostRegressor()
            estimator.load_model(str(path / "model.cbm"))
        elif candidate.engine == "hgb":
            estimator = import_module("joblib").load(path / "model.joblib")
        return cls(
            candidate,
            metadata["origin"],
            history,
            estimator,
            np.array(metadata["calibration"], dtype=np.float64),
        )


def fit(history: History, origin: int, candidate: Candidate) -> Trained:
    if not 35 <= origin <= len(history.values):
        raise ValueError("origin requires at least 35 days of history")
    result = Trained(candidate, origin, history)
    if candidate.engine == "baseline":
        return result
    if candidate.features in {"calendar", "weekday", "nosummer"}:
        features = calendar_features(0, origin)
        targets = history.values[:origin].reshape(-1)
        weights = (
            np.repeat(np.exp((np.arange(origin) - origin) / candidate.decay_days), 240)
            if candidate.decay_days
            else None
        )
    else:
        features, targets, scale = training_snapshots(history, origin)
        weights = scale if candidate.normalized else None
        if candidate.normalized:
            targets = targets / scale
    features = features[:, columns(candidate)]
    if candidate.engine == "cat":
        result.estimator = import_module("catboost").CatBoostRegressor(
            iterations=candidate.iterations,
            depth=candidate.depth,
            learning_rate=0.06,
            loss_function=candidate.loss,
            thread_count=4,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            has_time=True,
            cat_features=list(range(min(4, features.shape[1])))
            if candidate.native_categories
            else [],
            one_hot_max_size=candidate.one_hot_max_size,
        )
    else:
        result.estimator = import_module("sklearn.ensemble").HistGradientBoostingRegressor(
            loss=candidate.loss,
            max_iter=candidate.iterations,
            max_leaf_nodes=23,
            l2_regularization=10,
            learning_rate=0.07,
            early_stopping=False,
            random_state=42,
            categorical_features=[0, 1, 2, 3],
        )
    with import_module("threadpoolctl").threadpool_limits(limits=4):
        result.estimator.fit(native(features, candidate), targets, sample_weight=weights)
    if candidate.calibrated:
        features = calendar_features(origin - 28, 28)[:, columns(candidate)]
        fitted = np.maximum(0, result.estimator.predict(native(features, candidate)))
        denominator = fitted.reshape(28, 10, 24).sum(axis=(0, 2))
        result.calibration = history.values[origin - 28 : origin].sum(axis=(0, 2)) / np.maximum(
            1, denominator
        )
    return result


def feature_names(candidate: Candidate) -> list[str]:
    names = (
        CALENDAR_NAMES
        if candidate.features in {"calendar", "weekday", "nosummer"}
        else (PROFILE_NAMES)
    )
    return [names[i] for i in columns(candidate)]

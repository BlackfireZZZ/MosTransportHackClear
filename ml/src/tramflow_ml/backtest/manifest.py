"""The experiment manifest: what ran, on which data, under which rules.

Nothing here reads the wall clock, a filesystem path or a dict iteration order, so two
runs of the same configuration over the same data produce the same manifest byte for
byte. A manifest that changed every run would identify nothing.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tramflow_ml.backtest.records import BACKTEST_VERSION, BacktestError
from tramflow_ml.features import FEATURE_VERSION
from tramflow_ml.features.records import digest_of

MANIFEST_VERSION = "backtest-manifest.v1"


@dataclass(frozen=True, slots=True, order=True)
class ModelVersion:
    """A model identified by name and version, the pair a result is attributed to."""

    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise BacktestError("a model must carry a non-empty name and version")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Hashes of the configuration, the data and the folds, plus the horizon reports."""

    target: str
    unit: str
    config_hash: str
    data_hash: str
    fold_hash: str
    models: tuple[ModelVersion, ...]
    horizons: tuple[Mapping[str, object], ...]

    def body(self) -> dict[str, object]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "backtest_version": BACKTEST_VERSION,
            "feature_version": FEATURE_VERSION,
            "target": self.target,
            "unit": self.unit,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "fold_hash": self.fold_hash,
            "models": [model.to_dict() for model in self.models],
            "horizons": [dict(horizon) for horizon in self.horizons],
        }

    @property
    def manifest_hash(self) -> str:
        return digest_of(self.body())

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "manifest_hash": self.manifest_hash}


def build_manifest(
    target: str,
    unit: str,
    config_hash: str,
    data_hash: str,
    fold_hash: str,
    models: Sequence[ModelVersion],
    horizons: Sequence[Mapping[str, object]],
) -> ExperimentManifest:
    """Model order follows the caller's, so a manifest records who was compared with whom."""
    return ExperimentManifest(
        target=target,
        unit=unit,
        config_hash=config_hash,
        data_hash=data_hash,
        fold_hash=fold_hash,
        models=tuple(models),
        horizons=tuple(horizons),
    )

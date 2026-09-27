"""Regression checks for the user-uploaded route-hour ensemble."""

import hashlib
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from tramflow_ml.competition import validate_submission
from tramflow_ml.competition_cli import main as competition_main

ROOT = Path(__file__).parents[1] / "competition_submissions" / "2026-09-27"
BEST = ROOT / "submission-catboost50-uniform-recency60-routeshape.csv"


def test_best_uploaded_csv_has_exact_grid_and_known_bytes() -> None:
    assert hashlib.sha256(BEST.read_bytes()).hexdigest() == (
        "edc07cca26ff21dd5228f744dea79940eb81a65c3ce5368aa743e2fc2fd0f11f"
    )
    rows = pd.read_csv(BEST, sep=";")
    validate_submission(rows)
    assert len(rows) == 14_640
    assert (rows.prediction >= 0).all()
    assert (rows.prediction % 1 == 0).all()


@pytest.mark.parametrize(
    "module_name",
    (
        "overnight_catboost_daily50",
        "overnight_catboost_recency",
        "overnight_catboost_recency_ensemble_submit",
    ),
)
def test_scripts_refuse_unverified_archive_before_zero_fill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module_name: str
) -> None:
    monkeypatch.setenv("TRAMFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(ROOT))
    for name in (
        "overnight_catboost_daily50",
        "overnight_catboost_recency",
        "overnight_catboost_recency_ensemble_submit",
    ):
        sys.modules.pop(name, None)
    module = importlib.import_module(module_name)

    def reject(*_args: object) -> str:
        raise ValueError("archive proof rejected")

    monkeypatch.setattr(module, "_verified_proof", reject)
    with pytest.raises(ValueError, match="archive proof rejected"):
        module.main()


def test_submit_rejects_retrospective_weather(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected(*_args: object) -> str:
        raise AssertionError("invalid submit must fail before reading external data")

    monkeypatch.setattr("tramflow_ml.competition_cli._verified_proof", unexpected)
    monkeypatch.setattr("tramflow_ml.competition_cli.load_weather", unexpected)
    monkeypatch.setattr(
        sys, "argv",
        [
            "tramflow-competition", "--archive", str(tmp_path / "data.zip"),
            "--reconciliation", str(tmp_path / "proof.json"),
            "--weather", str(tmp_path / "actual-weather.csv"),
            "submit", "--model", "catboost_weather_relative",
            "--output", str(tmp_path / "submission.csv"),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        competition_main()
    assert "retrospective actual weather" in capsys.readouterr().err

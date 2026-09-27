import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_evaluation(dataset: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tramflow_ml.cli",
            "evaluate",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_cli_writes_passing_golden_report(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "evaluation.json"
    result = run_evaluation(Path("ml/evals/golden_cases.json"), output)

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report == json.loads(result.stdout)
    assert report["passed"] is True
    assert set(report["by_horizon"]) == {"day", "month", "year"}


def test_cli_fails_when_year_coverage_is_hidden_by_overall(tmp_path: Path) -> None:
    cases = [
        {
            "id": horizon,
            "horizon": horizon,
            "scenario": "coverage regression",
            "actual": [11 if horizon == "year" else 10] * count,
            "prediction": [10] * count,
            "baseline_prediction": [0] * count,
            "lower_bound": [10] * count,
            "upper_bound": [10] * count,
        }
        for horizon, count in [("day", 8), ("month", 1), ("year", 1)]
    ]
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps(cases))
    output = tmp_path / "evaluation.json"

    result = run_evaluation(dataset, output)

    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text())
    assert report == json.loads(result.stdout)
    assert report["overall"]["interval_coverage"] == 0.9
    assert report["by_horizon"]["year"]["interval_coverage"] == 0
    assert report["passed"] is False


@pytest.mark.parametrize(
    ("invalid_input", "diagnostic"),
    [
        ("nonfinite", "passenger counts and interval bounds must be finite"),
        ("missing_horizon", "missing required horizons: year"),
        ("zero_demand", "year: zero passenger demand; WAPE is undefined"),
    ],
)
def test_cli_rejects_invalid_input_without_report(
    tmp_path: Path, invalid_input: str, diagnostic: str
) -> None:
    cases = json.loads(Path("ml/evals/golden_cases.json").read_text())
    if invalid_input == "nonfinite":
        cases[0]["actual"][0] = float("nan")
    elif invalid_input == "missing_horizon":
        cases = [case for case in cases if case["horizon"] != "year"]
    else:
        for case in cases:
            if case["horizon"] == "year":
                case["actual"] = [0] * len(case["actual"])
    dataset = tmp_path / "invalid.json"
    dataset.write_text(json.dumps(cases))
    output = tmp_path / "evaluation.json"

    result = run_evaluation(dataset, output)

    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not output.exists()
    assert not result.stdout.strip()

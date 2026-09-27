import hashlib
import json
from pathlib import Path

import pytest

from app.infrastructure.scored_bundle import (
    load_approved_scored_bundle,
    load_scored_bundle,
)


def copy_current_bundle(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = Path(__file__).resolve().parents[2] / "ml/competition_submissions"
    manifest = json.loads((root / "current.json").read_text())
    for key in ("csv", "producer", "evaluation"):
        relative = manifest[key]
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    path = tmp_path / "current.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_current_bundle_is_complete_and_bound_to_evaluation() -> None:
    root = Path(__file__).resolve().parents[2]
    bundle = load_scored_bundle(root / "ml/competition_submissions/current.json")
    assert len(bundle.points) == 14640
    assert bundle.model_version == "catboost50-uniform-recency60-routeshape-v1"
    assert bundle.csv_sha256 == (
        "edc07cca26ff21dd5228f744dea79940eb81a65c3ce5368aa743e2fc2fd0f11f"
    )
    assert load_approved_scored_bundle(
        root / "ml/competition_submissions/current.json",
        root / "ml/competition_submissions/approved.json",
    ) == bundle


def test_changed_report_needs_new_approval(tmp_path: Path) -> None:
    path, manifest = copy_current_bundle(tmp_path)
    report_path = tmp_path / manifest["evaluation"]
    report = json.loads(report_path.read_text())
    for origin in report["origins"]:
        origin["candidate_score"] = 0.99
    report_bytes = json.dumps(report).encode()
    report_path.write_bytes(report_bytes)
    manifest["evaluation_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_scored_bundle(path).model_version == manifest["model_version"]
    root = Path(__file__).resolve().parents[2] / "ml/competition_submissions"
    with pytest.raises(ValueError, match="no approved evaluation"):
        load_approved_scored_bundle(path, root / "approved.json")


def test_quality_window_must_end_before_scored_period(tmp_path: Path) -> None:
    path, manifest = copy_current_bundle(tmp_path)
    report_path = tmp_path / manifest["evaluation"]
    report = json.loads(report_path.read_text())
    report["origins"][-1]["origin"] = "2025-10-01"
    report_bytes = json.dumps(report).encode()
    report_path.write_bytes(report_bytes)
    manifest["evaluation_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="quality origins"):
        load_scored_bundle(path)


def test_quality_gate_rejects_compensated_temporal_collapse(tmp_path: Path) -> None:
    path, manifest = copy_current_bundle(tmp_path)
    report_path = tmp_path / manifest["evaluation"]
    report = json.loads(report_path.read_text())
    for item, candidate in zip(report["origins"], (0.95, 0.95, 0.1), strict=True):
        item["baseline_score"] = 0.5
        item["candidate_score"] = candidate
    report_bytes = json.dumps(report).encode()
    report_path.write_bytes(report_bytes)
    manifest["evaluation_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="quality gate"):
        load_scored_bundle(path)


def test_another_model_csv_can_use_the_same_publication_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2] / "ml/competition_submissions"
    current = load_scored_bundle(root / "current.json")
    manifest = json.loads((root / "current.json").read_text())
    csv_bytes = (
        root / "2026-09-27/submission-calendar_robust_28_routeshape.csv"
    ).read_bytes()
    (tmp_path / "candidate.csv").write_bytes(csv_bytes)
    producer = b"candidate configuration v2\n"
    (tmp_path / "producer.txt").write_bytes(producer)
    manifest.update(
        csv="candidate.csv", csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
        producer="producer.txt", config_sha256=hashlib.sha256(producer).hexdigest(),
        model_version="alternate-v2", feature_version="alternate-features-v2",
        evaluation="evaluation.json",
    )
    report = {
        "schema_version": "tramflow.route-hour-evaluation.v1",
        "model_version": "alternate-v2",
        "config_sha256": manifest["config_sha256"],
        "csv_sha256": manifest["csv_sha256"],
        "baseline_model_version": "fixture-baseline-v1",
        "evidence": "synthetic contract fixture",
        "horizon_days": 61,
        "origins": [
            {"origin": origin, "baseline_score": 0.7, "candidate_score": 0.8}
            for origin in ("2025-05-01", "2025-07-01", "2025-09-01")
        ],
    }
    report_bytes = json.dumps(report).encode()
    (tmp_path / "evaluation.json").write_bytes(report_bytes)
    manifest["evaluation_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    path = tmp_path / "current.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = load_scored_bundle(path)
    assert len(candidate.points) == 14640
    assert candidate.manifest_sha256 != current.manifest_sha256
    assert candidate.model_version == "alternate-v2"


def test_bundle_rejects_csv_mismatch_before_publication(tmp_path: Path) -> None:
    (tmp_path / "predictions.csv").write_text(
        "route;date;hour;prediction\n1;2025-11-01;0;1\n", encoding="utf-8"
    )
    producer = b"pass\n"
    config_sha256 = hashlib.sha256(producer).hexdigest()
    report = {
        "schema_version": "tramflow.route-hour-evaluation.v1",
        "model_version": "candidate-v1",
        "config_sha256": config_sha256,
        "csv_sha256": "0" * 64,
        "baseline_model_version": "baseline-v1",
        "evidence": "test fixture",
        "horizon_days": 61,
        "origins": [
            {"origin": origin, "baseline_score": 0.7, "candidate_score": 0.8}
            for origin in ("2025-05-01", "2025-07-01", "2025-09-01")
        ],
    }
    report_bytes = json.dumps(report).encode()
    (tmp_path / "evaluation.json").write_bytes(report_bytes)
    (tmp_path / "producer.py").write_bytes(producer)
    manifest = {
        "schema_version": "tramflow.scored-route-hour.v1",
        "csv": "predictions.csv",
        "csv_sha256": "0" * 64,
        "archive_sha256": "b" * 64,
        "model_version": "candidate-v1",
        "feature_version": "feature-v1",
        "calendar_version": "calendar-v1",
        "source_version": "organizer-archive-" + "b" * 16,
        "config_sha256": config_sha256,
        "producer": "producer.py",
        "producer_commit": "a" * 40,
        "forecast_origin": "2025-11-01T00:00:00+03:00",
        "data_cutoff": "2025-10-31T23:59:59+03:00",
        "generated_at": "2026-09-27T09:54:21+03:00",
        "target": "validation_count",
        "unit": "event_count",
        "evaluation": "evaluation.json",
        "evaluation_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    path = tmp_path / "current.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_scored_bundle(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"data_cutoff": "2025-11-01T00:00:01+03:00"}, "origin/cutoff"),
        ({"csv": "../submission.csv"}, "escapes"),
        ({"evaluation_sha256": "0" * 64}, "checksum"),
        ({"source_version": "unbound-v1"}, "source version"),
    ],
)
def test_bundle_rejects_inconsistent_metadata(
    tmp_path: Path, change: dict[str, str], message: str
) -> None:
    root = Path(__file__).resolve().parents[2] / "ml/competition_submissions"
    original_manifest = json.loads((root / "current.json").read_text())
    manifest = dict(original_manifest)
    manifest.update(change)
    if change.get("csv") == "../submission.csv":
        (tmp_path / "submission.csv").write_bytes((root / original_manifest["csv"]).read_bytes())
    copied = tmp_path / "bundle"
    copied.mkdir()
    for key in ("csv", "producer", "evaluation"):
        original = original_manifest[key]
        destination = copied / original
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / original).read_bytes())
    candidate = copied / "current.json"
    candidate.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_scored_bundle(candidate)


def test_bundle_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "current.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_scored_bundle(path)


@pytest.mark.parametrize("bad_score", [0.7, 0.6, 1.01, float("nan")])
def test_bundle_rejects_failed_quality_gate(tmp_path: Path, bad_score: float) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "ml/competition_submissions/current.json").read_text())
    report_path = root / "ml/competition_submissions" / manifest["evaluation"]
    report = json.loads(report_path.read_text())
    report["origins"][0]["candidate_score"] = bad_score
    report["origins"][1]["candidate_score"] = 0.0
    report["origins"][2]["candidate_score"] = 0.0
    report_bytes = json.dumps(report).encode()
    (tmp_path / "evaluation.json").write_bytes(report_bytes)
    manifest["evaluation"] = "evaluation.json"
    manifest["evaluation_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    csv_bytes = (root / "ml/competition_submissions" / manifest["csv"]).read_bytes()
    (tmp_path / "predictions.csv").write_bytes(csv_bytes)
    manifest["csv"] = "predictions.csv"
    producer_bytes = (root / "ml/competition_submissions" / manifest["producer"]).read_bytes()
    (tmp_path / "producer.py").write_bytes(producer_bytes)
    manifest["producer"] = "producer.py"
    path = tmp_path / "current.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="quality"):
        load_scored_bundle(path)

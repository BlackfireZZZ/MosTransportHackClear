"""Validate a versioned route-hour submission before database publication."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

MOSCOW = ZoneInfo("Europe/Moscow")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}\Z")
MANIFEST_FIELDS = frozenset({
    "schema_version", "csv", "csv_sha256", "archive_sha256", "producer",
    "config_sha256", "evaluation", "evaluation_sha256", "model_version",
    "feature_version", "calendar_version", "source_version", "forecast_origin",
    "data_cutoff", "generated_at", "producer_commit", "target", "unit",
})
REPORT_FIELDS = frozenset({
    "schema_version", "model_version", "config_sha256", "csv_sha256",
    "baseline_model_version", "evidence", "horizon_days", "origins",
})


@dataclass(frozen=True)
class ScoredBundle:
    manifest_sha256: str
    csv_sha256: str
    evaluation_sha256: str
    archive_sha256: str
    model_version: str
    feature_version: str
    calendar_version: str
    source_version: str
    config_sha256: str
    producer_commit: str
    forecast_origin: datetime
    data_cutoff: datetime
    generated_at: datetime
    points: dict[tuple[int, date, int], int]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("artifact metadata has a duplicate JSON key")
        value[key] = item
    return value


def _read_json(path: Path, limit: int, expected_sha256: str | None = None) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("artifact metadata exceeds size limit")
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("artifact checksum mismatch")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact metadata is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("artifact metadata must be an object")
    return value


def _fields(value: dict[str, Any], expected: frozenset[str]) -> None:
    if value.keys() != expected:
        raise ValueError("artifact metadata fields differ from the versioned contract")


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _version(value: Any, name: str) -> str:
    if not isinstance(value, str) or VERSION.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded version identifier")
    return value


def _relative_file(root: Path, raw: Any, expected_sha256: str, limit: int) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError("artifact path must be relative to the bundle")
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("artifact path escapes or is absent from the bundle")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("artifact exceeds size limit")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("artifact checksum mismatch")
    return path


def _timestamp(raw: Any, name: str) -> datetime:
    if not isinstance(raw, str):
        raise ValueError(f"{name} must have an explicit offset")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO timestamp") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must have an explicit offset")
    return value.astimezone(MOSCOW)


def _quality(report: dict[str, Any], manifest: dict[str, Any], cutoff: date) -> None:
    _fields(report, REPORT_FIELDS)
    if report["schema_version"] != "tramflow.route-hour-evaluation.v1":
        raise ValueError("quality report schema is unsupported")
    for name in ("model_version", "config_sha256", "csv_sha256"):
        if report[name] != manifest[name]:
            raise ValueError("quality report is not bound to the artifact")
    _version(report["baseline_model_version"], "baseline_model_version")
    if not isinstance(report["evidence"], str) or not report["evidence"].strip():
        raise ValueError("quality report needs a reproducible evidence reference")
    if type(report["horizon_days"]) is not int or report["horizon_days"] != 61:
        raise ValueError("quality report must cover complete 61-day horizons")
    origins = report["origins"]
    if not isinstance(origins, list) or len(origins) < 3:
        raise ValueError("quality report needs three temporal slices")
    dates: list[date] = []
    gains = 0
    for item in origins:
        if not isinstance(item, dict) or item.keys() != {
            "origin", "baseline_score", "candidate_score"
        }:
            raise ValueError("quality origin fields differ from the contract")
        if not isinstance(item["origin"], str):
            raise ValueError("quality origin must be a date")
        try:
            origin = date.fromisoformat(item["origin"])
        except ValueError as error:
            raise ValueError("quality origin must be a date") from error
        dates.append(origin)
        for score in (item["baseline_score"], item["candidate_score"]):
            if isinstance(score, bool) or not isinstance(score, (float, int)) \
                    or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("quality scores must be finite values from zero to one")
        gains += item["candidate_score"] > item["baseline_score"]
    if dates != sorted(set(dates)) or any(
        (right - left).days < 45 for left, right in zip(dates, dates[1:], strict=False)
    ) or any(day + timedelta(days=60) > cutoff for day in dates):
        raise ValueError("quality origins must be separated chronological history")
    if gains < 2 or any(
        item["candidate_score"] < item["baseline_score"] for item in origins
    ):
        raise ValueError("quality gate failed on temporal slices")


def load_scored_bundle(path: Path) -> ScoredBundle:
    manifest = _read_json(path, 64 * 1024)
    _fields(manifest, MANIFEST_FIELDS)
    if manifest["schema_version"] != "tramflow.scored-route-hour.v1":
        raise ValueError("unsupported scored bundle schema")
    for name in ("csv_sha256", "archive_sha256", "config_sha256", "evaluation_sha256"):
        _digest(manifest[name], name)
    if not isinstance(manifest["producer_commit"], str) or re.fullmatch(
        r"[a-f0-9]{40}", manifest["producer_commit"]
    ) is None:
        raise ValueError("producer commit must be a full Git SHA")
    for name in ("model_version", "feature_version", "calendar_version", "source_version"):
        _version(manifest[name], name)
    if manifest["source_version"] != f"organizer-archive-{manifest['archive_sha256'][:16]}":
        raise ValueError("source version does not match the organizer archive")
    if (manifest["target"], manifest["unit"]) != ("validation_count", "event_count"):
        raise ValueError("scored bundle target/unit differs from organizer contract")
    origin = _timestamp(manifest["forecast_origin"], "forecast_origin")
    cutoff = _timestamp(manifest["data_cutoff"], "data_cutoff")
    generated = _timestamp(manifest["generated_at"], "generated_at")
    if origin != datetime(2025, 11, 1, tzinfo=MOSCOW) or not cutoff < origin:
        raise ValueError("scored bundle origin/cutoff differs from organizer contract")
    if generated < origin:
        raise ValueError("artifact generation precedes its forecast origin")
    root = path.resolve().parent
    _relative_file(root, manifest["producer"], manifest["config_sha256"], 2 * 1024 * 1024)
    report_path = _relative_file(
        root, manifest["evaluation"], manifest["evaluation_sha256"], 256 * 1024
    )
    _quality(
        _read_json(report_path, 256 * 1024, manifest["evaluation_sha256"]),
        manifest, cutoff.date(),
    )
    csv_path = _relative_file(root, manifest["csv"], manifest["csv_sha256"], 2 * 1024 * 1024)
    from app.infrastructure.forecast_publication import load_scored_csv

    points = load_scored_csv(csv_path, manifest["csv_sha256"])
    digest = hashlib.sha256(json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return ScoredBundle(
        manifest_sha256=digest,
        csv_sha256=manifest["csv_sha256"],
        evaluation_sha256=manifest["evaluation_sha256"],
        archive_sha256=manifest["archive_sha256"],
        model_version=manifest["model_version"],
        feature_version=manifest["feature_version"],
        calendar_version=manifest["calendar_version"],
        source_version=manifest["source_version"],
        config_sha256=manifest["config_sha256"],
        producer_commit=manifest["producer_commit"],
        forecast_origin=origin,
        data_cutoff=cutoff,
        generated_at=generated,
        points=points,
    )


def load_approved_scored_bundle(manifest_path: Path, registry_path: Path) -> ScoredBundle:
    bundle = load_scored_bundle(manifest_path)
    registry = _read_json(registry_path, 64 * 1024)
    if registry.keys() != {"schema_version", "approved"} or registry["schema_version"] != (
        "tramflow.scored-route-hour-approvals.v1"
    ):
        raise ValueError("approval registry schema is unsupported")
    entries = registry["approved"]
    if not isinstance(entries, list):
        raise ValueError("approval registry entries must be a list")
    identity = {
        "manifest_sha256": bundle.manifest_sha256,
        "csv_sha256": bundle.csv_sha256,
        "evaluation_sha256": bundle.evaluation_sha256,
        "producer_commit": bundle.producer_commit,
        "model_version": bundle.model_version,
    }
    if identity not in entries:
        raise ValueError("scored bundle has no approved evaluation and provenance")
    return bundle

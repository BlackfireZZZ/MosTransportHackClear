"""Calendar-only experimental datasets from hash-bound retrospective stop inference."""

import csv
import gzip
import io
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .audit import digest, write_json
from .service_policy import PolicyDecision, load_service_policy

SPLITS = ("train", "development", "diagnostic")
LABEL_ORIGIN = "inferred_conditional_uncalibrated"
MIXED_PRECISION_REASON = "mixed_schedule_precision_uncalibrated"
COMMON_FIELDS = ["day", "route", "event_hour", "stop_id", "direction"]
SUFFIX_FIELDS = ["applicable_pattern", "hour", "weekday", "label_origin", "training_eligible"]


def _checked_hash(path: Path, expected: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"reconstruction source missing or changed: {path.name}")


def _number(value: str, *, integer: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or integer and not number.is_integer():
        raise ValueError("source counts must be finite nonnegative values")
    return number


def _day_sources(
    run: Path, source: dict[str, Any], *, require_patterns: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    if source.get("schema_version") != "boarding-real-run.v1" or source.get("complete") is not True:
        raise ValueError("completed boarding-real-run.v1 reconstruction required")
    composer_hash = source.get("implementation", {}).get("composition.py", "")
    if (not isinstance(composer_hash, str) or len(composer_hash) != 64
            or any(c not in "0123456789abcdef" for c in composer_hash)):
        raise ValueError("source composition implementation SHA256 required")
    days = source.get("config", {}).get("dates", [])
    if not days or len(set(days)) != len(days):
        raise ValueError("unique nonempty reconstruction dates required")
    for day in days:
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError("canonical civil dates required")
    if set(source.get("day_receipts", {})) != set(days):
        raise ValueError("reconstruction receipts do not cover configured dates")
    reports = source.get("reports", [])
    if len(reports) != len(days) or {r["day"] for r in reports} != set(days):
        raise ValueError("reconstruction reports do not cover configured dates")
    indexed = {r["day"]: r for r in reports}
    checked = []
    required = {"stop_hour_soft_counts.csv.gz", "stop_hour_weak_counts.csv",
                "mass_ledger.csv", "evaluation.json"}
    if require_patterns:
        required.add("patterns.json")
    for day in sorted(days):
        folder = run / day
        receipt = folder / "receipt.json"
        _checked_hash(receipt, source["day_receipts"][day])
        files = json.loads(receipt.read_text())["files"]
        if not required.issubset(files):
            raise ValueError("incomplete reconstruction receipt")
        for name, sha in files.items():
            if Path(name).name != name or name in {".", "..", "receipt.json"}:
                raise ValueError("unsafe reconstruction receipt filename")
            _checked_hash(folder / name, sha)
        report = json.loads((folder / "evaluation.json").read_text())
        if report != indexed[day]:
            raise ValueError("reconstruction day report disagrees with manifest")
        checked.append((day, report))
    for field in ("source_success", "assigned_weak", "ambiguous", "unassigned"):
        if source.get(field) != sum(r[field] for _, r in checked):
            raise ValueError("reconstruction aggregate report mismatch")
    return checked


def _hour(raw: str, day: str) -> datetime:
    instant = datetime.strptime(raw, "%Y-%m-%d %H")
    if instant.strftime("%Y-%m-%d %H") != raw or str(instant.date()) != day:
        raise ValueError("source-hour row escapes its civil date")
    return instant.replace(tzinfo=ZoneInfo("Europe/Moscow"))


def _ledger(folder: Path, day: str) -> dict[tuple[str, str], tuple[int, int, int]]:
    ledger = {}
    with (folder / "mass_ledger.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            _hour(row["event_hour"], day)
            key = row["route"], row["event_hour"]
            if not key[0] or key in ledger:
                raise ValueError("duplicate or empty source mass key")
            values = [int(_number(row.get(status, "0"), integer=True)) for status in (
                "inferred_stable", "ambiguous", "unassigned",
            )]
            ledger[key] = values[0], values[1], values[2]
    return ledger


def _rows(
    folder: Path, day: str, *, soft: bool,
) -> dict[tuple[str, str, str, str, bool], float]:
    name = "stop_hour_soft_counts.csv.gz" if soft else "stop_hour_weak_counts.csv"
    groups: dict[tuple[str, str, str, str, bool], list[float]] = defaultdict(list)
    hours = set()
    with (gzip.open(folder / name, "rt", newline="") if soft
          else (folder / name).open(newline="")) as handle:
        for row in csv.DictReader(handle):
            raw_hour = row["event_hour"]
            if raw_hour not in hours:
                _hour(raw_hour, day)
                hours.add(raw_hour)
            if any(not row[k] for k in ("route", "stop_id", "direction")):
                raise ValueError("stop identity cannot be empty")
            applicable = True
            if soft:
                if row["applicable_pattern"] not in {"True", "False"}:
                    raise ValueError("invalid applicable_pattern flag")
                applicable = row["applicable_pattern"] == "True"
            key = raw_hour, row["route"], row["stop_id"], row["direction"], applicable
            groups[key].append(_number(row["expected_count" if soft else "inferred_count"],
                                       integer=not soft))
    return {key: math.fsum(values) for key, values in groups.items()}


def _verify_mass(
    rows: dict[tuple[str, str, str, str, bool], float],
    ledger: dict[tuple[str, str], tuple[int, int, int]],
    *, soft: bool,
) -> None:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (hour, route, _, _, _), value in rows.items():
        totals[(route, hour)].append(value)
    if set(totals) - set(ledger):
        raise ValueError("stop counts contain unknown route-hour mass")
    for key, (strict, ambiguous, _) in ledger.items():
        expected = strict + ambiguous if soft else strict
        if not math.isclose(math.fsum(totals[key]), expected, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("stop counts violate original route-hour mass")


def export_real_dataset(
    run: Path, out: Path, *, service_policy: Path | None = None,
) -> dict[str, Any]:
    """Export observed soft/strict cells; missing cells are never interpreted as zero.

    training_eligible marks applicable-pattern experimental targets only. Source
    histories and retrospective posteriors are not point-in-time verified labels.
    """
    if out.exists() and any(out.iterdir()):
        raise ValueError("dataset output must be empty")
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("completed reconstruction manifest required")
    source = json.loads(manifest_path.read_text())
    source_hash = digest(manifest_path)
    policy = load_service_policy(service_policy) if service_policy is not None else None
    days = _day_sources(run, source, require_patterns=policy is not None)
    out.mkdir(parents=True, exist_ok=True)
    stats = {f"{kind}_{split}": {"rows": 0, "count": 0.0, "blocked_pattern_count": 0.0,
                                "algorithmic_affected_count": 0.0}
             for kind in ("soft", "weak") for split in SPLITS}
    if policy is not None:
        for stat in stats.values():
            stat.update(policy_blocked_count=0.0, newly_policy_blocked_count=0.0,
                        restored_engine_count=0.0)
    paths = {name: out / f"{name}.csv.gz" for name in stats}
    with ExitStack() as stack:
        writers = {}
        for name, path in paths.items():
            binary = stack.enter_context(path.with_suffix(".gz.tmp").open("wb"))
            compressed = stack.enter_context(gzip.GzipFile(
                fileobj=binary, filename="", mode="wb", mtime=0, compresslevel=1,
            ))
            text = stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8", newline=""))
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(COMMON_FIELDS + ["expected_count" if name.startswith("soft")
                                            else "inferred_count"] + SUFFIX_FIELDS
                            + ["algorithmic_quality_reasons"]
                            + (["engine_applicable_pattern", "service_policy_reasons"]
                               if policy is not None else []))
            writers[name] = writer
        for day, report in days:
            folder = run / day
            ledger = _ledger(folder, day)
            patterns = json.loads((folder / "patterns.json").read_text()) if policy else {}
            decisions: dict[tuple[str, str, bool, str], PolicyDecision] = {}
            for index, field in enumerate(("assigned_weak", "ambiguous", "unassigned")):
                if sum(values[index] for values in ledger.values()) != report[field]:
                    raise ValueError("source ledger disagrees with reconstruction report")
            if sum(sum(values) for values in ledger.values()) != report["source_success"]:
                raise ValueError("source success disagrees with reconstruction ledger")
            split = "train" if day < "2025-07-01" else (
                "development" if day < "2025-09-01" else "diagnostic"
            )
            for kind in ("soft", "weak"):
                rows = _rows(folder, day, soft=kind == "soft")
                _verify_mass(rows, ledger, soft=kind == "soft")
                if kind == "soft" and not math.isclose(
                    math.fsum(rows.values()), report["soft_expected_count"],
                    rel_tol=1e-9, abs_tol=1e-6,
                ):
                    raise ValueError("soft counts disagree with reconstruction report")
                name = f"{kind}_{split}"
                daily_count = []
                blocked_count = []
                for (raw_hour, route, stop, direction, applicable), count in sorted(rows.items()):
                    instant = _hour(raw_hour, day)
                    eligible = applicable
                    extra: list[Any] = []
                    if policy is not None:
                        if route not in patterns or "warnings" not in patterns[route]:
                            raise ValueError("policy requires receipt-bound route pattern warnings")
                        key = route, raw_hour, applicable, kind
                        if key not in decisions:
                            decisions[key] = policy.evaluate(
                                route, instant, engine_applicable=applicable,
                                warnings=tuple(patterns[route]["warnings"]), soft=kind == "soft",
                            )
                        decision = decisions[key]
                        eligible = decision.eligible
                        extra = [applicable, json.dumps(decision.reasons, separators=(",", ":"))]
                        if not eligible:
                            stats[name]["policy_blocked_count"] += count
                            if applicable:
                                stats[name]["newly_policy_blocked_count"] += count
                        elif not applicable:
                            stats[name]["restored_engine_count"] += count
                    quality_reasons = []
                    if route == "7" and day == "2025-07-29":
                        eligible = False
                        quality_reasons.append(MIXED_PRECISION_REASON)
                        stats[name]["algorithmic_affected_count"] += count
                    writers[name].writerow([
                        day, route, instant.isoformat(), stop, direction,
                        format(count, ".17g") if kind == "soft" else str(int(count)),
                        applicable, instant.hour, instant.weekday(), LABEL_ORIGIN, eligible,
                        json.dumps(quality_reasons, separators=(",", ":")),
                    ] + extra)
                    stats[name]["rows"] += 1
                    daily_count.append(count)
                    if not applicable:
                        blocked_count.append(count)
                stats[name]["count"] += math.fsum(daily_count)
                stats[name]["blocked_pattern_count"] += math.fsum(blocked_count)
    _checked_hash(manifest_path, source_hash)
    if policy is not None and service_policy is not None:
        _checked_hash(service_policy, policy.registry_sha256)
    for path in paths.values():
        path.with_suffix(".gz.tmp").replace(path)
    result = {
        "schema_version": "boarding-real-training.v1", "complete": True,
        "feature_version": "calendar.v1", "source_manifest_sha256": source_hash,
        "timezone": "Europe/Moscow", "time_basis": "original_payment_time",
        "target": "validation_count", "unit": "event_count",
        "label_origin": LABEL_ORIGIN, "real_stop_accuracy": "unverified",
        "observed_stop_labels": False, "point_in_time_sources_verified": False,
        "calibrated": False, "mode": "retrospective_experimental_soft_labels",
        "training_eligibility": (
            "applicable pattern after algorithmic guards; experimental targets, not ground truth"
        ),
        "strict_coverage": "partial accepted-event counts; absent cells are missing, not zero",
        "soft_coverage": "decoded events only; blocked patterns retained with eligibility false",
        "feature_availability": "civil calendar only; no stop lags or future features",
        "diagnostic_is_blind": False,
        "split_unit": "whole civil day",
        "split_boundaries": {"train_end_exclusive": "2025-07-01",
                             "development_end_exclusive": "2025-09-01"},
        "dates": [day for day, _ in days], "date_count": len(days),
        "routes": source["config"]["routes"],
        "source_success": source["source_success"],
        "source_rejected": sum(r["source_rejected"] for _, r in days),
        "source_decoded": source["source_success"] - source["unassigned"],
        "source_unassigned": source["unassigned"],
        "source_strict_accepted": source["assigned_weak"],
        "source_outside_selected_dates_or_routes": None,
        "out_of_scope_count_available": False,
        "scope": "exact reconstruction dates and routes; excluded source mass is unknown",
        "algorithmic_exclusions": [{
            "reason": MIXED_PRECISION_REASON,
            "source_schema_version": "boarding-real-run.v1",
            "source_composition_sha256": source["implementation"]["composition.py"],
            "route": "7", "day": "2025-07-29",
            "affected_soft_mass": sum(
                stats["soft_" + split]["algorithmic_affected_count"] for split in SPLITS
            ),
            "affected_strict_mass": sum(
                stats["weak_" + split]["algorithmic_affected_count"] for split in SPLITS
            ),
            "basis": "unequal exact/proxy clock precision changes state support for equal clocks",
            "action": "training eligibility false; counts retained without renormalization",
            "official_service_notice": False,
        }],
        "files": {name: {"name": path.name, "sha256": digest(path), **stats[name]}
                  for name, path in paths.items()},
    }
    if policy is not None:
        result["soft_coverage"] = (
            "decoded events only; engine pattern flags retained; "
            "explicit policy controls eligibility"
        )
        result["training_eligibility"] = (
            "experimental targets after service and algorithmic guards; not ground truth"
        )
        result["service_policy"] = {
            "registry_sha256": policy.registry_sha256,
            "source_urls": policy.source_urls,
            "policy_version": "whole-hour-notice-exclusions.v1",
            "complete_historical_calendar": False,
            "unknown_start_precautions": "not evidence of actual historical closures",
            "raw_engine_blocked_soft_mass": sum(
                stats["soft_" + split]["blocked_pattern_count"] for split in SPLITS
            ),
            "policy_blocked_soft_mass": sum(
                stats["soft_" + split]["policy_blocked_count"] for split in SPLITS
            ),
            "policy_blocked_strict_mass": sum(
                stats["weak_" + split]["policy_blocked_count"] for split in SPLITS
            ),
            "restored_soft_mass": sum(
                stats["soft_" + split]["restored_engine_count"] for split in SPLITS
            ),
            "mass_policy": "retain every row and count; do not renormalize or create strict labels",
        }
    write_json(out / "manifest.json", result)
    return result

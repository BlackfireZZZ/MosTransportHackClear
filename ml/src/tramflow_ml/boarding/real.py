"""Real AFC inference: candidate stops and conservative weak-label eligibility."""

import gzip
import hashlib
import io
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, write_json
from .composition import RouteEvidence, as_of, compose_route, timetable_emissions
from .historical import load_historical_catalog
from .partitions import read_day, write_frame
from .sequence import SequenceConfig, infer_sequence

ROUTES = ("1", "7", "11", "12", "17", "25", "26", "28", "50")


@dataclass(frozen=True)
class RealConfig:
    dates: tuple[str, ...]
    routes: tuple[str, ...] = ROUTES
    gap_seconds: float = 30
    max_span_seconds: float = 90
    context_seconds: int = 0
    min_posterior: float = 0.5
    min_scenario_agreement: float = 0.75
    max_stationary_seconds: float = 300
    seed: int = 20260926

    def __post_init__(self) -> None:
        if not self.dates or len(self.dates) != len(set(self.dates)):
            raise ValueError("unique nonempty frozen dates required")
        for day in self.dates:
            date.fromisoformat(day)
        if not self.routes or len(self.routes) != len(set(self.routes)):
            raise ValueError("unique nonempty routes required")
        if not (
            0 < self.gap_seconds <= self.max_span_seconds
            and self.context_seconds == 0
            and 0 < self.min_posterior <= 1
            and 0 < self.min_scenario_agreement <= 1
            and self.max_stationary_seconds > 0
        ):
            raise ValueError("invalid inference configuration")
        if not all(
            math.isfinite(x)
            for x in [
                self.gap_seconds,
                self.max_span_seconds,
                self.min_posterior,
                self.min_scenario_agreement,
                self.max_stationary_seconds,
            ]
        ):
            raise ValueError("finite inference parameters required")


def burst_groups(times: np.ndarray[Any, Any], gap: float, span: float) -> np.ndarray[Any, Any]:
    groups = np.zeros(len(times), dtype=np.int64)
    start = 0
    current = 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > gap or times[i] - times[start] > span:
            current += 1
            start = i
        groups[i] = current
    return groups


def session_keys(frame: Any) -> Any:
    """Within-source vehicle pooling is inferred, and conflicts fall back to device groups."""
    frame = frame.copy()
    frame["hour"] = frame.event_at.str[:13]
    complete = frame.device_key.ne("") & frame.vehicle_key.ne("")
    links = frame.loc[complete].groupby(["hour", "device_key"]).vehicle_key.nunique()
    conflict = {(h, d) for (h, d), n in links.items() if n > 1}
    frame["identity_conflict"] = [
        (h, d) in conflict for h, d in zip(frame.hour, frame.device_key, strict=True)
    ]
    vehicle_conflicts = frame.loc[frame.vehicle_key.ne("")].copy()
    vehicle_conflicts["route_exit"] = vehicle_conflicts.route + ":" + vehicle_conflicts.exit_key
    by_vehicle = vehicle_conflicts.groupby(["hour", "vehicle_key"]).route_exit.nunique()
    conflicting_vehicles = {(h, v) for (h, v), n in by_vehicle.items() if n > 1}
    frame["identity_conflict"] = frame.identity_conflict | np.array(
        [(h, v) in conflicting_vehicles for h, v in zip(frame.hour, frame.vehicle_key, strict=True)]
    )
    frame["identity_kind"] = np.where(
        frame.vehicle_key.ne("") & ~frame.identity_conflict, "inferred_vehicle_pool", "device_only"
    )
    frame["group"] = np.where(
        frame.identity_kind.eq("inferred_vehicle_pool"),
        "v:" + frame.vehicle_key,
        "d:" + frame.device_key + ":" + frame.vehicle_key,
    )
    missing = frame.device_key.eq("") & frame.vehicle_key.eq("")
    frame.loc[missing, "group"] = "unknown:" + frame.loc[missing, "event_key"]
    frame.loc[missing, "identity_kind"] = "identity_missing"
    frame["group"] = frame.route + ":" + frame.exit_key + ":" + frame.group
    return frame


def _session(
    frame: Any, evidence: RouteEvidence, config: RealConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = frame.sort_values(["second", "event_key"]).reset_index(drop=True)
    times = frame.second.to_numpy(dtype=np.float64)
    groups = burst_groups(times, config.gap_seconds, config.max_span_seconds)
    n = int(groups[-1]) + 1
    bt = np.bincount(groups, weights=times) / np.bincount(groups)
    emission = timetable_emissions(bt, evidence)
    no_schedule = timetable_emissions(bt, evidence, use_schedule=False)
    deltas = np.diff(bt)
    bounded = deltas[deltas <= 1800]
    steps = min(128, max(12, int(math.ceil(float(bounded.max()) / 30)) + 3 if len(bounded) else 12))
    base = SequenceConfig(max_steps=steps)
    scenarios = [
        (base, emission),
        (replace(base, speed_mps=4), emission),
        (replace(base, speed_mps=6), emission),
        (base, no_schedule),
    ]
    outputs = [infer_sequence(bt, evidence.template, c, log_emissions=e) for c, e in scenarios]
    primary = outputs[0]
    path = primary.path_states
    posterior = primary.posterior[np.arange(n), path]
    agreement = np.mean(np.array([o.path_states == path for o in outputs]), axis=0)
    unique = (primary.top_probabilities[:, 0] - primary.top_probabilities[:, 1]) > 1e-9
    stationary = np.zeros(n, dtype=np.bool_)
    run_start = 0
    for i in range(1, n + 1):
        if i == n or path[i] != path[run_start] or primary.path_steps[i] != 0:
            if bt[i - 1] - bt[run_start] > config.max_stationary_seconds:
                stationary[run_start:i] = True
            run_start = i
    accepted = (posterior >= config.min_posterior) & (agreement >= config.min_scenario_agreement)
    segment_support = np.zeros(n, dtype=int)
    boundaries = sorted({0, n, *primary.reset_indices})
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        segment_support[start:end] = end - start
    accepted &= unique & ~stationary & evidence.applicable_pattern & (segment_support >= 4)
    if any("step_bound_sensitive" in o.flags for o in outputs):
        accepted[:] = False
    counts = np.bincount(groups)
    session_id = hashlib.sha256(
        (
            str(frame.group.iloc[0]) + ":" + str(frame.loc[frame.core, "event_at"].iloc[0])[:10]
        ).encode()
    ).hexdigest()[:24]
    core = frame.core.to_numpy(dtype=np.bool_)
    core_counts = np.bincount(groups, weights=core.astype(int), minlength=n).astype(int)
    burst_results = []
    latent = 0
    observed = 0
    trip = 0
    for i in range(n):
        step = int(primary.path_steps[i])
        if step < 0:
            trip += 1
        elif step > 0:
            old = int(path[i - 1])
            trip += sum(
                evidence.template.terminal_after[(old + k) % len(evidence.visits)]
                for k in range(step)
            )
            latent += max(0, step - 1)
        observed += int(i == 0 or step != 0)
        state = int(path[i])
        visit = evidence.visits[state]
        order = [state] + [int(v) for v in np.argsort(-primary.posterior[i]) if v != state][:2]
        alternatives = [
            {
                "stop_id": evidence.visits[j]["stop_id"],
                "pattern_id": evidence.visits[j]["pattern_id"],
                "direction": evidence.visits[j]["direction"],
                "sequence": evidence.visits[j]["pattern_sequence"],
                "weight": float(primary.posterior[i, j]),
            }
            for j in order
        ]
        unresolved = max(0.0, 1 - sum(a["weight"] for a in alternatives))
        reason = (
            "stable_inferred"
            if accepted[i]
            else "symmetry_unidentifiable"
            if not unique[i]
            else "known_service_variant_unmodeled"
            if not evidence.applicable_pattern
            else "stationary_path_anomaly"
            if stationary[i]
            else "sparse_or_scenario_sensitive"
        )
        burst_results.append(
            {
                "burst": i,
                "session_id": session_id,
                "journey_hypothesis": trip,
                "burst_time": float(bt[i]),
                "events": int(counts[i]),
                "core_events": int(core_counts[i]),
                "stop_id": visit["stop_id"] if unique[i] else None,
                "stop_name": visit["name"] if unique[i] else None,
                "lat": visit["lat"] if unique[i] else None,
                "lon": visit["lon"] if unique[i] else None,
                "pattern_id": visit["pattern_id"],
                "direction": visit["direction"],
                "sequence": visit["pattern_sequence"],
                "conditional_posterior": float(posterior[i]),
                "scenario_agreement": float(agreement[i]),
                "scenario_state_probabilities": [float(o.posterior[i, state]) for o in outputs],
                "connected_segment_groups": int(segment_support[i]),
                "weak_label_eligible": bool(accepted[i]),
                "reason": reason,
                "schedule_kind": evidence.schedule_kinds[state],
                "confidence_kind": "conditional_model_posterior",
                "calibrated": False,
                "alternatives": alternatives,
                "unresolved_weight": unresolved,
                "path_steps": step,
                "residual_seconds": float(primary.residual_seconds[i]),
                "path_is_physical_truth": False,
            }
        )
    rows = []
    for i, row in enumerate(frame.itertuples(index=False)):
        if not row.core:
            continue
        b = burst_results[int(groups[i])]
        accepted_event = b["weak_label_eligible"] and not row.identity_conflict
        rows.append(
            {
                "event_key": row.event_key,
                "event_at": row.event_at,
                "route": row.route,
                "session_id": session_id,
                "burst": int(groups[i]),
                "journey_hypothesis": b["journey_hypothesis"],
                "stop_id": b["stop_id"],
                "pattern_id": b["pattern_id"],
                "direction": b["direction"],
                "sequence": b["sequence"],
                "conditional_posterior": b["conditional_posterior"],
                "scenario_agreement": b["scenario_agreement"],
                "assignment_status": "inferred_stable" if accepted_event else "ambiguous",
                "weak_label_eligible": accepted_event,
                "identity_kind": row.identity_kind,
                "reason": "identity_conflict" if row.identity_conflict else b["reason"],
                "schedule_kind": b["schedule_kind"],
                "calibrated": False,
            }
        )
    soft_rows = []
    hours = frame.event_at.str[:13].to_numpy()
    for hour in sorted(set(hours[core])):
        burst_mass = np.bincount(groups[core & (hours == hour)], minlength=n)
        state_mass = burst_mass @ primary.posterior
        for state, mass in enumerate(state_mass):
            if mass > 0:
                soft_rows.append(
                    {
                        "route": evidence.template.route,
                        "event_hour": hour,
                        "stop_id": evidence.visits[state]["stop_id"],
                        "direction": evidence.visits[state]["direction"],
                        "expected_count": float(mass),
                        "applicable_pattern": evidence.applicable_pattern,
                    }
                )
    baseline_path = emission.argmax(axis=1)
    baseline_unique = (np.sort(emission, axis=1)[:, -1] - np.sort(emission, axis=1)[:, -2]) > 1e-9
    summary = {
        "session_id": session_id,
        "soft_rows": soft_rows,
        "schedule_only_unique_groups": int(baseline_unique.sum()),
        "schedule_only_agrees_groups": int(((baseline_path == path) & baseline_unique).sum()),
        "route": evidence.template.route,
        "source_success": int(core.sum()),
        "observed_payment_groups": n,
        "candidate_visits_with_support": observed,
        "candidate_latent_visits": latent,
        "candidate_visit_total": observed + latent,
        "independently_confirmed_visits": None,
        "open_start": True,
        "open_end": True,
        "gap_resets": len(primary.reset_indices),
        "stationary_anomaly_groups": int(stationary.sum()),
        "schedule_no_schedule_path_agreement": float(np.mean(outputs[-1].path_states == path)),
        "mean_scenario_agreement": float(agreement.mean()),
        "flags": list(primary.flags),
        "sources": evidence.source_urls,
        "warnings": evidence.warnings,
        "bursts": [b for b in burst_results if b["core_events"] > 0],
    }
    return rows, summary


def reconstruct_day(
    partitions: Path,
    histories: Path,
    graph: Path,
    timetables: Path,
    day: str,
    config: RealConfig,
    out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    d = date.fromisoformat(day)
    core = read_day(partitions, day, config.routes)
    if core.empty:
        raise ValueError("selected day has no source rows")
    events = core.loc[core.success.eq("1")].copy()
    if events.empty:
        raise ValueError("selected day has no successful payments")
    events["second"] = (
        pd.to_datetime(events.event_at)
        .dt.tz_localize("Europe/Moscow")
        .dt.as_unit("ns")
        .astype("int64")
        / 1e9
    )
    events["core"] = True
    events = session_keys(events)
    if events.event_key.duplicated().any():
        raise ValueError("duplicate context event")
    patterns = load_historical_catalog(histories, graph, as_of(d))
    schedules = json.loads(timetables.read_text())
    by_route = {}
    unsupported = {}
    for route in config.routes:
        try:
            by_route[route] = compose_route(patterns, schedules, route, d)
        except ValueError as exc:
            unsupported[route] = str(exc)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    summaries = []
    failures: Counter[str] = Counter()
    for (_, route), group in events.groupby(["group", "route"], sort=True):
        if not bool(group.core.any()):
            continue
        try:
            if route not in by_route:
                raise ValueError("pattern unavailable")
            rows, summary = _session(group, by_route[route], config)
            results.extend(rows)
            summaries.append(summary)
        except ValueError as exc:
            reason = (
                "search_budget_exceeded"
                if "budget" in str(exc)
                else "pattern_or_timing_incompatible"
            )
            failures[reason] += 1
            for row in group.loc[group.core].itertuples(index=False):
                results.append(
                    {
                        "event_key": row.event_key,
                        "event_at": row.event_at,
                        "route": route,
                        "session_id": None,
                        "burst": None,
                        "journey_hypothesis": None,
                        "stop_id": None,
                        "pattern_id": None,
                        "direction": None,
                        "sequence": None,
                        "conditional_posterior": None,
                        "scenario_agreement": None,
                        "assignment_status": "unassigned",
                        "weak_label_eligible": False,
                        "identity_kind": row.identity_kind,
                        "reason": reason,
                        "schedule_kind": "absent",
                        "calibrated": False,
                    }
                )
    output = pd.DataFrame(results).sort_values("event_key")
    expected = core.loc[core.success.eq("1")].groupby(["route", core.event_at.str[:13]]).size()
    actual = output.groupby(["route", output.event_at.str[:13]]).size()
    if not expected.sort_index().equals(actual.sort_index()) or output.event_key.duplicated().any():
        raise ValueError("G0 original route-hour mass mismatch")
    write_frame(out / "event_assignments.csv.gz", output)
    with (
        (out / "burst_candidates.jsonl.gz").open("wb") as raw,
        gzip.GzipFile(fileobj=raw, filename="", mtime=0, mode="wb", compresslevel=1) as gz,
    ):
        with io.TextIOWrapper(gz, encoding="utf-8") as f:
            for summary in summaries:
                for b in summary["bursts"]:
                    if not math.isclose(
                        sum(a["weight"] for a in b["alternatives"]) + b["unresolved_weight"], 1
                    ):
                        raise ValueError("G0 candidate weight mass mismatch")
                    f.write(json.dumps(b, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(
        out / "journey_hypotheses.json",
        [{k: v for k, v in s.items() if k not in {"bursts", "soft_rows"}} for s in summaries],
    )
    write_json(
        out / "patterns.json",
        {
            r: {
                "template": asdict(e.template),
                "visits": e.visits,
                "sources": e.source_urls,
                "warnings": e.warnings,
                "applicable_pattern": e.applicable_pattern,
            }
            for r, e in by_route.items()
        },
    )
    output["event_hour"] = output.event_at.str[:13]
    accepted = output.loc[output.weak_label_eligible]
    grouped = (
        accepted.groupby(["route", "event_hour", "stop_id", "direction"])
        .size()
        .reset_index(name="inferred_count")
    )
    grouped.to_csv(out / "stop_hour_weak_counts.csv", index=False)
    mass = output.groupby(["route", "event_hour", "assignment_status"]).size().unstack(fill_value=0)
    mass.to_csv(out / "mass_ledger.csv")
    soft = pd.DataFrame(
        [row for summary in summaries for row in summary["soft_rows"]],
        columns=[
            "route",
            "event_hour",
            "stop_id",
            "direction",
            "expected_count",
            "applicable_pattern",
        ],
    )
    soft = soft.groupby(
        ["route", "event_hour", "stop_id", "direction", "applicable_pattern"], as_index=False
    ).expected_count.sum()
    soft_mass = soft.groupby(["route", "event_hour"]).expected_count.sum()
    decoded_mass = (
        output.loc[~output.assignment_status.eq("unassigned")]
        .groupby(["route", "event_hour"])
        .size()
    )
    if not np.allclose(
        soft_mass.reindex(decoded_mass.index, fill_value=0), decoded_mass, atol=1e-7
    ):
        raise ValueError("G0 full posterior route-hour mass mismatch")
    write_frame(out / "stop_hour_soft_counts.csv.gz", soft)
    report = {
        "day": day,
        "source_success": len(output),
        "source_rejected": int(core.success.eq("0").sum()),
        "assigned_weak": len(accepted),
        "ambiguous": int(output.assignment_status.eq("ambiguous").sum()),
        "unassigned": int(output.assignment_status.eq("unassigned").sum()),
        "candidate_stop_rows": int(output.stop_id.notna().sum()),
        "soft_expected_count": float(soft.expected_count.sum()),
        "soft_source_hour_conservation": True,
        "schedule_only_unique_groups": sum(s["schedule_only_unique_groups"] for s in summaries),
        "schedule_only_agrees_groups": sum(s["schedule_only_agrees_groups"] for s in summaries),
        "sessions": len(summaries),
        "burst_groups": sum(s["observed_payment_groups"] for s in summaries),
        "candidate_latent_visits": sum(s["candidate_latent_visits"] for s in summaries),
        "source_failures": unsupported,
        "decode_failures": dict(failures),
        "mean_scenario_agreement": float(output.scenario_agreement.mean())
        if output.scenario_agreement.notna().any()
        else None,
        "weak_by_route": accepted.route.value_counts().to_dict(),
        "source_by_route": output.route.value_counts().to_dict(),
        "original_payment_hour_conservation": True,
        "real_stop_accuracy": "unverified",
        "calibrated": False,
        "mode": "retrospective_offline_open_midnight",
        "wall_seconds": time.monotonic() - started,
    }
    write_json(out / "evaluation.json", report)
    return report


def _run_day(
    day: str,
    *,
    partitions: Path,
    histories: Path,
    graph: Path,
    timetables: Path,
    out: Path,
    config: RealConfig,
) -> dict[str, Any]:
    part = out / day
    receipt = part / "receipt.json"
    if receipt.exists():
        metadata = json.loads(receipt.read_text())
        if any(digest(part / name) != sha for name, sha in metadata["files"].items()):
            raise ValueError("completed reconstruction partition changed")
        report = json.loads((part / "evaluation.json").read_text())
    else:
        report = reconstruct_day(partitions, histories, graph, timetables, day, config, part)
        write_json(receipt, {"files": {p.name: digest(p) for p in part.iterdir() if p.is_file()}})
    print(
        json.dumps(
            {
                "day": day,
                "successful": report["source_success"],
                "weak_labels": report["assigned_weak"],
                "seconds": round(report["wall_seconds"], 1),
            }
        ),
        flush=True,
    )
    return dict(report)


def reconstruct(
    partitions: Path,
    histories: Path,
    graph: Path,
    timetables: Path,
    out: Path,
    config: RealConfig,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    if (out / "manifest.json").exists():
        raise ValueError("reconstruction complete; choose new run")
    implementation = {
        p.name: digest(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("composition.py"),
            Path(__file__).with_name("sequence.py"),
            Path(__file__).with_name("historical.py"),
            Path(__file__).with_name("partitions.py"),
        ]
    }
    signature = {
        "schema_version": "boarding-real-run.v1",
        "config": asdict(config),
        "partitions_manifest_sha256": digest(partitions / "manifest.json"),
        "history_manifest_sha256": digest(histories / "history-manifest.json"),
        "graph_sha256": digest(graph),
        "timetables_sha256": digest(timetables),
        "implementation": implementation,
    }
    signature = json.loads(json.dumps(signature))
    if out.exists() and any(out.iterdir()) and not (out / "checkpoint.json").exists():
        raise ValueError("unrecognized partial reconstruction")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint.json").exists() and json.loads(
        (out / "checkpoint.json").read_text()
    ) != signature:
        raise ValueError("reconstruction source/config/code changed")
    write_json(out / "checkpoint.json", signature)
    task = partial(
        _run_day,
        partitions=partitions,
        histories=histories,
        graph=graph,
        timetables=timetables,
        out=out,
        config=config,
    )
    if workers == 1:
        reports = [task(day) for day in sorted(config.dates)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            reports = list(pool.map(task, sorted(config.dates)))
    result = {
        **signature,
        "complete": True,
        "source_success": sum(r["source_success"] for r in reports),
        "assigned_weak": sum(r["assigned_weak"] for r in reports),
        "ambiguous": sum(r["ambiguous"] for r in reports),
        "unassigned": sum(r["unassigned"] for r in reports),
        "candidate_stop_rows": sum(r["candidate_stop_rows"] for r in reports),
        "reports": reports,
        "real_stop_accuracy": "unverified",
        "observed_stop_labels": False,
        "confidence_kind": "conditional_uncalibrated",
        "timezone": "Europe/Moscow",
        "target": "validation_count",
        "time_basis": "original_payment_time",
        "mode": "retrospective_offline",
        "availability": "unknown",
        "seed": config.seed,
        "source_attribution": [
            "OpenStreetMap contributors, ODbL 1.0",
            "transport.mos.ru published schedules",
        ],
        "day_receipts": {d: digest(out / d / "receipt.json") for d in sorted(config.dates)},
    }
    if (
        result["source_success"]
        != result["assigned_weak"] + result["ambiguous"] + result["unassigned"]
    ):
        raise ValueError("G0 final mass mismatch")
    write_json(out / "manifest.json", result)
    return result

"""Reproducible paired v1/v2 audit; inferred graph statistics are not stop accuracy."""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, write_json
from .composition import as_of, compose_route
from .dense_profiles import fit_dense_profiles, summarize_transitions, validate_dense_holdout
from .historical import load_historical_catalog
from .partitions import read_day, write_frame
from .real import ROUTES, session_keys
from .schedule_intervals import estimate_schedule_edges
from .timing_v2 import TimingConfig, edge_durations, reconstruct_session

DEFAULT_DATES = tuple(f"2025-{m:02d}-{d:02d}" for m in range(1, 11) for d in (15, 18))


def compare_flows(before: Any, after: Any) -> dict[str, Any]:
    keys = ["route", "event_hour", "stop_id", "direction"]
    a = before.groupby(keys).expected_count.sum()
    b = after.groupby(keys).expected_count.sum()
    joined = pd.concat([a.rename("v1"), b.rename("v2")], axis=1).fillna(0)
    mass = float(joined.v1.sum())
    if not np.isclose(mass, float(joined.v2.sum()), rtol=0, atol=1e-6):
        raise ValueError("paired soft flow mass mismatch")
    route_hour = joined.groupby(level=[0, 1]).sum()
    if not np.allclose(route_hour.v1, route_hour.v2, rtol=0, atol=1e-6):
        raise ValueError("paired route/hour mass mismatch")
    moved = float(abs(joined.v1 - joined.v2).sum() / 2)
    stop = joined.groupby(level=[0, 2, 3]).sum()
    correlation = float(stop.v1.corr(stop.v2)) if len(stop) > 1 else float("nan")
    return {
        "source_events": mass,
        "mass_moved_between_stop_direction_hours": moved,
        "mass_moved_fraction": moved / mass if mass else None,
        "stop_total_pearson": correlation if np.isfinite(correlation) else None,
        "v1_stop_hour_cells": len(a),
        "v2_stop_hour_cells": len(b),
        "route_hour_mass_conserved": True,
    }


def compare_events(before: Any, after: Any) -> dict[str, Any]:
    if before.event_key.duplicated().any() or after.event_key.duplicated().any():
        raise ValueError("duplicate event identity")
    merged = before.merge(after, on="event_key", suffixes=("_v1", "_v2"), validate="one_to_one")
    if len(merged) != len(before) or len(merged) != len(after):
        raise ValueError("paired comparison lost original events")
    if not (
        merged.route_v1.eq(merged.route_v2).all()
        and merged.event_at_v1.eq(merged.event_at_v2).all()
    ):
        raise ValueError("comparison changed source route or event time")
    comparable = merged.stop_id_v1.notna() & merged.stop_id_v2.notna()
    same = (
        (merged.stop_id_v1 == merged.stop_id_v2)
        & (merged.direction_v1 == merged.direction_v2)
        & comparable
    )
    return {
        "events": len(merged),
        "candidate_to_null": int((merged.stop_id_v1.notna() & merged.stop_id_v2.isna()).sum()),
        "null_to_candidate": int((merged.stop_id_v1.isna() & merged.stop_id_v2.notna()).sum()),
        "both_null": int((merged.stop_id_v1.isna() & merged.stop_id_v2.isna()).sum()),
        "both_have_candidate": int(comparable.sum()),
        "same_stop_direction": int(same.sum()),
        "agreement_on_comparable": float(same.sum() / comparable.sum())
        if comparable.any()
        else None,
        "v2_raw_stable": int(after.raw_stable.sum()),
        "v2_training_eligible": 0,
    }


def _day(day: str, source: Path, out: Path) -> dict[str, Any]:
    dest = out / day
    receipt = dest / "receipt.json"
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        if any(digest(dest / name) != sha for name, sha in existing.items()):
            raise ValueError("changed completed day")
        return dict(json.loads((dest / "summary.json").read_text()))
    frame = read_day(source / "boarding-date-shards", day, ROUTES)
    frame = frame.loc[frame.success.eq("1")].copy()
    frame["second"] = (
        pd.to_datetime(frame.event_at)
        .dt.tz_localize("Europe/Moscow")
        .dt.as_unit("ns")
        .astype("int64")
        / 1e9
    )
    frame["core"] = True
    frame = session_keys(frame)
    graph = source.parent.parent / "data/tram_graph.json"
    patterns = load_historical_catalog(
        source / "real-sources/osm", graph, as_of(date.fromisoformat(day))
    )
    schedules = json.loads((source / "real-sources/timetables/timetables.json").read_text())
    evidence = {r: compose_route(patterns, schedules, r, date.fromisoformat(day)) for r in ROUTES}
    estimates = {r: estimate_schedule_edges(e) for r, e in evidence.items()}
    durations = {r: edge_durations(e, estimates[r]) for r, e in evidence.items()}
    rows, soft, transitions, anchors = [], [], [], []
    session_reports = []
    failures: Counter[str] = Counter()
    for (_, route), block in frame.groupby(["group", "route"], sort=True):
        try:
            result = reconstruct_session(block, evidence[route], durations[route][0])
            rows.extend(result["events"])
            soft.extend(result["soft"])
            transitions.extend(result["transitions"])
            anchors.extend(result["anchors"])
            session_reports.append(
                {
                    k: v
                    for k, v in result.items()
                    if k not in {"events", "soft", "transitions", "anchors"}
                }
            )
        except ValueError as exc:
            failures[str(exc)] += len(block)
    if failures:
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / "failures.json", dict(failures))
        raise ValueError(f"incomplete day {day}: {dict(failures)}; no complete result published")
    assignments = pd.DataFrame(rows)
    assignments["stop_id"] = assignments.stop_id.astype("string")
    assignment_keys = set(assignments.event_key)
    if assignment_keys != set(frame.event_key) or len(assignments) != len(frame):
        raise ValueError("source event conservation failed")
    counts = (
        pd.DataFrame(soft)
        .groupby(["route", "event_hour", "stop_id", "direction"], as_index=False)
        .expected_count.sum()
    )
    prior = source / "real-payment-stops-2025-v1" / day
    baseline = pd.read_csv(
        prior / "event_assignments.csv.gz",
        dtype={"stop_id": "string", "direction": "string", "route": str},
    )
    baseline_soft = pd.read_csv(
        prior / "stop_hour_soft_counts.csv.gz",
        dtype={"stop_id": str, "direction": str, "route": str},
    )
    report = {
        "date": day,
        "events": compare_events(baseline, assignments),
        "flow": compare_flows(baseline_soft, counts),
        "routes": {
            r: {
                "events": compare_events(
                    baseline.loc[baseline.route.eq(r)], assignments.loc[assignments.route.eq(r)]
                ),
                "flow": compare_flows(
                    baseline_soft.loc[baseline_soft.route.eq(r)], counts.loc[counts.route.eq(r)]
                ),
            }
            for r in ROUTES
        },
        "sessions": len(session_reports),
        "bursts": sum(s["bursts"] for s in session_reports),
        "resets": sum(s["resets"] for s in session_reports),
        "schedule_edges": {
            r: {
                "accepted": sum(e["accepted"] for e in estimates[r]["edges"]),
                "total": len(estimates[r]["edges"]),
            }
            for r in ROUTES
        },
        "baseline_files": {
            p.name: digest(p)
            for p in [prior / "event_assignments.csv.gz", prior / "stop_hour_soft_counts.csv.gz"]
        },
    }
    write_frame(dest / "event_assignments.csv.gz", assignments.sort_values("event_key"))
    write_frame(dest / "stop_hour_soft_counts.csv.gz", counts)
    write_frame(
        dest / "transitions.csv.gz",
        pd.DataFrame(transitions).assign(
            skipped_stops=[json.dumps(t["skipped_stops"]) for t in transitions]
        ),
    )
    write_json(dest / "anchors.json", anchors)
    write_json(dest / "schedule_edges.json", estimates)
    write_json(dest / "summary.json", report)
    write_json(receipt, {p.name: digest(p) for p in dest.iterdir() if p.is_file()})
    print(
        json.dumps(
            {
                "completed": day,
                "events": len(assignments),
                "agreement": report["events"]["agreement_on_comparable"],
            }
        ),
        flush=True,
    )
    return report


def load_transitions(out: Path, dates: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for day in dates:
        frame = pd.read_csv(
            out / day / "transitions.csv.gz",
            dtype={
                "route": str,
                "from_stop": str,
                "to_stop": str,
                "direction": str,
                "vehicle_key": str,
            },
            keep_default_na=False,
        )
        frame["skipped_stops"] = frame.skipped_stops.map(json.loads)
        rows.extend(frame.to_dict("records"))
    return rows


def run(source: Path, out: Path, dates: tuple[str, ...], workers: int = 4) -> dict[str, Any]:
    if not dates or len(set(dates)) != len(dates) or not 1 <= workers <= 6:
        raise ValueError("unique dates and 1..6 workers required")
    code = Path(__file__).parent
    signature = {
        "schema_version": "boarding-timing-experiment.v2",
        "dates": list(dates),
        "timezone": "Europe/Moscow",
        "target": "conditional_validation_count",
        "config": asdict(TimingConfig()),
        "code": {p.name: digest(p) for p in code.glob("*.py")},
        "source_partitions": digest(source / "boarding-date-shards/manifest.json"),
        "schedule_sha256": digest(source / "real-sources/timetables/timetables.json"),
        "history_sha256": digest(source / "real-sources/osm/history-manifest.json"),
        "graph_sha256": digest(source.parent.parent / "data/tram_graph.json"),
        "baseline_manifest_sha256": digest(source / "real-payment-stops-2025-v1/manifest.json"),
        "profile_cutoff": "2025-07-01",
        "real_stop_accuracy": "unverified",
        "training_promotion": False,
    }
    checkpoint = out / "checkpoint.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text()) != signature:
        raise ValueError("changed experiment inputs/code; use a new output directory")
    if (out / "manifest.json").exists():
        raise ValueError("experiment complete; choose new output")
    if out.exists() and any(out.iterdir()) and not checkpoint.exists():
        raise ValueError("unrecognized partial output")
    out.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint, signature)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reports = list(pool.map(partial(_day, source=source, out=out), dates))
    transitions = load_transitions(out, dates)
    profiles = fit_dense_profiles(transitions, "2025-07-01")
    write_json(out / "dense_profiles.json", profiles)
    write_json(out / "transition_statistics.json", summarize_transitions(transitions))
    write_json(out / "dense_holdout.json", validate_dense_holdout(transitions, profiles))
    totals = {
        "events": sum(r["events"]["events"] for r in reports),
        "same_stop_direction": sum(r["events"]["same_stop_direction"] for r in reports),
        "both_have_candidate": sum(r["events"]["both_have_candidate"] for r in reports),
        "mass_moved": sum(r["flow"]["mass_moved_between_stop_direction_hours"] for r in reports),
        "raw_stable": sum(r["events"]["v2_raw_stable"] for r in reports),
    }
    manifest = {
        **signature,
        "totals": totals,
        "days": reports,
        "summary_files": {
            p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "checkpoint.json"
        },
        "complete": True,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(args.source, args.out, tuple(args.dates), args.workers)
    print(json.dumps(result["totals"]))


if __name__ == "__main__":
    main()

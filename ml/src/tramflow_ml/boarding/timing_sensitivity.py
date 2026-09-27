"""Matched perturbations and ablations of v2; agreement is not ground truth."""

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, write_json
from .composition import as_of, compose_route
from .historical import load_historical_catalog
from .partitions import read_day
from .real import ROUTES, session_keys
from .real_stability import perturb, select_sessions
from .schedule_intervals import estimate_schedule_edges
from .timing_experiment import compare_events
from .timing_v2 import TimingConfig, edge_durations, reconstruct_session


def run(source: Path, out: Path, profiles: Path | None = None) -> dict[str, Any]:
    if out.exists():
        raise ValueError("sensitivity output already exists")
    results = []
    dates = ("2025-01-15", "2025-07-29", "2025-10-15")
    profile_data = json.loads(profiles.read_text()) if profiles else None
    for day in dates:
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
        patterns = load_historical_catalog(
            source / "real-sources/osm",
            source.parent.parent / "data/tram_graph.json",
            as_of(date.fromisoformat(day)),
        )
        schedules = json.loads((source / "real-sources/timetables/timetables.json").read_text())
        for route in ROUTES:
            evidence = compose_route(patterns, schedules, route, date.fromisoformat(day))
            edges, _ = edge_durations(evidence, estimate_schedule_edges(evidence))
            geometry, _ = edge_durations(evidence, {"edges": []})
            for rank, block in select_sessions(frame.loc[frame.route.eq(route)]):
                baseline = reconstruct_session(block, evidence, edges)
                before = pd.DataFrame(baseline["events"])
                variants = [
                    "mean_time",
                    "geometry_only",
                    "gap20_span60",
                    "gap40_span120",
                    "thin20pct",
                    "jitter5seconds",
                    "without_schedule",
                ]
                if profile_data is not None and day >= profile_data["cutoff_date"]:
                    variants.append("dense_profiles")
                for variant in variants:
                    config = TimingConfig()
                    selected = block
                    selected_edges = edges
                    if variant == "mean_time":
                        config = replace(config, timestamp="mean")
                    elif variant == "geometry_only":
                        selected_edges = geometry
                    elif variant == "without_schedule":
                        config = replace(config, clock_weight=0)
                    elif variant == "gap20_span60":
                        config = replace(config, gap_seconds=20, span_seconds=60)
                    elif variant == "gap40_span120":
                        config = replace(config, gap_seconds=40, span_seconds=120)
                    elif variant in {"thin20pct", "jitter5seconds"}:
                        selected = perturb(block, variant, 20260926)
                    changed = reconstruct_session(
                        selected,
                        evidence,
                        selected_edges,
                        config,
                        profiles=profile_data if variant == "dense_profiles" else None,
                    )
                    after = pd.DataFrame(changed["events"])
                    matched = before.loc[before.event_key.isin(after.event_key)]
                    comparison = compare_events(matched, after)
                    results.append(
                        {
                            "date": day,
                            "route": route,
                            "rank": rank,
                            "variant": variant,
                            "source_events": len(before),
                            "profile_edges_applied": changed["profile_edges_applied"],
                            **comparison,
                        }
                    )
            print(json.dumps({"sensitivity_date": day, "route": route}), flush=True)
    totals = {}
    for variant in sorted({r["variant"] for r in results}):
        part = [r for r in results if r["variant"] == variant]
        denominator = sum(r["both_have_candidate"] for r in part)
        same = sum(r["same_stop_direction"] for r in part)
        totals[variant] = {
            "retained_events": sum(r["events"] for r in part),
            "comparable_events": denominator,
            "same_stop_direction": same,
            "agreement": same / denominator if denominator else None,
            "profile_edges_applied": sum(r["profile_edges_applied"] for r in part),
            "sessions": len(part),
        }
    report = {
        "schema_version": "boarding-timing-sensitivity.v2",
        "dates": dates,
        "selection": "per date/route low, median, high event-count vehicle sessions >=20 events",
        "seed": 20260926,
        "results": results,
        "totals": totals,
        "real_accuracy": "unverified",
        "profiles_sha256": digest(profiles) if profiles else None,
        "implementation": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    args = parser.parse_args()
    run(args.source, args.out, args.profiles)


if __name__ == "__main__":
    main()

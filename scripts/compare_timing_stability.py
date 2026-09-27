"""Paired retrospective sensitivity; agreement is repeatability, not stop accuracy."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.composition import as_of, compose_route
from tramflow_ml.boarding.historical import load_historical_catalog
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import RealConfig, _session, session_keys
from tramflow_ml.boarding.real_stability import SCENARIOS, perturb, select_sessions
from tramflow_ml.boarding.schedule_intervals import estimate_schedule_edges
from tramflow_ml.boarding.timing_v2 import TimingConfig, edge_durations, reconstruct_session

SEED = 20260926
SLICES = (("2025-01-15", "1"), ("2025-07-29", "7"))


def check_mass(frame: Any, rows: list[dict[str, Any]], soft: list[dict[str, Any]]) -> dict[str, Any]:
    source = {r.event_key: r for r in frame.itertuples(index=False)}
    if len(source) != len(frame) or len(rows) != len(source):
        raise ValueError("source and output event counts differ")
    emitted = {r["event_key"]: r for r in rows}
    if len(emitted) != len(rows) or emitted.keys() != source.keys():
        raise ValueError("source event identity changed")
    expected = Counter((str(r.route), str(r.event_at)[:13]) for r in source.values())
    actual: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        original = source[row["event_key"]]
        if row["event_at"] != original.event_at or str(row["route"]) != str(original.route):
            raise ValueError("original event hour or route changed")
    for row in soft:
        value = float(row["expected_count"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid posterior mass")
        actual[str(row["route"]), row["event_hour"]] += value
    if actual.keys() != expected.keys() or any(
        not math.isclose(actual[k], count, rel_tol=0, abs_tol=1e-6)
        for k, count in expected.items()
    ):
        raise ValueError("original route-hour posterior mass changed")
    return {
        "source_events": len(frame),
        "source_route_hours": len(expected),
        "soft_mass": sum(actual.values()),
        "max_route_hour_error": max(abs(actual[k] - count) for k, count in expected.items()),
        "mass_conserved": True,
    }


def compare(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    a = {r["event_key"]: r for r in before}
    b = {r["event_key"]: r for r in after}
    if len(a) != len(before) or len(b) != len(after) or not b.keys() <= a.keys():
        raise ValueError("paired comparison requires unique retained original event keys")
    comparable = [k for k in b if a[k]["stop_id"] is not None and b[k]["stop_id"] is not None]
    same_stop = sum(str(a[k]["stop_id"]) == str(b[k]["stop_id"]) for k in comparable)
    same_direction = sum(
        (str(a[k]["stop_id"]), str(a[k]["direction"]))
        == (str(b[k]["stop_id"]), str(b[k]["direction"]))
        for k in comparable
    )
    stable_a = {k for k in b if a[k].get("raw_stable", a[k].get("weak_label_eligible", False))}
    stable_b = {k for k in b if b[k].get("raw_stable", b[k].get("weak_label_eligible", False))}
    return {
        "baseline_events": len(a),
        "retained_events": len(b),
        "dropped_events": len(a) - len(b),
        "comparable_nonnull_events": len(comparable),
        "same_stop": same_stop,
        "same_stop_direction": same_direction,
        "stop_only_agreement": same_stop / len(comparable) if comparable else None,
        "stop_direction_agreement": same_direction / len(comparable) if comparable else None,
        "both_null_events": sum(a[k]["stop_id"] is None and b[k]["stop_id"] is None for k in b),
        "baseline_null_only": sum(a[k]["stop_id"] is None and b[k]["stop_id"] is not None for k in b),
        "variant_null_only": sum(a[k]["stop_id"] is not None and b[k]["stop_id"] is None for k in b),
        "baseline_raw_stable_on_retained": len(stable_a),
        "variant_raw_stable": len(stable_b),
        "raw_stable_retained": len(stable_a & stable_b),
        "raw_stable_retention": len(stable_a & stable_b) / len(stable_a) if stable_a else None,
    }


def implementation() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "ml/src/tramflow_ml/boarding").glob("*.py"))
    files.append(Path(__file__).resolve())
    return {str(p.relative_to(root)): digest(p) for p in files}


def run(source_root: Path, output: Path, seed: int = SEED) -> dict[str, Any]:
    versions = implementation()
    sources = source_root / "ml/artifacts/real-sources"
    partitions = source_root / "ml/artifacts/boarding-date-shards"
    histories = sources / "osm"
    graph = source_root / "data/tram_graph.json"
    timetable_path = sources / "timetables/timetables.json"
    timetables = json.loads(timetable_path.read_text())
    sessions = []
    for day, route in SLICES:
        frame = read_day(partitions, day)
        frame = frame.loc[frame.success.eq("1")].copy()
        frame["second"] = (
            pd.to_datetime(frame.event_at).dt.tz_localize("Europe/Moscow").dt.as_unit("ns").astype("int64")
            / 1e9
        )
        frame["core"] = True
        frame = session_keys(frame)
        patterns = load_historical_catalog(histories, graph, as_of(date.fromisoformat(day)))
        evidence = compose_route(patterns, timetables, route, date.fromisoformat(day))
        estimate = estimate_schedule_edges(evidence)
        durations, _ = edge_durations(evidence, estimate)
        base_v1 = RealConfig(dates=(day,), routes=(route,), seed=seed)
        base_v2 = TimingConfig()
        for rank, selected in select_sessions(frame.loc[frame.route.eq(route)]):
            original = selected.copy(deep=True)
            rows_v1, summary_v1 = _session(selected, evidence, base_v1)
            summary_v2 = reconstruct_session(selected, evidence, durations, base_v2)
            rows_v2 = summary_v2["events"]
            case = {
                "day": day,
                "route": route,
                "count_rank": rank,
                "session_id": summary_v1["session_id"],
                "source_events": len(selected),
                "event_keys_sha256": hashlib.sha256(
                    "\n".join(sorted(selected.event_key)).encode()
                ).hexdigest(),
                "accepted_schedule_edges": estimate["accepted_edge_count"],
                "baseline_v1": check_mass(selected, rows_v1, summary_v1["soft_rows"]),
                "baseline_v2": check_mass(selected, rows_v2, summary_v2["soft"]),
                "baseline_v1_v2": compare(rows_v1, rows_v2),
                "scenarios": [],
            }
            for scenario in SCENARIOS:
                altered = perturb(selected, scenario, seed)
                v1, v2 = base_v1, base_v2
                if scenario == "gap20_span60":
                    v1 = replace(v1, gap_seconds=20, max_span_seconds=60)
                    v2 = replace(v2, gap_seconds=20, span_seconds=60)
                elif scenario == "gap40_span120":
                    v1 = replace(v1, gap_seconds=40, max_span_seconds=120)
                    v2 = replace(v2, gap_seconds=40, span_seconds=120)
                changed_v1, changed_summary_v1 = _session(altered, evidence, v1)
                changed_summary_v2 = reconstruct_session(altered, evidence, durations, v2)
                changed_v2 = changed_summary_v2["events"]
                case["scenarios"].append(
                    {
                        "scenario": scenario,
                        "v1": {
                            **compare(rows_v1, changed_v1),
                            **check_mass(altered, changed_v1, changed_summary_v1["soft_rows"]),
                            "bursts": changed_summary_v1["observed_payment_groups"],
                        },
                        "v2": {
                            **compare(rows_v2, changed_v2),
                            **check_mass(altered, changed_v2, changed_summary_v2["soft"]),
                            "bursts": changed_summary_v2["bursts"],
                        },
                        "v1_v2": compare(changed_v1, changed_v2),
                    }
                )
            pd.testing.assert_frame_equal(original, selected)
            sessions.append(case)
            print(json.dumps({"day": day, "route": route, "rank": rank, "events": len(selected)}), flush=True)
    aggregates = {}
    for scenario in SCENARIOS:
        engines = {}
        for engine in ("v1", "v2"):
            selected_metrics = [
                variant[engine]
                for case in sessions
                for variant in case["scenarios"]
                if variant["scenario"] == scenario
            ]
            totals = {
                key: sum(m[key] for m in selected_metrics)
                for key in (
                    "baseline_events", "retained_events", "comparable_nonnull_events",
                    "same_stop", "same_stop_direction", "both_null_events", "baseline_null_only",
                    "variant_null_only", "baseline_raw_stable_on_retained", "variant_raw_stable",
                    "raw_stable_retained",
                )
            }
            n, stable = totals["comparable_nonnull_events"], totals["baseline_raw_stable_on_retained"]
            engines[engine] = {
                **totals,
                "stop_only_agreement": totals["same_stop"] / n if n else None,
                "stop_direction_agreement": totals["same_stop_direction"] / n if n else None,
                "raw_stable_retention": totals["raw_stable_retained"] / stable if stable else None,
                "mass_conserved": all(m["mass_conserved"] for m in selected_metrics),
            }
        aggregates[scenario] = engines
    if versions != implementation():
        raise ValueError("implementation changed during paired experiment")
    report = {
        "schema_version": "boarding-timing-paired-sensitivity.v1",
        "seed": seed,
        "timezone": "Europe/Moscow",
        "selection": "same six sessions as real_stability: route1 Jan15 and route7 Jul29 low/median/high",
        "mode": "retrospective_offline",
        "accuracy_measured": False,
        "profiles_used": False,
        "meaning": "within-engine perturbation repeatability, not empirical stop accuracy",
        "perturbed_time_field": "second only; original event_at remains the mass ledger",
        "source_partition_manifest_sha256": digest(partitions / "manifest.json"),
        "history_manifest_sha256": digest(histories / "history-manifest.json"),
        "graph_sha256": digest(graph),
        "timetables_sha256": digest(timetable_path),
        "implementation": versions,
        "sessions": sessions,
        "aggregates": aggregates,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = run(args.source_root, args.output, args.seed)
    print(json.dumps(report["aggregates"], indent=2))


if __name__ == "__main__":
    main()

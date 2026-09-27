"""Audit candidate coalescing and fit intervals for the three displayed real windows."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.composition import as_of, compose_route
from tramflow_ml.boarding.dense_audit import align
from tramflow_ml.boarding.historical import load_historical_catalog
from tramflow_ml.boarding.schedule_intervals import estimate_schedule_edges
from tramflow_ml.boarding.timing_v2 import edge_durations
from tramflow_ml.boarding.wave_merge import adaptive_supports, coalesce


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--visual-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first-anchor", action="store_true")
    a = p.parse_args()
    if a.out.exists():
        raise ValueError("choose a new output directory")
    a.out.mkdir(parents=True)
    manifest = json.loads((a.run / "manifest.json").read_text())
    for name in ["windows.json", "rows.json", "thresholds.json"]:
        assert digest(a.run / name) == manifest["files"][name]
    windows = json.loads((a.run / "windows.json").read_text())
    rows = json.loads((a.run / "rows.json").read_text())
    threshold = json.loads((a.run / "thresholds.json").read_text())["adaptive"]["95"]
    methods = ["consensus3_5s", "adaptive_scan", "feature_gmm"]
    results = []
    visual = json.loads(a.visual_data.read_text())
    patterns = load_historical_catalog(
        a.source / "real-sources/osm",
        a.source.parent.parent / "data/tram_graph.json",
        as_of(date(2025, 5, 15)),
    )
    schedules = json.loads(
        (a.source / "real-sources/timetables/timetables.json").read_text()
    )
    for w, r in zip(windows, rows, strict=True):
        if w["date"] < "2025-05-01":
            continue
        supports = adaptive_supports(w["times"], w["devices"], threshold)
        output = {
            "date": w["date"],
            "route": w["route"],
            "rank": w["rank"],
            "methods": {},
        }
        for method in methods:
            output["methods"][method] = coalesce(
                w["times"], r["methods"][method]["times"], supports,
                first_observation_anchor=a.first_anchor,
            )
        results.append(output)
        if (
            w["date"] != "2025-05-15"
            or w["rank"] != 0
            or w["route"] not in ["17", "12", "11"]
        ):
            continue
        v = next(x for x in visual if x["route"] == w["route"])
        assert v["total"] == len(w["times"])
        evidence = compose_route(patterns, schedules, w["route"], date(2025, 5, 15))
        edges, basis = edge_durations(evidence, estimate_schedule_edges(evidence))
        mask = np.array([stop["boardable"] for stop in evidence.visits])
        v["variants"] = {}
        for variant in ["raw", "merged"]:
            vm = {"methods": {}, "fits": {}}
            for method in methods:
                onsets = (
                    v["methods"][method]
                    if variant == "raw"
                    else output["methods"][method]["times"]
                )
                gaps = np.diff(onsets)
                fit = align(gaps, edges, mask, max_step=6, scales=(1.0,))
                rng = np.random.default_rng(20260926)
                null = [
                    align(
                        gaps, rng.permutation(edges), mask, max_step=6, scales=(1.0,)
                    )["mae_seconds"]
                    for _ in range(9)
                ]
                geometry = [
                    any(
                        basis[(fit["path"][i] + j) % len(edges)]
                        == "geometry_assumption"
                        for j in range(k)
                    )
                    for i, k in enumerate(fit["steps"])
                ]
                vm["methods"][method] = onsets
                vm["fits"][method] = {
                    "expected": fit["predicted_seconds"],
                    "steps": fit["steps"],
                    "geometry": geometry,
                    "mae": fit["mae_seconds"],
                    "phases": fit["competing_phases_within_10s"],
                    "null_mae": float(np.median(null)),
                }
            v["variants"][variant] = vm
        v["merge"] = output["methods"]
    summary = {"test_windows": len(results), "methods": {}}
    for method in methods:
        values = [r["methods"][method] for r in results]
        summary["methods"][method] = {
            "before": sum(v["input_candidates"] for v in values),
            "after": sum(len(v["times"]) for v in values),
            "removed": sum(v["merged_candidates"] for v in values),
            "anchors_added": sum(v["anchor"]["added"] for v in values),
            "quiet_gap_vetoes": sum(
                d["decision"] == "keep_quiet_gap"
                for v in values
                for d in v["decisions"]
            ),
        }
    write_json(a.out / "summary.json", summary)
    write_json(a.out / "rows.json", results)
    write_json(a.out / "visual-data.json", visual)
    write_json(
        a.out / "manifest.json",
        {
            "schema_version": "boarding-wave-merge.v2",
            "first_observation_anchor": a.first_anchor,
            "timezone": "Europe/Moscow",
            "date_range": ["2025-05-15", "2025-10-18"],
            "source_manifest_sha256": digest(a.run / "manifest.json"),
            "quiet_score_is_calibrated_probability": False,
            "stop_accuracy_verified": False,
            "script_sha256": digest(Path(__file__)),
            "module_sha256": digest(
                Path(__file__).parents[1] / "ml/src/tramflow_ml/boarding/wave_merge.py"
            ),
            "files": {p.name: digest(p) for p in a.out.iterdir() if p.is_file()},
        },
    )
    print(json.dumps(summary))
    for v in visual:
        print(
            v["route"],
            {
                m: [len(v["variants"][k]["methods"][m]) for k in ["raw", "merged"]]
                for m in methods
            },
        )


if __name__ == "__main__":
    main()

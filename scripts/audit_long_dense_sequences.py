"""Longer onset sequences test phase transfer on independently selected busy vehicles."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tramflow_ml.boarding.audit import write_json
from tramflow_ml.boarding.burst_detection import detect
from tramflow_ml.boarding.composition import as_of, compose_route
from tramflow_ml.boarding.dense_audit import align, prefix_prediction
from tramflow_ml.boarding.historical import load_historical_catalog
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import session_keys
from tramflow_ml.boarding.schedule_intervals import estimate_schedule_edges
from tramflow_ml.boarding.timing_v2 import edge_durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--short-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for day in ["2025-01-15", "2025-06-18", "2025-10-15"]:
        chosen = json.loads((args.short_run / (day + ".json")).read_text())["rows"]
        keys = {r["session_key"] for r in chosen if r["route"] in ["17", "12", "11"]}
        frame = read_day(args.source / "boarding-date-shards", day, ("17", "12", "11"))
        frame = frame.loc[frame.success.eq("1")].copy()
        frame["second"] = (
            pd.to_datetime(frame.event_at)
            .dt.tz_localize("Europe/Moscow")
            .dt.as_unit("ns")
            .astype("int64")
            / 1e9
        )
        frame = session_keys(frame)
        patterns = load_historical_catalog(
            args.source / "real-sources/osm",
            args.source.parent.parent / "data/tram_graph.json",
            as_of(date.fromisoformat(day)),
        )
        schedules = json.loads(
            (args.source / "real-sources/timetables/timetables.json").read_text()
        )
        for (key, route), block in frame.groupby(["group", "route"], sort=True):
            if key not in keys:
                continue
            evidence = compose_route(
                patterns, schedules, route, date.fromisoformat(day)
            )
            edges, _ = edge_durations(evidence, estimate_schedule_edges(evidence))
            mask = np.array([v["boardable"] for v in evidence.visits])
            times = np.sort(block.second.to_numpy(dtype=float))
            for method in ["gap30", "rate_dbscan", "decay_onset"]:
                detected = detect(times, method)
                candidates = []
                for i in range(len(detected["times"]) - 39):
                    count = detected["counts"][i : i + 40]
                    gaps = np.diff(detected["times"][i : i + 40])
                    if (
                        np.count_nonzero(count >= 3) >= 27
                        and max(gaps) <= 600
                        and min(gaps) >= 15
                        and max(detected["durations"][i : i + 40]) <= 180
                    ):
                        candidates.append((int(count.sum()), -i))
                row = {
                    "date": day,
                    "route": route,
                    "session_key": key,
                    "method": method,
                    "full_session_events": len(times),
                    "onsets": len(detected["times"]),
                    "eligible": bool(candidates),
                }
                if candidates:
                    start = -max(candidates)[1]
                    selected = detected["times"][start : start + 40]
                    gaps = np.diff(selected)
                    fit = align(gaps, edges, mask)
                    prefix = prefix_prediction(gaps, edges, mask, prefix=12)
                    rng = np.random.default_rng(20260926)
                    null = [
                        align(gaps, rng.permutation(edges), mask)["mae_seconds"]
                        for _ in range(9)
                    ]
                    keep = np.unique(np.r_[0, np.arange(1, 39, 2), 39])
                    thinned = align(np.diff(selected[keep]), edges, mask, max_step=6)
                    row.update(
                        fit=fit,
                        prefix=prefix,
                        shuffled_median=float(np.median(null)),
                        beats_all_9_shuffles=bool(
                            fit["mae_seconds"] < min(null) - 1e-9
                        ),
                        retained_position_agreement=float(
                            np.mean(np.array(fit["path"])[keep] == thinned["path"])
                        ),
                        observed_gaps_seconds=gaps.tolist(),
                        burst_counts=detected["counts"][start : start + 40].tolist(),
                    )
                rows.append(row)
        print(day, flush=True)
    write_json(
        args.out,
        {
            "selection": "same top3 busiest30min selected vehicle sessions, full-day sequence, top3 routes,3dates",
            "observations_per_window": 40,
            "rows": rows,
            "accuracy": "unverified",
        },
    )


if __name__ == "__main__":
    main()

"""Observed-density selection and controlled burst/route fingerprint experiments."""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, write_json
from .burst_detection import METHODS, detect, heldout_shape, match_onsets, randomize_local_rate
from .composition import as_of, compose_route
from .dense_audit import dense_window, diagnose
from .historical import load_historical_catalog
from .partitions import read_day
from .real import ROUTES, session_keys
from .schedule_intervals import estimate_schedule_edges
from .timing_experiment import DEFAULT_DATES
from .timing_v2 import edge_durations


def busiest(times: Any) -> Any:
    bins = np.floor(times / 300).astype(int)
    start = int(bins.min())
    counts = np.bincount(bins - start)
    padded = np.pad(counts, (0, 5))
    totals = np.convolve(padded, np.ones(6, dtype=int), mode="valid")
    index = int(np.argmax(totals))
    begin = (start + index) * 300
    return times[(times >= begin) & (times < begin + 1800)]


def _day(day: str, source: Path, out: Path) -> dict[str, Any]:
    frame = read_day(source / "boarding-date-shards", day, ROUTES)
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
        source / "real-sources/osm",
        source.parent.parent / "data/tram_graph.json",
        as_of(date.fromisoformat(day)),
    )
    schedules = json.loads((source / "real-sources/timetables/timetables.json").read_text())
    rows, selection = [], []
    for route in ROUTES:
        evidence = compose_route(patterns, schedules, route, date.fromisoformat(day))
        edges, basis = edge_durations(evidence, estimate_schedule_edges(evidence))
        mask = np.array([v["boardable"] for v in evidence.visits])
        candidates = []
        route_frame = frame.loc[frame.route.eq(route)]
        excluded = 0
        for key, group in route_frame.groupby("group", sort=True):
            if (
                group.identity_conflict.any()
                or not group.identity_kind.eq("inferred_vehicle_pool").all()
            ):
                excluded += len(group)
                continue
            times = np.sort(group.second.to_numpy(dtype=float))
            roi = busiest(times)
            if len(roi) >= 60:
                candidates.append((len(roi), str(key), roi))
        candidates.sort(key=lambda v: (-v[0], v[1]))
        selected = candidates[:3]
        selection.append(
            {
                "date": day,
                "route": route,
                "source_events": len(route_frame),
                "excluded_identity_events": excluded,
                "candidate_sessions": len(candidates),
                "selected_sessions": len(selected),
                "selected_events": sum(c[0] for c in selected),
            }
        )
        for rank, (_, key, times) in enumerate(selected):
            seed = 20260926 + int(route) * 100 + rank
            null_times = randomize_local_rate(times, seed)
            selected_indices = np.random.default_rng(seed).choice(
                len(times), size=len(times) * 4 // 5, replace=False
            )
            selected_mask = np.zeros(len(times), dtype=bool)
            selected_mask[selected_indices] = True
            thinned = times[selected_mask]
            heldout = times[~selected_mask]
            for method in METHODS:
                detected = detect(times, method)
                null = detect(null_times, method)
                thin = detect(thinned, method)
                row: dict[str, Any] = {
                    "date": day,
                    "route": route,
                    "rank": rank,
                    "session_key": key,
                    "method": method,
                    "source_events": len(times),
                    "events_per_minute": len(times) / 30,
                    "bursts": len(detected["times"]),
                    "null_bursts": len(null["times"]),
                    "covered_events": detected["covered_events"],
                    "forced_boundaries": int(detected["forced_boundaries"].sum()),
                    "onset_thinning": match_onsets(detected["times"], thin["times"]),
                    "heldout_shape": heldout_shape(
                        thin["times"], heldout, float(times[0]), float(times[-1])
                    ),
                    "applicable_pattern": evidence.applicable_pattern,
                    "schedule_edges": basis.count("inferred_schedule_interval"),
                    "total_edges": len(edges),
                }
                start = dense_window(detected)
                row["has_dense_window"] = start is not None
                if start is not None:
                    selected_times = detected["times"][start : start + 12]
                    result = diagnose(selected_times, edges, mask, seed)
                    row["analysis"] = result
                    row["observed_gaps_seconds"] = np.diff(selected_times).tolist()
                    row["burst_counts"] = detected["counts"][start : start + 12].tolist()
                    row["burst_times"] = selected_times.tolist()
                    row["stop_names"] = [evidence.visits[s]["name"] for s in result["fit"]["path"]]
                    row["stop_ids"] = [
                        str(evidence.visits[s]["stop_id"]) for s in result["fit"]["path"]
                    ]
                    row["directions"] = [
                        evidence.visits[s]["direction"] for s in result["fit"]["path"]
                    ]
                    row["strong_candidate"] = (
                        result["strong_candidate"] and evidence.applicable_pattern
                    )
                rows.append(row)
        print(json.dumps({"date": day, "route": route, "sessions": len(selected)}), flush=True)
    result = {"date": day, "selection": selection, "rows": rows}
    write_json(out / (day + ".json"), result)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def section(part: list[dict[str, Any]]) -> dict[str, Any]:
        matched = [r for r in part if r["has_dense_window"]]

        def average(key: str) -> float | None:
            return float(np.mean([r["analysis"][key] for r in matched])) if matched else None

        fit = [r["analysis"]["fit"] for r in matched]
        total = sum(r["bursts"] for r in part)
        return {
            "segments": len(part),
            "source_events": sum(r["source_events"] for r in part),
            "bursts": total,
            "null_bursts": sum(r["null_bursts"] for r in part),
            "forced_boundaries": sum(r["forced_boundaries"] for r in part),
            "covered_events": sum(r["covered_events"] for r in part),
            "onset_thinning_matched": sum(r["onset_thinning"]["matched"] for r in part),
            "onset_thinning_reference": sum(r["onset_thinning"]["reference"] for r in part),
            "heldout_shape_counts": {
                key: sum(r["heldout_shape"][key] for r in part)
                for key in ["onsets", "before_events", "early_events", "late_events"]
            },
            "dense_windows": len(matched),
            "median_fit_mae_seconds": float(np.median([f["mae_seconds"] for f in fit]))
            if fit
            else None,
            "mean_within_30": float(np.mean([f["within_30"] for f in fit])) if fit else None,
            "median_competing_phases": float(
                np.median([f["competing_phases_within_10s"] for f in fit])
            )
            if fit
            else None,
            "beats_all_9_shuffles": sum(r["analysis"]["beats_all_9_shuffles"] for r in matched),
            "beats_shuffle_by_20pct": sum(
                r["analysis"]["beats_shuffled_median_by_20pct"] for r in matched
            ),
            "median_consecutive_mae_seconds": float(
                np.median([r["analysis"]["consecutive"]["mae_seconds"] for r in matched])
            )
            if matched
            else None,
            "median_prefix_holdout_mae_seconds": float(
                np.median([r["analysis"]["prefix"]["holdout_mae_seconds"] for r in matched])
            )
            if matched
            else None,
            "median_shuffled_prefix_holdout_mae_seconds": float(
                np.median([r["analysis"]["shuffled_prefix_holdout_median"] for r in matched])
            )
            if matched
            else None,
            "retained_position_agreement": average("retained_positions_after_masking"),
            "strong_candidates": sum(r.get("strong_candidate", False) for r in part),
        }

    return {
        "overall": {m: section([r for r in rows if r["method"] == m]) for m in METHODS},
        "routes": {
            r: {
                m: section([x for x in rows if x["route"] == r and x["method"] == m])
                for m in METHODS
            }
            for r in ROUTES
        },
        "temporal": {
            period: {
                m: section(
                    [r for r in rows if r["method"] == m and (r["date"] < "2025-07-01") == early]
                )
                for m in METHODS
            }
            for period, early in [("early", True), ("later", False)]
        },
    }


def synthetic() -> dict[str, Any]:
    rows = []
    rng = np.random.default_rng(20260926)
    for rate in [0.02, 0.08, 0.2]:
        for amplitude in [5, 15, 30]:
            onset = np.arange(180.0, 3421, 180)
            background = rng.uniform(0, 3600, rng.poisson(rate * 3600))
            pulses = [t + rng.exponential(12, rng.poisson(amplitude)) for t in onset]
            times = np.sort(np.r_[background, *pulses])
            for method in METHODS:
                d = detect(times, method)
                rows.append(
                    {
                        "background_per_second": rate,
                        "expected_pulse_events": amplitude,
                        "method": method,
                        **match_onsets(onset, d["times"], 20),
                    }
                )
    return {
        "seed": 20260926,
        "model": "Poisson background + exponential payment-delay pulses, not real door labels",
        "rows": rows,
    }


def run(source: Path, out: Path, dates: tuple[str, ...], workers: int) -> None:
    if out.exists():
        raise ValueError("choose new experiment directory")
    out.mkdir(parents=True)
    manifest = {
        "schema_version": "boarding-dense-onset-audit.v1",
        "dates": dates,
        "routes": ROUTES,
        "methods": METHODS,
        "timezone": "Europe/Moscow",
        "seed": 20260926,
        "selection": (
            "top3 busiest30min vehicle sessions perroute/day before detectors; "
            "max12burst densitywindow"
        ),
        "implementation": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "source_sha256": digest(source / "boarding-date-shards/manifest.json"),
        "schedule_sha256": digest(source / "real-sources/timetables/timetables.json"),
        "history_sha256": digest(source / "real-sources/osm/history-manifest.json"),
        "graph_sha256": digest(source.parent.parent / "data/tram_graph.json"),
        "real_stop_accuracy": "unverified",
    }
    write_json(out / "config.json", manifest)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(partial(_day, source=source, out=out), dates))
    rows = [r for p in parts for r in p["rows"]]
    write_json(out / "summary.json", summarize(rows))
    write_json(out / "synthetic.json", synthetic())
    ranking: Counter[str] = Counter()
    for p in (source / "real-payment-stops-2025-v1").glob("2025-*/evaluation.json"):
        ranking.update(json.loads(p.read_text())["source_by_route"])
    write_json(out / "ranking.json", dict(ranking.most_common()))
    write_json(
        out / "manifest.json",
        {
            **manifest,
            "complete": True,
            "files": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    run(a.source, a.out, tuple(a.dates), a.workers)


if __name__ == "__main__":
    main()

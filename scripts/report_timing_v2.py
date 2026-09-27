"""Build paired spatial diagnostics from a completed timing-v2 experiment."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    dates = manifest["dates"]
    total_parts = []
    by_day = []
    baseline_residual = []
    baseline_steps = []
    schedules = Counter()
    durations = Counter()
    uniform_moved = Counter()
    clock_disabled = Counter()
    confidence = []
    for day in dates:
        old = args.source / "real-payment-stops-2025-v1" / day
        new = args.run / day
        pair = []
        patterns = json.loads((old / "patterns.json").read_text())
        for route, pattern in patterns.items():
            if (
                "incomplete_timetable_coverage_clock_evidence_disabled"
                in pattern["warnings"]
            ):
                clock_disabled[route] += 1
        events = pd.read_csv(
            new / "event_assignments.csv.gz", usecols=["conditional_posterior"]
        )
        confidence.extend(events.conditional_posterior.tolist())
        for label, folder in [("v1", old), ("v2", new)]:
            frame = pd.read_csv(
                folder / "stop_hour_soft_counts.csv.gz",
                dtype={"route": str, "stop_id": str, "direction": str},
            )
            if label == "v2":
                for route, part in frame.groupby("route"):
                    visits = [v for v in patterns[route]["visits"] if v["boardable"]]
                    weights = Counter(
                        (str(v["stop_id"]), str(v["direction"])) for v in visits
                    )
                    uniform_rows = []
                    for hour, mass in (
                        part.groupby("event_hour").expected_count.sum().items()
                    ):
                        for (stop, direction), weight in weights.items():
                            uniform_rows.append(
                                {
                                    "event_hour": hour,
                                    "stop_id": stop,
                                    "direction": direction,
                                    "expected_count": mass * weight / len(visits),
                                }
                            )
                    uniform = pd.DataFrame(uniform_rows)
                    columns = ["event_hour", "stop_id", "direction"]
                    values = pd.concat(
                        [
                            part.set_index(columns).expected_count.rename("v2"),
                            uniform.set_index(columns).expected_count.rename("uniform"),
                        ],
                        axis=1,
                    ).fillna(0)
                    uniform_moved[route] += float(
                        abs(values.v2 - values.uniform).sum() / 2
                    )
            grouped = frame.groupby(
                ["route", "stop_id", "direction"]
            ).expected_count.sum()
            pair.append(grouped.rename(label))
        joined = pd.concat(pair, axis=1).fillna(0)
        total_parts.append(joined)
        denominator = float(joined.v1.sum())
        by_day.append(
            {
                "date": day,
                "events": denominator,
                "daily_spatial_mass_moved_fraction": float(
                    abs(joined.v1 - joined.v2).sum() / 2 / denominator
                ),
            }
        )
        with gzip.open(old / "burst_candidates.jsonl.gz", "rt") as stream:
            for line in stream:
                b = json.loads(line)
                if b["path_steps"] > 0:
                    baseline_residual.append(abs(b["residual_seconds"]))
                    baseline_steps.append(b["path_steps"])
        audit = json.loads((new / "schedule_edges.json").read_text())
        for route in audit.values():
            for edge in route["edges"]:
                schedules[edge["reason"]] += 1
                if edge["accepted"]:
                    durations[str(edge["seconds"])] += 1
    stops = pd.concat(total_parts).groupby(level=[0, 1, 2]).sum().reset_index()
    stops["absolute_difference"] = abs(stops.v2 - stops.v1)
    stops["relative_difference"] = (stops.v2 - stops.v1) / stops.v1.replace(0, np.nan)
    catalog = pd.read_csv(
        args.source / "real-stop-training-2025-v1/stop_catalog.csv",
        dtype={"route": str, "stop_id": str, "direction": str},
    )
    catalog = catalog.sort_values("last_date").drop_duplicates(
        ["route", "stop_id", "direction"], keep="last"
    )
    stops = stops.merge(
        catalog[
            ["route", "stop_id", "direction", "name", "lat", "lon", "coordinate_source"]
        ],
        on=["route", "stop_id", "direction"],
        how="left",
        validate="one_to_one",
    )
    features = []
    for row in stops.to_dict("records"):
        if not np.isfinite(row["lat"]) or not np.isfinite(row["lon"]):
            continue
        properties = {
            k: (None if pd.isna(v) else v)
            for k, v in row.items()
            if k not in {"lat", "lon"}
        }
        properties.update(
            kind="conditional_validation_mass",
            measured_stop_truth=False,
            sample_days=len(dates),
            geometry_is_historical=False,
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                "properties": properties,
            }
        )
    by_route = []
    for route, rows in stops.groupby("route", sort=True):
        days = [d["routes"][route] for d in manifest["days"]]
        comparable = sum(d["events"]["both_have_candidate"] for d in days)
        matched = sum(d["events"]["same_stop_direction"] for d in days)
        total = float(rows.v1.sum())
        by_route.append(
            {
                "route": route,
                "events": int(total),
                "assignment_agreement": matched / comparable if comparable else None,
                "v2_distance_from_uniform_fraction": uniform_moved[route] / total,
                "clock_disabled_days": clock_disabled[route],
                "hourly_mass_moved_fraction": sum(
                    d["flow"]["mass_moved_between_stop_direction_hours"] for d in days
                )
                / total,
                "period_spatial_mass_moved_fraction": float(
                    abs(rows.v2 - rows.v1).sum() / 2 / total
                ),
                "stop_totals_pearson": float(rows.v1.corr(rows.v2)),
                "v2_raw_stable": sum(d["events"]["v2_raw_stable"] for d in days),
            }
        )
    report = {
        "dates": dates,
        "days": len(dates),
        "events": manifest["totals"]["events"],
        "scope": "20 fixed dates, not a complete 304-day replacement",
        "routes": by_route,
        "daily": by_day,
        "schedule_edge_reasons": dict(schedules),
        "schedule_durations_seconds": dict(durations),
        "v1_moving_transitions": len(baseline_steps),
        "v1_unobserved_visit_fraction": sum(k - 1 for k in baseline_steps)
        / sum(baseline_steps),
        "v1_absolute_residual_seconds": dict(
            zip(
                ["p10", "p50", "p90", "p95"],
                np.quantile(baseline_residual, [0.1, 0.5, 0.9, 0.95]).tolist(),
            )
        ),
        "stop_direction_features": len(features),
        "selected_state_posterior_quantiles": dict(
            zip(
                ["p10", "p50", "p90", "p95", "p99"],
                np.quantile(confidence, [0.1, 0.5, 0.9, 0.95, 0.99]).tolist(),
            )
        ),
        "v2_distance_from_uniform_fraction": sum(uniform_moved.values())
        / manifest["totals"]["events"],
        "note": "Candidate agreement and model residuals do not measure real stop accuracy.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stops.to_csv(args.out / "stop_comparison.csv", index=False)
    (args.out / "graph_comparison.geojson").write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features}, ensure_ascii=False
        )
    )
    (args.out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

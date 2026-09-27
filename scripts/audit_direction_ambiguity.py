"""Hash-bound audit of exact tied prefix states; no direction ground truth."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from audit_direction_pairs import cluster_interval, recover_duties
from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.clock_gtfs import profiles_for_day
from tramflow_ml.boarding.direction_ambiguity import compare_ambiguity
from tramflow_ml.boarding.direction_comparison import (
    compare_relative,
    scheduled_direction,
)
from tramflow_ml.boarding.direction_probe import duty_component, shuffle_gaps
from tramflow_ml.boarding.direction_selective import select_direction


def fit(job):
    row, profiles, duty = job
    times = [86400 + t for t in row["onsets"]]
    selected = [p for p in profiles if duty_component(p["trip_id"]) == duty]
    result = {k: row[k] for k in ("window_id", "route", "date", "split", "period")}
    result["scheduled"] = scheduled_direction(times[0], selected)
    result["baseline"] = compare_relative(times, selected)
    result["variants"] = {
        str(step): compare_ambiguity(times, selected, max_step=step) for step in (6, 3)
    }
    result["selection"] = {
        str(m): select_direction(times[:4], selected, minimum_margin_per_interval=m)
        for m in (0, 15)
    }
    result["shuffle_selection"] = {str(m): [] for m in (0, 15)}
    result["shuffle_delta"] = []
    d = result["scheduled"]["direction"]
    for index in range(19):
        seed = int(
            hashlib.sha256(f"{row['window_id']}/{index}".encode()).hexdigest()[:8], 16
        )
        shuffled = shuffle_gaps(times, seed)
        null = compare_ambiguity(shuffled, selected)
        if d is not None:
            other = str(1 - int(d))
            result["shuffle_delta"].append(
                null[other]["mean_capped_loss"] - null[d]["mean_capped_loss"]
            )
        for margin in (0, 15):
            choice = select_direction(
                shuffled[:4], selected, minimum_margin_per_interval=margin
            )
            chosen = choice["direction"]
            result["shuffle_selection"][str(margin)].append(
                {
                    "direction": chosen,
                    "advantage": null[str(1 - int(chosen))]["mean_capped_loss"]
                    - null[chosen]["mean_capped_loss"]
                    if chosen is not None
                    else None,
                }
            )
    return result


def summarize(rows, step="6"):
    labelled = [r for r in rows if r["scheduled"]["direction"] is not None]

    def mean(values):
        return float(np.mean(values)) if values else None

    pairs = []
    for r in labelled:
        d = r["scheduled"]["direction"]
        o = str(1 - int(d))
        v = r["variants"][step]
        pairs.append(
            (r, v[d], v[o], r["baseline"][d]["suffix"], r["baseline"][o]["suffix"])
        )
    both = [x for x in pairs if x[1]["all_feasible"] and x[2]["all_feasible"]]
    deltas = [b["mean_capped_loss"] - a["mean_capped_loss"] for _, a, b, _, _ in pairs]
    out = {
        "windows": len(rows),
        "labelled": len(labelled),
        "unknown": len(rows) - len(labelled),
        "date_clusters": len({r["date"] for r in labelled}),
        "assigned_mean_capped": mean(
            [a["mean_capped_loss"] for _, a, b, _, _ in pairs]
        ),
        "opposite_mean_capped": mean(
            [b["mean_capped_loss"] for _, a, b, _, _ in pairs]
        ),
        "delta": mean(deltas),
        "delta_ci": cluster_interval(labelled, deltas),
        "both_all_states_feasible": len(both),
        "paired_assigned": mean([a["mean_capped_loss"] for _, a, b, _, _ in both]),
        "paired_opposite": mean([b["mean_capped_loss"] for _, a, b, _, _ in both]),
        "assigned_ambiguous": sum(a["state_count"] > 1 for _, a, b, _, _ in pairs),
        "opposite_ambiguous": sum(b["state_count"] > 1 for _, a, b, _, _ in pairs),
        "assigned_any_feasible": sum(a["any_feasible"] for _, a, b, _, _ in pairs),
        "opposite_any_feasible": sum(b["any_feasible"] for _, a, b, _, _ in pairs),
        "assigned_all_feasible": sum(a["all_feasible"] for _, a, b, _, _ in pairs),
        "opposite_all_feasible": sum(b["all_feasible"] for _, a, b, _, _ in pairs),
        "assigned_mean_envelope_width": mean(
            [a["max_capped_loss"] - a["min_capped_loss"] for _, a, b, _, _ in pairs]
        ),
        "opposite_mean_envelope_width": mean(
            [b["max_capped_loss"] - b["min_capped_loss"] for _, a, b, _, _ in pairs]
        ),
        "prefix_only_bilateral_winners": sum(
            a["prefix_cost"] is not None
            and b["prefix_cost"] is not None
            and abs(a["prefix_cost"] - b["prefix_cost"]) > 1e-9
            for _, a, b, _, _ in pairs
        ),
        "robust_assigned_better": sum(
            a["max_capped_loss"] < b["min_capped_loss"] - 1e-9
            for _, a, b, _, _ in pairs
        ),
        "robust_opposite_better": sum(
            b["max_capped_loss"] < a["min_capped_loss"] - 1e-9
            for _, a, b, _, _ in pairs
        ),
    }
    if step == "6":
        out["baseline_assigned_capped"] = mean(
            [a["capped_loss"] for _, _, _, a, b in pairs]
        )
        out["baseline_opposite_capped"] = mean(
            [b["capped_loss"] for _, _, _, a, b in pairs]
        )
        out["baseline_refusals_with_alternative_assigned"] = sum(
            not old["feasible"] and a["any_feasible"] for _, a, _, old, _ in pairs
        )
        out["baseline_refusals_with_alternative_opposite"] = sum(
            not old["feasible"] and b["any_feasible"] for _, _, b, _, old in pairs
        )
        nulls = [float(np.mean(r["shuffle_delta"])) for r in labelled]
        contrast = [a - b for a, b in zip(deltas, nulls)]
        out["shuffle_delta"] = mean(nulls)
        out["real_minus_shuffle"] = mean(contrast)
        out["contrast_ci"] = cluster_interval(labelled, contrast)
    return out


def summarize_selection(rows, margin):
    from collections import Counter

    accepted = [r for r in rows if r["selection"][margin]["direction"] is not None]
    pairs = []
    for r in accepted:
        d = r["selection"][margin]["direction"]
        o = str(1 - int(d))
        a, b = r["variants"]["6"][d], r["variants"]["6"][o]
        pairs.append((r, a, b))
    both = [x for x in pairs if x[1]["all_feasible"] and x[2]["all_feasible"]]
    labelled = [r for r in accepted if r["scheduled"]["direction"] is not None]

    def mean(v):
        return float(np.mean(v)) if v else None

    controls = [r["shuffle_selection"][margin] for r in rows]
    return {
        "windows": len(rows),
        "accepted": len(accepted),
        "abstained": len(rows) - len(accepted),
        "date_clusters": len({r["date"] for r in accepted}),
        "reasons": dict(Counter(r["selection"][margin]["reason"] for r in rows)),
        "schedule_comparable": len(labelled),
        "schedule_agreements_not_accuracy": sum(
            r["scheduled"]["direction"] == r["selection"][margin]["direction"]
            for r in labelled
        ),
        "mean_chosen_capped": mean([a["mean_capped_loss"] for _, a, b in pairs]),
        "mean_opposite_capped": mean([b["mean_capped_loss"] for _, a, b in pairs]),
        "both_all_feasible": len(both),
        "paired_chosen": mean([a["mean_capped_loss"] for _, a, b in both]),
        "paired_opposite": mean([b["mean_capped_loss"] for _, a, b in both]),
        "chosen_all_feasible": sum(a["all_feasible"] for _, a, b in pairs),
        "opposite_all_feasible": sum(b["all_feasible"] for _, a, b in pairs),
        "wins_ties_losses": [
            sum(
                a["mean_capped_loss"] < b["mean_capped_loss"] - 1e-9 for _, a, b in both
            ),
            sum(
                abs(a["mean_capped_loss"] - b["mean_capped_loss"]) <= 1e-9
                for _, a, b in both
            ),
            sum(
                a["mean_capped_loss"] > b["mean_capped_loss"] + 1e-9 for _, a, b in both
            ),
        ],
        "shuffle_mean_acceptance_fraction": mean(
            [sum(v["direction"] is not None for v in c) / 19 for c in controls]
        ),
        "shuffle_zero_abstention_advantage": mean(
            [sum(v["advantage"] or 0 for v in c) / 19 for c in controls]
        ),
        "real_zero_abstention_advantage": sum(
            b["mean_capped_loss"] - a["mean_capped_loss"] for _, a, b in pairs
        )
        / len(rows)
        if rows
        else None,
        "abstention_zero_is_utility_not_correct_label": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "feed", "partitions", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    manifest = json.loads((args.source_run / "manifest.json").read_text())
    for path, expected in [
        (args.feed, manifest["feed_sha256"]),
        (args.source_run / "rows.json", manifest["files"]["rows.json"]),
        (args.partitions / "manifest.json", manifest["partitions_manifest_sha256"]),
    ]:
        if digest(path) != expected:
            raise ValueError("source hash mismatch")
    feed = json.loads(args.feed.read_text())
    windows = json.loads((args.source_run / "rows.json").read_text())
    duties = recover_duties(
        windows, feed, args.partitions, manifest["source_archive_sha256"]
    )
    jobs = [
        (
            r,
            profiles_for_day(feed, r["route"], date.fromisoformat(r["date"])),
            duties[r["window_id"]],
        )
        for r in windows
    ]
    args.out.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(fit, jobs))
    summary = {
        "schema": "direction-ambiguity.v1",
        "geographic_accuracy_measured": False,
        "all": summarize(rows),
        "splits": {
            s: summarize([r for r in rows if r["split"] == s])
            for s in ("exploratory", "heldout")
        },
        "heldout_step3": summarize([r for r in rows if r["split"] == "heldout"], "3"),
        "heldout_routes": {
            route: summarize(
                [r for r in rows if r["split"] == "heldout" and r["route"] == route]
            )
            for route in ("11", "12", "17")
        },
        "heldout_dates": {
            day: summarize(
                [r for r in rows if r["split"] == "heldout" and r["date"] == day]
            )
            for day in sorted({r["date"] for r in rows if r["split"] == "heldout"})
        },
    }
    summary["selective"] = {
        split: {
            str(m): summarize_selection(
                [r for r in rows if split == "all" or r["split"] == split], str(m)
            )
            for m in (0, 15)
        }
        for split in ("all", "exploratory", "heldout")
    }
    write_json(args.out / "rows.json", rows)
    write_json(args.out / "summary.json", summary)
    root = Path(__file__).resolve().parents[1]
    code = [
        "scripts/audit_direction_ambiguity.py",
        "scripts/audit_direction_pairs.py",
        "uv.lock",
        "ml/pyproject.toml",
    ]
    code += sorted(
        str(path.relative_to(root))
        for path in (root / "ml/src/tramflow_ml/boarding").glob("*.py")
    )
    write_json(
        args.out / "manifest.json",
        {
            "schema": "direction-ambiguity.v1",
            "source_manifest_sha256": digest(args.source_run / "manifest.json"),
            "feed_sha256": digest(args.feed),
            "source_rows_sha256": digest(args.source_run / "rows.json"),
            "partitions_manifest_sha256": digest(args.partitions / "manifest.json"),
            "code": {name: digest(root / name) for name in code},
            "files": {
                name: digest(args.out / name) for name in ("rows.json", "summary.json")
            },
            "config": {
                "max_steps": [6, 3],
                "prefix_size": 4,
                "failure_cost": 900,
                "shuffles": 19,
                "selective_margin_per_interval": [0, 15],
                "skip_penalty": 15,
                "exact_tie_tolerance": 1e-9,
                "state_weighting": "uniform_unique_endpoint",
            },
            "direction_truth_available": False,
            "retrospective_only": True,
            "timezone": "Europe/Moscow",
        },
    )
    print(json.dumps(summary["splits"]["heldout"], indent=2))


if __name__ == "__main__":
    main()

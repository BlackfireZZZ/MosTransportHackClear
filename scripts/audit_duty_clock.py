"""Early-only duty enrollment and one frozen shift for later whole-trip schedules."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from audit_direction_pairs import cluster_interval
from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.clock_gtfs import profiles_for_day
from tramflow_ml.boarding.direction_probe import duty_component, shuffle_gaps
from tramflow_ml.boarding.duty_clock import (
    BOUND,
    calibrate,
    causal_onsets,
    clocks,
    direction_summary,
    guarded_calibration,
    measure,
    split_profiles,
)
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import session_keys
from tramflow_ml.boarding.service_policy import load_service_policy

BASE = 86400
START, END = BASE + 6 * 3600, BASE + 10 * 3600
LATE_START, LATE_END = BASE + 14 * 3600, BASE + 22 * 3600
CUTOFF = BASE + 12 * 3600


def early_vehicle_keys(block):
    return sorted(
        block.loc[
            block.vehicle_key.ne("") & block.second.ge(START) & block.second.lt(END),
            "vehicle_key",
        ].unique()
    )


def prepare_vehicle(session, mapping, allowed):
    session = session.sort_values(["second", "event_key"])
    morning = session.loc[session.second.ge(START) & session.second.lt(END)]
    if morning.empty:
        return {"reason": "no_early_events"}
    if morning.identity_conflict.any():
        return {"reason": "early_identity_conflict"}
    exits = morning.exit_key.unique()
    if len(exits) != 1:
        return {"reason": "multiple_early_duties"}
    duty = mapping.get(exits[0])
    if duty is None:
        return {"reason": "unknown_early_duty"}
    if not all(allowed[h] for h in range(6, 10)):
        return {"reason": "early_service_exclusion"}
    onset_set = set(causal_onsets(session.second.astype(float).tolist()))
    onset_rows = session.loc[session.second.isin(onset_set)].drop_duplicates("second")
    early = (
        onset_rows.loc[
            onset_rows.second.ge(START) & onset_rows.second.lt(END), "second"
        ]
        .astype(float)
        .tolist()
    )
    if len(early) < 20 or early[-1] - early[0] < 7200:
        return {"reason": "insufficient_early_onsets"}
    return {
        "reason": "enrolled",
        "duty": duty,
        "exit_key": exits[0],
        "early": early,
        "onset_rows": onset_rows,
        "early_payments": len(morning),
    }


def shuffled(values, seed):
    return shuffle_gaps(values, seed) if len(values) > 1 else list(values)


def run_group(job):
    meta, early, late, selected, wrong = job
    a = split_profiles(selected, CUTOFF)
    b = split_profiles(wrong, CUTOFF)
    early_clocks, late_clocks = clocks(a["early"]), clocks(a["late"])
    wrong_early, wrong_late = clocks(b["early"]), clocks(b["late"])
    fit = calibrate(early, early_clocks)
    wrong_fit = calibrate(early, wrong_early)
    guard = guarded_calibration(early, early_clocks, BASE + 8 * 3600)
    wrong_guard = guarded_calibration(early, wrong_early, BASE + 8 * 3600)
    if fit["shift"] is None:
        raise ValueError("enrolled group must have calibration schedule")
    methods = {
        "zero": measure(late, late_clocks, 0),
        "fitted": measure(late, late_clocks, fit["shift"]),
        "guarded": measure(late, late_clocks, guard["shift"]),
        "wrong_guarded": measure(late, wrong_late, wrong_guard["shift"]),
        "shift_plus1020": measure(late, late_clocks, fit["shift"] + 1020),
        "wrong_zero": measure(late, wrong_late, 0),
        "wrong_fitted": measure(late, wrong_late, wrong_fit["shift"])
        if wrong_fit["shift"] is not None
        else measure(late, [], 0),
    }
    null_gains = []
    null_guarded_gains = []
    for i in range(19):
        seed = int(
            hashlib.sha256(f"{meta['group_id']}/{i}".encode()).hexdigest()[:8], 16
        )
        train_null = shuffled(early, seed)
        test_null = shuffled(late, seed + 1)
        null_fit = calibrate(train_null, early_clocks)
        null_guard = guarded_calibration(train_null, early_clocks, BASE + 8 * 3600)
        null_zero = measure(test_null, late_clocks, 0)["capped_mae"]
        null_fitted = measure(test_null, late_clocks, null_fit["shift"])["capped_mae"]
        null_gains.append(null_zero - null_fitted if null_zero is not None else None)
        guarded_loss = measure(test_null, late_clocks, null_guard["shift"])[
            "capped_mae"
        ]
        null_guarded_gains.append(
            null_zero - guarded_loss if null_zero is not None else None
        )
    complete = [
        p
        for p in a["late"]
        if p["times"][0] - BOUND >= LATE_START and p["times"][-1] + BOUND < LATE_END
    ]
    complete_times = [
        t
        for t in late
        if any(p["times"][0] - BOUND <= t <= p["times"][-1] + BOUND for p in complete)
    ]
    phase = {}
    for label, lo, hi in [
        ("14-18", LATE_START, BASE + 18 * 3600),
        ("18-22", BASE + 18 * 3600, LATE_END),
    ]:
        events = [t for t in late if lo <= t < hi]
        phase[label] = {
            name: measure(events, late_clocks, s)
            for name, s in [("zero", 0), ("fitted", fit["shift"])]
        }
    return {
        **meta,
        "early_onsets": len(early),
        "late_onsets": len(late),
        "fit": fit,
        "wrong_fit": wrong_fit,
        "guard": guard,
        "wrong_guard": wrong_guard,
        "guarded_direction": direction_summary(late, a["late"], guard),
        "methods": methods,
        "direction": direction_summary(late, a["late"], fit),
        "zero_direction": direction_summary(
            late, a["late"], {"shift": 0, "near_shifts": [0]}
        ),
        "early_trips": len(a["early"]),
        "late_trips": len(a["late"]),
        "crossing_trips": a["crossing"],
        "early_schedule_clocks": len(early_clocks),
        "wrong_early_schedule_clocks": len(wrong_early),
        "wrong_early_trips": len(b["early"]),
        "wrong_late_trips": len(b["late"]),
        "late_schedule_clocks": len(late_clocks),
        "wrong_late_schedule_clocks": len(wrong_late),
        "late_overlapping_trip_pairs": sum(
            max(p["times"][0], q["times"][0]) <= min(p["times"][-1], q["times"][-1])
            for i, p in enumerate(a["late"])
            for q in a["late"][i + 1 :]
        ),
        "complete_reference_trips": len(complete),
        "complete_observations": len(complete_times),
        "complete_trip_proxy": {
            name: measure(complete_times, late_clocks, s)
            for name, s in [("zero", 0), ("fitted", fit["shift"])]
        },
        "late_phases": phase,
        "shuffle_gains": null_gains,
        "shuffle_guarded_gains": null_guarded_gains,
    }


def summarize(rows):
    evaluated = [r for r in rows if r["late_onsets"]]

    def avg(v):
        return float(np.mean(v)) if v else None

    methods = (
        "zero",
        "fitted",
        "guarded",
        "shift_plus1020",
        "wrong_zero",
        "wrong_fitted",
        "wrong_guarded",
    )
    gains = [
        r["methods"]["zero"]["capped_mae"] - r["methods"]["fitted"]["capped_mae"]
        for r in evaluated
    ]
    wrong = [
        r["methods"]["wrong_fitted"]["capped_mae"]
        - r["methods"]["fitted"]["capped_mae"]
        for r in evaluated
    ]
    contrasts = [
        g - float(np.mean(r["shuffle_gains"])) for r, g in zip(evaluated, gains)
    ]
    n = sum(r["late_onsets"] for r in evaluated)
    complete = [r for r in rows if r["complete_observations"]]
    guarded_gains = [
        r["methods"]["zero"]["capped_mae"] - r["methods"]["guarded"]["capped_mae"]
        for r in evaluated
    ]
    guard_contrasts = [
        g - float(np.mean(r["shuffle_guarded_gains"]))
        for r, g in zip(evaluated, guarded_gains)
    ]
    guard_accepted = [r for r in evaluated if r["guard"]["accepted"]]

    return {
        "guarded_accepted_enrolled": sum(r["guard"]["accepted"] for r in rows),
        "guarded_accepted_evaluated": len(guard_accepted),
        "guard_reasons": dict(Counter(r["guard"]["reason"] for r in rows)),
        "guarded_gain": avg(guarded_gains),
        "guarded_gain_ci": cluster_interval(evaluated, guarded_gains),
        "guarded_contrast": avg(guard_contrasts),
        "guarded_contrast_ci": cluster_interval(evaluated, guard_contrasts),
        "guard_accepted_only_gain": avg(
            [
                r["methods"]["zero"]["capped_mae"]
                - r["methods"]["guarded"]["capped_mae"]
                for r in guard_accepted
            ]
        ),
        "guard_accepted_only_fitted_loss": avg(
            [r["methods"]["guarded"]["capped_mae"] for r in guard_accepted]
        ),
        "guarded_direction_assigned": sum(
            r["guarded_direction"]["assigned"] for r in rows
        ),
        "guarded_direction_stable": sum(r["guarded_direction"]["stable"] for r in rows),
        "enrolled": len(rows),
        "evaluated": len(evaluated),
        "no_late_evidence": len(rows) - len(evaluated),
        "date_clusters": len({r["date"] for r in evaluated}),
        "early_onsets": sum(r["early_onsets"] for r in rows),
        "late_onsets": n,
        "group_mean_loss": {
            m: avg([r["methods"][m]["capped_mae"] for r in evaluated]) for m in methods
        },
        "onset_weighted_loss": {
            m: sum(r["methods"][m]["capped_mae"] * r["late_onsets"] for r in evaluated)
            / n
            if n
            else None
            for m in methods
        },
        "within60": {
            m: sum(r["methods"][m]["within60"] for r in evaluated) for m in methods
        },
        "zero_minus_fitted": avg(gains),
        "gain_ci": cluster_interval(evaluated, gains),
        "wrong_minus_fitted": avg(wrong),
        "wrong_ci": cluster_interval(evaluated, wrong),
        "real_gain_minus_shuffle_gain": avg(contrasts),
        "contrast_ci": cluster_interval(evaluated, contrasts),
        "wins_ties_losses": [
            sum(g > 1e-9 for g in gains),
            sum(abs(g) <= 1e-9 for g in gains),
            sum(g < -1e-9 for g in gains),
        ],
        "bound_hits": sum(r["fit"]["at_bound"] for r in rows),
        "shift_quantiles": np.quantile(
            [r["fit"]["shift"] for r in rows], [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
        if rows
        else [],
        "near_shift_count_median": float(
            np.median([len(r["fit"]["near_shifts"]) for r in rows])
        )
        if rows
        else None,
        "near_shift_span_median": float(
            np.median(
                [
                    max(r["fit"]["near_shifts"]) - min(r["fit"]["near_shifts"])
                    for r in rows
                ]
            )
        )
        if rows
        else None,
        "missing_own_late_schedule": sum(r["late_schedule_clocks"] == 0 for r in rows),
        "missing_wrong_late_schedule": sum(
            r["wrong_late_schedule_clocks"] == 0 for r in rows
        ),
        "missing_wrong_early_fit": sum(r["wrong_fit"]["shift"] is None for r in rows),
        "late_policy_excluded_onsets": sum(
            r["late_policy_excluded_onsets"] for r in rows
        ),
        "late_conflict_excluded_onsets": sum(
            r["late_conflict_excluded_onsets"] for r in rows
        ),
        "late_duty_changed_groups": sum(
            r["late_changed_duty_events"] > 0 for r in rows
        ),
        "late_changed_duty_events": sum(r["late_changed_duty_events"] for r in rows),
        "late_conflict_events": sum(r["late_conflict_events"] for r in rows),
        "direction_assigned": sum(r["direction"]["assigned"] for r in rows),
        "direction_stable": sum(r["direction"]["stable"] for r in rows),
        "zero_direction_assigned": sum(r["zero_direction"]["assigned"] for r in rows),
        "direction_comparable_to_zero": sum(
            r["direction"]["comparable_to_zero"] for r in rows
        ),
        "direction_changed_from_zero": sum(
            r["direction"]["changed_from_zero"] for r in rows
        ),
        "late_schedule_clocks": sum(r["late_schedule_clocks"] for r in rows),
        "wrong_late_schedule_clocks": sum(
            r["wrong_late_schedule_clocks"] for r in rows
        ),
        "overlapping_late_trip_pairs": sum(
            r["late_overlapping_trip_pairs"] for r in rows
        ),
        "complete_proxy_groups": len(complete),
        "complete_proxy_gain": avg(
            [
                r["complete_trip_proxy"]["zero"]["capped_mae"]
                - r["complete_trip_proxy"]["fitted"]["capped_mae"]
                for r in complete
            ]
        ),
        "late_phase_gain": {
            p: avg(
                [
                    r["late_phases"][p]["zero"]["capped_mae"]
                    - r["late_phases"][p]["fitted"]["capped_mae"]
                    for r in rows
                    if r["late_phases"][p]["zero"]["events"]
                ]
            )
            for p in ("14-18", "18-22")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-manifest",
        "partitions",
        "feed",
        "service-policy",
        "source-root",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new output directory required")
    source = json.loads(args.source_manifest.read_text())
    for path, key in [
        (args.partitions / "manifest.json", "partitions_manifest_sha256"),
        (args.feed, "feed_sha256"),
        (args.service_policy, "service_policy_sha256"),
    ]:
        if digest(path) != source[key]:
            raise ValueError("source hash mismatch: " + key)
    feed = json.loads(args.feed.read_text())
    policy = load_service_policy(args.service_policy, source_root=args.source_root)
    jobs = []
    enrollment = []
    for day in source["dates"]:
        raw = read_day(args.partitions, day, tuple(source["routes"]))
        raw = raw.loc[raw.success.eq("1")].copy()
        raw["second"] = (
            pd.to_datetime(raw.event_at).dt.hour * 3600
            + pd.to_datetime(raw.event_at).dt.minute * 60
            + pd.to_datetime(raw.event_at).dt.second
            + BASE
        )
        raw = session_keys(raw)
        for route in source["routes"]:
            profiles = profiles_for_day(feed, route, date.fromisoformat(day))
            duties = sorted({duty_component(p["trip_id"]) for p in profiles})
            if None in duties:
                raise ValueError("unrecognized duty suffix")
            mapping = {
                hashlib.sha256(
                    (source["source_archive_sha256"] + ":exit:" + d).encode()
                ).hexdigest()[:24]: d
                for d in duties
            }
            allowed = {
                hour: policy.evaluate(
                    route,
                    datetime.fromisoformat(day).replace(
                        hour=hour, tzinfo=ZoneInfo("Europe/Moscow")
                    ),
                    engine_applicable=True,
                    warnings=(),
                    soft=True,
                ).eligible
                for hour in range(24)
            }
            block = raw.loc[raw.route.eq(route)].copy()
            counts = Counter(
                successful_events=len(block),
                missing_vehicle_events=int(block.vehicle_key.eq("").sum()),
            )
            vehicles = early_vehicle_keys(block)
            counts["no_early_events"] = block.loc[
                block.vehicle_key.ne(""), "vehicle_key"
            ].nunique() - len(vehicles)
            for ordinal, vehicle in enumerate(vehicles):
                session = block.loc[block.vehicle_key.eq(vehicle)].sort_values(
                    ["second", "event_key"]
                )
                prepared = prepare_vehicle(session, mapping, allowed)
                if prepared["reason"] != "enrolled":
                    counts[prepared["reason"]] += 1
                    continue
                duty, early = prepared["duty"], prepared["early"]
                onset_rows = prepared["onset_rows"]
                selected = [p for p in profiles if duty_component(p["trip_id"]) == duty]
                split = split_profiles(selected, CUTOFF)
                if len(split["early"]) < 2:
                    counts["fewer_than_two_early_reference_trips"] += 1
                    continue
                wrong = (
                    duties[(duties.index(duty) + 1) % len(duties)]
                    if len(duties) > 1
                    else None
                )
                incorrect = [
                    p for p in profiles if duty_component(p["trip_id"]) == wrong
                ]
                later = session.loc[
                    session.second.ge(LATE_START) & session.second.lt(LATE_END)
                ]
                late_rows = onset_rows.loc[
                    onset_rows.second.ge(LATE_START) & onset_rows.second.lt(LATE_END)
                ]
                policy_mask = late_rows.second.map(
                    lambda t: allowed[int((t - BASE) // 3600)]
                )
                late = (
                    late_rows.loc[policy_mask & ~late_rows.identity_conflict, "second"]
                    .astype(float)
                    .tolist()
                )
                meta = {
                    "group_id": f"{day}/{route}/{ordinal}",
                    "date": day,
                    "route": route,
                    "split": "exploratory"
                    if day < "2025-07-01"
                    else "retrospective_late_dates",
                    "early_payments": prepared["early_payments"],
                    "late_payments": len(later),
                    "late_changed_duty_events": int(
                        later.exit_key.ne(prepared["exit_key"]).sum()
                    ),
                    "late_conflict_events": int(later.identity_conflict.sum()),
                    "late_policy_excluded_onsets": int((~policy_mask).sum()),
                    "late_conflict_excluded_onsets": int(
                        (policy_mask & late_rows.identity_conflict).sum()
                    ),
                }
                jobs.append((meta, early, late, selected, incorrect))
                counts["enrolled"] += 1
            enrollment.append(
                {
                    "date": day,
                    "route": route,
                    "counts": dict(counts),
                    "excluded_hours": [h for h in range(24) if not allowed[h]],
                }
            )
        print("enrolled", day, len(jobs), flush=True)
    args.out.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_group, jobs):
            rows.append(row)
            if len(rows) % 25 == 0:
                print("fit", len(rows), "/", len(jobs), flush=True)
    summary = {
        "schema": "duty-clock.v1",
        "all": summarize(rows),
        "splits": {
            s: summarize([r for r in rows if r["split"] == s])
            for s in ("exploratory", "retrospective_late_dates")
        },
        "routes": {
            r: summarize([x for x in rows if x["route"] == r]) for r in source["routes"]
        },
        "dates": {
            d: summarize([r for r in rows if r["date"] == d]) for d in source["dates"]
        },
        "unchanged_late_duty_sensitivity": summarize(
            [r for r in rows if r["late_changed_duty_events"] == 0]
        ),
        "bilateral_schedule_control": summarize(
            [
                r
                for r in rows
                if r["wrong_fit"]["shift"] is not None
                and r["wrong_late_schedule_clocks"]
                and r["late_schedule_clocks"]
            ]
        ),
        "bilateral_schedule_routes": {
            route: summarize(
                [
                    r
                    for r in rows
                    if r["route"] == route
                    and r["wrong_fit"]["shift"] is not None
                    and r["wrong_late_schedule_clocks"]
                    and r["late_schedule_clocks"]
                ]
            )
            for route in source["routes"]
        },
        "enrollment": enrollment,
        "geographic_accuracy_measured": False,
    }
    write_json(args.out / "rows.json", rows)
    write_json(args.out / "summary.json", summary)
    root = Path(__file__).resolve().parents[1]
    code = [
        "scripts/audit_duty_clock.py",
        "scripts/audit_direction_pairs.py",
        "ml/pyproject.toml",
        "uv.lock",
    ]
    code += sorted(
        str(p.relative_to(root))
        for p in (root / "ml/src/tramflow_ml/boarding").glob("*.py")
    )
    write_json(
        args.out / "manifest.json",
        {
            "schema": "duty-clock.v1",
            "source_manifest_sha256": digest(args.source_manifest),
            "sources": {
                k: source[k]
                for k in (
                    "feed_sha256",
                    "partitions_manifest_sha256",
                    "service_policy_sha256",
                    "source_archive_sha256",
                )
            },
            "code": {p: digest(root / p) for p in code},
            "files": {p: digest(args.out / p) for p in ("rows.json", "summary.json")},
            "config": {
                "early_hours": [6, 10],
                "late_hours": [14, 22],
                "trip_split_hour": 12,
                "shift_bound": 1800,
                "shift_step": 30,
                "residual_cap": 300,
                "near_optimal_mae_delta": 5,
                "onset_silence": 60,
                "minimum_early_onsets": 20,
                "minimum_early_span": 7200,
                "shuffles": 19,
                "guard": {
                    "midpoint_hour": 8,
                    "minimum_events_per_half": 10,
                    "maximum_near_shift_span": 120,
                    "maximum_shift_disagreement": 60,
                    "minimum_early_validation_gain": 5,
                },
            },
            "direction_truth_available": False,
            "historical_operation_confirmed": False,
            "timezone": "Europe/Moscow",
        },
    )
    print(json.dumps(summary["all"], indent=2), flush=True)


if __name__ == "__main__":
    main()

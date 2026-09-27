"""Frozen coverage-selected device pilot and bounded detector sensitivity study."""

import csv
import gzip
import io
import json
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, verify_run, write_json
from .bursts import Burst, Payment, detect
from .stability import evaluate_stability


def select_cohort(run: Path) -> dict[str, Any]:
    totals_map: dict[str, int] = {}
    route_days: dict[str, set[str]] = {}
    days: set[str] = set()
    with (run / "mass_ledger.csv").open() as f:
        for row in csv.DictReader(f):
            day = row["event_hour"][:10]
            days.add(day)
            route = row["route"]
            if route in {"1", "7", "11", "12"}:
                totals_map[route] = totals_map.get(route, 0) + int(row["source_success"])
                route_days.setdefault(route, set()).add(day)
    totals = [(r, n, len(route_days[r])) for r, n in totals_map.items()]
    ranked = sorted(totals, key=lambda r: (r[1] / max(r[2], 1), r[0]))
    routes = sorted({ranked[0][0], ranked[-1][0]}) if ranked else []
    splits: dict[str, list[str]] = {}
    for split, lo, hi in [
        ("train", "2025-01-01", "2025-07-01"),
        ("development", "2025-07-01", "2025-09-01"),
        ("diagnostic", "2025-09-01", "2025-11-01"),
    ]:
        candidates = sorted(
            d for d in days if lo <= d < hi and date.fromisoformat(d).weekday() == 4
        )
        for candidate in candidates:
            block = [str(date.fromisoformat(candidate) + timedelta(days=i)) for i in range(4)]
            if all(d in days and d < hi for d in block):
                splits[split] = block
                break
        else:
            splits[split] = sorted(d for d in days if lo <= d < hi)[:4]
    return {
        "schema_version": "boarding-cohort.v1",
        "routes": routes,
        "dates": splits,
        "selection": "lowest/highest successful rows per observed day among 1/7/11/12; "
        "first covered Friday-Monday per chronological split",
        "route_density": {r: n / max(d, 1) for r, n, d in totals},
        "negative_control": "route5: audit all successful rows, no manufactured journeys",
        "frozen_before_detection": True,
        "complete_service_days_verified": False,
        "exclude_bad_vehicle_days": False,
        "diagnostic_is_blind": False,
    }


def pilot(run: Path, out: Path) -> dict[str, Any]:
    started = time.monotonic()
    verify_run(run)
    if out.exists() and any(out.iterdir()):
        raise ValueError("pilot output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    cohort = select_cohort(run)
    write_json(out / "cohort_manifest.json", cohort)
    dates = {d for block in cohort["dates"].values() for d in block}
    blocks = []
    manifest = json.loads((run / "manifest.json").read_text())
    for shard in manifest["shards"]:
        name = shard["name"]
        frame = pd.read_csv(run / "shards" / name, dtype=str, keep_default_na=False)
        chosen = (
            frame.event_at.str[:10].isin(dates)
            & frame.route.isin(cohort["routes"])
            & frame.in_range.eq("1")
        )
        if bool(chosen.any()):
            blocks.append(frame.loc[chosen])
    frame = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    events: list[Payment] = []
    if not frame.empty:
        stamp = pd.to_datetime(frame.event_at, format="%Y-%m-%d %H:%M:%S")
        # pandas 3 uses microsecond resolution unless explicitly converted to ns.
        seconds = stamp.dt.tz_localize("Europe/Moscow").dt.as_unit("ns").astype("int64") / 1e9
        events = [
            Payment(
                str(r.event_key),
                float(t),
                str(r.device_key),
                str(r.vehicle_key),
                str(r.route),
                str(r.exit_key),
                r.success == "1",
            )
            for r, t in zip(frame.itertuples(index=False), seconds, strict=True)
        ]
    write_json(out / "stability.json", evaluate_stability(events))
    success = sum(e.success for e in events)
    config_grid = [(10, 30), (10, 90), (30, 90), (60, 180)]
    comparisons = []
    selected: tuple[Burst, ...] = ()
    for gap, span in config_grid:
        began = time.monotonic()
        bursts = detect(events, gap=gap, max_span=span)
        sizes = [b.successful for b in bursts]
        comparisons.append(
            {
                "detector": "D0",
                "gap": gap,
                "max_span": span,
                "bursts": len(bursts),
                "successful_mass": sum(sizes),
                "singleton_fraction": sum(len(b.keys) == 1 for b in bursts) / max(len(bursts), 1),
                "size_p50": float(np.median(sizes)) if sizes else None,
                "size_p90": float(np.quantile(sizes, 0.9)) if sizes else None,
                "wall_seconds": time.monotonic() - began,
                "geographic_accuracy": "unverified",
            }
        )
        if sum(sizes) != success or sum(len(b.keys) for b in bursts) != len(events):
            raise ValueError("G0 detector mass mismatch")
        if (gap, span) == (30, 90):
            selected = bursts
    group_rows: Counter[tuple[str, int]] = Counter()
    device_second: Counter[tuple[str, str, int]] = Counter()
    vehicle_second: dict[tuple[str, int], set[str]] = {}
    second_vehicles: dict[int, set[str]] = {}
    negative = 0
    previous: dict[tuple[str, str, str, str], float] = {}
    for e in events:
        key = (e.device, e.vehicle, e.route, e.exit)
        if key in previous and e.second < previous[key]:
            negative += 1
        previous[key] = e.second
        if e.success:
            group_rows[(e.route, int(e.second) // 3600)] += 1
            if e.device:
                device_second[(e.device, e.vehicle, int(e.second))] += 1
            if e.vehicle and e.device:
                vehicle_second.setdefault((e.vehicle, int(e.second)), set()).add(e.device)
                second_vehicles.setdefault(int(e.second), set()).add(e.vehicle)
    with (
        (out / "burst_hypotheses.csv.gz").open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as gz,
    ):
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(
                [
                    "burst_id",
                    "start",
                    "end",
                    "success",
                    "rejected",
                    "device",
                    "route",
                    "max_gap",
                    "identity_status",
                    "stop_id",
                ]
            )
            for i, b in enumerate(selected):
                w.writerow(
                    [
                        i,
                        b.start,
                        b.end,
                        b.successful,
                        b.rejected,
                        b.device,
                        b.route,
                        b.max_gap,
                        b.identity_status,
                        "",
                    ]
                )
    event_lookup = {e.event_key: e for e in events}
    with (
        (out / "event_to_burst.csv.gz").open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as gz,
    ):
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["event_key", "burst_id", "stop_id", "unresolved_weight"])
            for i, b in enumerate(selected):
                w.writerows(
                    (event_key, i, "", int(event_lookup[event_key].success)) for event_key in b.keys
                )
    features: dict[tuple[str, int], list[int]] = {}
    for b in selected:
        supported_hours = {
            (event_lookup[k].route, int(event_lookup[k].second) // 3600)
            for k in b.keys
            if event_lookup[k].success
        }
        for hour_key in supported_hours:
            features.setdefault(hour_key, []).append(b.successful)
    with (out / "route_hour_burst_features.csv").open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "route",
                "event_hour",
                "validation_count",
                "bursts_touching_hour",
                "burst_size_p50",
                "burst_size_p90",
                "feature_available_after",
                "offline_only",
            ]
        )
        for (route, hour), n in sorted(group_rows.items()):
            sizes = features[(route, hour)]
            t = datetime.fromtimestamp(hour * 3600, ZoneInfo("Europe/Moscow"))
            w.writerow(
                [
                    route,
                    t.isoformat(),
                    n,
                    len(sizes),
                    float(np.median(sizes)),
                    float(np.quantile(sizes, 0.9)),
                    "completed_selected_window",
                    True,
                ]
            )
    report = {
        "schema_version": "boarding-pilot.v1",
        "events": len(events),
        "successful": success,
        "rejected": len(events) - success,
        "comparisons": comparisons,
        "selected_default": {"gap": 30, "max_span": 90},
        "selection_reason": "predeclared scenario; no quality-based winner without gold",
        "time_quality": {
            "missing_device_success_excluded": sum(e.success and not e.device for e in events),
            "within_device_same_second_excess": sum(max(n - 1, 0) for n in device_second.values()),
            "vehicle_seconds_multiple_devices": sum(len(v) > 1 for v in vehicle_second.values()),
            "seconds_multiple_vehicles": sum(len(v) > 1 for v in second_vehicles.values()),
            "negative_gaps_in_source_order": negative,
            "clock_correction": False,
            "batching_proven": False,
        },
        "B0": {"assignment_coverage": 0, "unresolved_mass": success},
        "B1": {"status": "not_applicable", "reason": "no validated historical trip binding"},
        "B2_DP_real": {"status": "not_applicable", "reason": "unverified historical patterns"},
        "G0": "passed",
        "G1": "device_only",
        "G3": "unverified",
        "real_stop_accuracy": "unverified",
        "certified_stop_labels": 0,
        "expected_visit_count": None,
        "coverage_ledger": [],
        "journey_boundaries": "unresolved; long gaps and midnight are not inferred turnarounds",
        "feature_use": "retrospective descriptive only; use cutoff detector for forecasting",
        "wall_seconds": time.monotonic() - started,
    }
    write_json(out / "evaluation.json", report)
    write_json(
        out / "manifest.json",
        {
            "schema_version": "boarding-pilot-manifest.v1",
            "complete": True,
            "source_manifest_sha256": digest(run / "manifest.json"),
            "cohort_sha256": digest(out / "cohort_manifest.json"),
            "files": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
            "implementation": {
                p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("bursts.py")]
            },
            "stop_training_eligible": False,
        },
    )
    return report

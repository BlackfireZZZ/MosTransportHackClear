"""Frozen chronological comparison of adaptive payment pulse candidates."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, write_json
from .burst_detection import detect, match_onsets
from .dense_experiment import busiest
from .multiscale import SCALES, FeatureMixture, controls, detect_all, representation
from .partitions import read_day
from .real import ROUTES, session_keys
from .timing_experiment import DEFAULT_DATES

MAIN = [
    "decay_onset",
    "gap30",
    "scan2",
    "scan3",
    "scan5",
    "scan10",
    "adaptive_scan",
    "adaptive_scan_q90",
    "adaptive_scan_q99",
    "hdbscan3",
    "hdbscan5",
    "hdbscan8",
    "hdbscan5_raw",
    "feature_gmm",
    "synchrony",
    "consensus2_2s",
    "consensus2_5s",
    "consensus2_10s",
    "consensus3_5s",
    "algorithm_ml",
]
FAMILIES = ["adaptive_scan", "hdbscan5", "feature_gmm", "synchrony"]


def prepare(source: Path, old: Path) -> list[dict[str, Any]]:
    windows = []
    for day in DEFAULT_DATES:
        prior = json.loads((old / (day + ".json")).read_text())["rows"]
        keys = {(r["route"], r["session_key"]): r for r in prior if r["method"] == "gap30"}
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
        for (route, key), block in frame.groupby(["route", "group"], sort=True):
            if (route, key) not in keys:
                continue
            assert not block.identity_conflict.any()
            block = block.sort_values("second", kind="stable")
            t = block.second.to_numpy(dtype=float)
            roi = busiest(t)
            use = (t >= roi[0]) & (t <= roi[-1])
            device_keys = block.device_key.to_numpy()[use]
            d = pd.factorize(device_keys, sort=True)[0]
            d[device_keys == ""] = -1
            assert len(roi) == keys[(route, key)]["source_events"]
            windows.append(
                {
                    "date": day,
                    "route": route,
                    "rank": keys[(route, key)]["rank"],
                    "times": (roi - np.floor(roi[0] / 300) * 300).tolist(),
                    "devices": d.tolist(),
                }
            )
        print(json.dumps({"prepared": day, "total": len(windows)}), flush=True)
    return windows


def fit(windows: list[dict[str, Any]]) -> tuple[FeatureMixture, dict[str, Any]]:
    reps, flags, dates = [], [], []
    null_reps = []
    for i, window in enumerate(windows):
        if window["date"] >= "2025-05-01":
            continue
        t, d = np.array(window["times"]), np.array(window["devices"])
        a = representation(t, d)
        u, v = controls(t, d, 20260926 + i, "uniform")
        b = representation(u, v)
        reps.extend([a["features"], b["features"]])
        flags.extend([np.ones(len(a["times"]), dtype=bool), np.zeros(len(b["times"]), dtype=bool)])
        dates.extend([window["date"]] * (len(a["times"]) + len(b["times"])))
        null_reps.append(b)
    x, y, day = np.vstack(reps), np.concatenate(flags), np.array(dates)
    sample = np.random.default_rng(20260926).choice(len(x), min(30000, len(x)), replace=False)
    mixture = FeatureMixture().fit(x[sample], y[sample], day[sample])
    scans = np.vstack([r["scan"] for r in null_reps])
    sync = np.vstack([r["sync"] for r in null_reps])
    thresholds: dict[str, Any] = {
        f"scan{w}": float(np.quantile(scans[:, j], 0.95)) for j, w in enumerate(SCALES[:4])
    }
    thresholds.update(
        {
            "adaptive": {
                str(q): float(np.quantile(scans.max(axis=1), q / 100)) for q in [90, 95, 99]
            },
            "sync": float(np.quantile(sync[:, :4].max(axis=1), 0.95)),
            "gmm": float(
                np.quantile(mixture.score(np.vstack([r["features"] for r in null_reps])), 0.95)
            ),
            "fit_candidates": len(sample),
            "available_training_candidates": len(x),
            "calibration_candidates": len(scans),
        }
    )
    return mixture, thresholds


def all_methods(t: Any, d: Any, model: FeatureMixture, thresholds: dict[str, Any]) -> Any:
    out = detect_all(t, d, model, thresholds)
    for method in ["decay_onset", "gap30"]:
        out[method] = detect(t, method)["times"]
    return out


def profile(at: Any, t: Any, start: float, end: float) -> dict[str, int]:
    at = np.asarray(at)
    at = at[(at >= start + 90) & (at <= end - 90)]
    bounds = {
        "before30": (-30, 0),
        "first3": (0, 3),
        "first10": (0, 10),
        "first30": (0, 30),
        "tail30_90": (30, 90),
        "near2": (-2, 3),
    }
    result = {"onsets": len(at)}
    for name, (a, b) in bounds.items():
        result[name] = int((np.searchsorted(t, at + b) - np.searchsorted(t, at + a)).sum())
    return result


def evaluate(
    item: tuple[int, dict[str, Any]], model: FeatureMixture, thresholds: dict[str, Any]
) -> dict[str, Any]:
    i, window = item
    t, d = np.array(window["times"]), np.array(window["devices"])
    seed = 20260926 + i * 100
    actual = all_methods(t, d, model, thresholds)
    nulls: dict[str, Any] = {"uniform": [], "device_shift": []}
    for kind in nulls:
        for repeat in range(3):
            u, v = controls(t, d, seed + repeat, kind)
            nulls[kind].append(all_methods(u, v, model, thresholds))
    mask = np.random.default_rng(seed).random(len(t)) < 0.8
    thin = all_methods(t[mask], d[mask], model, thresholds)
    heldout = t[~mask]
    independent: dict[str, Any] = {}
    devices = np.unique(d[d >= 0])
    if len(devices) >= 2:
        use = np.isin(d, devices[::2])
        if use.sum() >= 10 and (~use).sum() >= 10:
            partial_methods = all_methods(t[use], d[use], model, thresholds)
            other_mask = (~use) & (d >= 0)
            other = t[other_mask]
            shifted, _ = controls(other, d[other_mask], seed + 10, "device_shift")
            for method in MAIN:
                independent[method] = {
                    "real": profile(partial_methods[method], other, t[0], t[-1]),
                    "shifted": profile(partial_methods[method], shifted, t[0], t[-1]),
                }
    rows = {}
    for method in MAIN:
        at = actual[method]
        gaps = np.diff(at)
        stats = profile(thin[method], heldout, t[0], t[-1])
        rows[method] = {
            "onsets": len(at),
            "times": at.tolist(),
            "null": {kind: [len(v[method]) for v in variants] for kind, variants in nulls.items()},
            "thinning": {str(tol): match_onsets(at, thin[method], tol) for tol in [2, 5, 10]},
            "heldout_payments": stats,
            "heldout_devices": independent.get(method),
            "short_gaps_under20": int((gaps < 20).sum()),
            "gaps": len(gaps),
        }
    pairwise = {}
    for a, b in combinations(FAMILIES, 2):
        pairwise[a + "/" + b] = {
            "real": {str(tol): match_onsets(actual[a], actual[b], tol) for tol in [2, 5, 10]},
            "uniform": [match_onsets(v[a], v[b], 5) for v in nulls["uniform"]],
            "device_shift": [match_onsets(v[a], v[b], 5) for v in nulls["device_shift"]],
        }
    return {
        "date": window["date"],
        "route": window["route"],
        "rank": window["rank"],
        "source_events": len(t),
        "devices": len(devices),
        "methods": rows,
        "pairwise": pairwise,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def section(part: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "windows": len(part),
            "source_events": sum(r["source_events"] for r in part),
            "methods": {},
            "pairwise": {},
        }
        for method in MAIN:
            values = [r["methods"][method] for r in part]
            out: dict[str, Any] = {
                k: sum(v[k] for v in values) for k in ["onsets", "gaps", "short_gaps_under20"]
            }
            out["null"] = {
                kind: np.sum([v["null"][kind] for v in values], axis=0).tolist()
                for kind in ["uniform", "device_shift"]
            }
            out["thinning"] = {
                str(tol): {
                    k: sum(v["thinning"][str(tol)][k] for v in values)
                    for k in ["reference", "detected", "matched"]
                }
                for tol in [2, 5, 10]
            }
            out["heldout_payments"] = (
                {
                    k: sum(v["heldout_payments"][k] for v in values)
                    for k in values[0]["heldout_payments"]
                }
                if values
                else {}
            )
            device_values = [v["heldout_devices"] for v in values if v["heldout_devices"]]
            out["heldout_device_windows"] = len(device_values)
            out["heldout_devices"] = (
                {
                    kind: {
                        k: sum(v[kind][k] for v in device_values) for k in device_values[0][kind]
                    }
                    for kind in ["real", "shifted"]
                }
                if device_values
                else {}
            )
            result["methods"][method] = out
        for a, b in combinations(FAMILIES, 2):
            pair = a + "/" + b
            values = [r["pairwise"][pair] for r in part]
            result["pairwise"][pair] = {
                "real": {
                    str(tol): {
                        k: sum(v["real"][str(tol)][k] for v in values)
                        for k in ["reference", "detected", "matched"]
                    }
                    for tol in [2, 5, 10]
                },
                **{
                    kind: [
                        {
                            k: sum(v[kind][rep][k] for v in values)
                            for k in ["reference", "detected", "matched"]
                        }
                        for rep in range(3)
                    ]
                    for kind in ["uniform", "device_shift"]
                },
            }
        return result

    test = [r for r in rows if r["date"] >= "2025-05-01"]
    return {
        "train": section([r for r in rows if r["date"] < "2025-05-01"]),
        "test": section(test),
        "test_routes": {
            route: section([r for r in test if r["route"] == route]) for route in ROUTES
        },
        "test_months": {
            month: section([r for r in test if r["date"][5:7] == month])
            for month in sorted({r["date"][5:7] for r in test})
        },
    }


def queued_times(events: Any, devices: Any) -> tuple[Any, Any]:
    events, devices = np.asarray(events, dtype=float), np.asarray(devices)
    serviced = events.copy()
    for device in np.unique(devices):
        indices = np.flatnonzero(devices == device)
        indices = indices[np.argsort(events[indices], kind="stable")]
        previous = -np.inf
        for i in indices:
            serviced[i] = max(events[i], previous + 1.5)
            previous = serviced[i]
    order = np.argsort(serviced, kind="stable")
    return np.floor(serviced[order]), devices[order]


def synthetic(model: FeatureMixture, thresholds: dict[str, Any]) -> dict[str, Any]:
    rows = []
    rng = np.random.default_rng(20260926)
    for size in [5, 6, 20, 50]:
        for tail in [2, 12, 40]:
            for device_count in [1, 3, 6]:
                truth = np.arange(180.0, 3421, 180)
                events, devices = [], []
                for onset in truth:
                    labels = rng.integers(0, device_count, size)
                    arrivals = rng.exponential(tail, size)
                    events.extend((onset + arrivals).tolist())
                    devices.extend(labels.tolist())
                n = rng.poisson(0.025 * 3600)
                events.extend(rng.integers(0, 3600, n).tolist())
                devices.extend(rng.integers(0, device_count, n).tolist())
                t, d = queued_times(events, devices)
                predicted = all_methods(t, d, model, thresholds)
                rows.append(
                    {
                        "people": size,
                        "delay_mean": tail,
                        "devices": device_count,
                        "methods": {m: match_onsets(truth, predicted[m], 10) for m in MAIN},
                    }
                )
    return {
        "note": "assumed exponential delays and1.5sdevicequeue; not real stop accuracy",
        "rows": rows,
    }


def run(source: Path, old: Path, out: Path, workers: int, prepared: Path | None = None) -> None:
    if out.exists():
        raise ValueError("choose a new result directory")
    out.mkdir(parents=True)
    windows = prepare(source, old) if prepared is None else json.loads(prepared.read_text())
    if len(windows) != 536 or sum(len(w["times"]) for w in windows) != 90021:
        raise ValueError("frozen536window/90021event selection changed")
    if sorted({w["date"] for w in windows}) != sorted(DEFAULT_DATES):
        raise ValueError("frozen date selection changed")
    write_json(out / "windows.json", windows)
    model, thresholds = fit(windows)
    write_json(out / "model.json", model.receipt())
    write_json(out / "thresholds.json", thresholds)
    print(json.dumps({"fitted": thresholds}), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(
            partial(evaluate, model=model, thresholds=thresholds), enumerate(windows)
        ):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(json.dumps({"evaluated": len(rows), "total": len(windows)}), flush=True)
    write_json(out / "rows.json", rows)
    write_json(out / "summary.json", summarize(rows))
    write_json(out / "synthetic.json", synthetic(model, thresholds))
    write_json(
        out / "manifest.json",
        {
            "complete": True,
            "schema_version": "boarding-multiscale.v1",
            "timezone": "Europe/Moscow",
            "dates": DEFAULT_DATES,
            "fit_end_exclusive": "2025-05-01",
            "prepared_cache_sha256": digest(prepared) if prepared else None,
            "seed": 20260926,
            "source_manifest_sha256": digest(source / "boarding-date-shards/manifest.json"),
            "selection_manifest_sha256": digest(old / "manifest.json"),
            "code": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            "files": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
            "door_or_stop_accuracy": "unverified",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--prior-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prepared", type=Path)
    a = p.parse_args()
    run(a.source, a.prior_run, a.out, a.workers, a.prepared)


if __name__ == "__main__":
    main()

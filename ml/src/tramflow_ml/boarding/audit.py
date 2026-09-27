"""Immutable row ledger and fail-closed source-mass audit, with resumable shards."""

import csv
import gzip
import hashlib
import io
import json
import platform
import resource
import sqlite3
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from .records import AuditConfig

COLUMNS = [
    "tran_no",
    "device_no",
    "tran_date_time",
    "validation_result",
    "ngpt_route",
    "garage_number",
    "bus_exit_no",
]
SENTINELS = frozenset({"", "0", "-1", "null", "NULL", "None", "nan"})


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


class HashReader(io.RawIOBase):
    def __init__(self, source: Any) -> None:
        self.source = source
        self.hash = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        chunk = self.source.read(len(buffer))
        self.hash.update(chunk)
        buffer[: len(chunk)] = chunk
        return len(chunk)


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA cache_size=-32768;
        CREATE TABLE IF NOT EXISTS chunks
          (name TEXT PRIMARY KEY, hash TEXT, rows INTEGER, stats TEXT);
        CREATE TABLE IF NOT EXISTS hours
          (route TEXT, hour TEXT, success INTEGER, rejected INTEGER,
           PRIMARY KEY(route,hour));
        CREATE TABLE IF NOT EXISTS relations
          (hour TEXT, device TEXT, vehicle TEXT, route TEXT, exit TEXT,
           first TEXT, last TEXT, success INTEGER, rejected INTEGER,
           PRIMARY KEY(hour,device,vehicle,route,exit));
    """)
    return db


def opaque(series: Any, field: str, salt: str, cache: dict[str, str]) -> Any:
    for raw in series.unique():
        key = field + ":" + str(raw)
        if key not in cache:
            cache[key] = (
                ""
                if raw in SENTINELS
                else hashlib.sha256((salt + ":" + key).encode()).hexdigest()[:24]
            )
    return series.map(lambda x: cache[field + ":" + str(x)])


def process(
    frame: Any,
    member: str,
    ordinal: int,
    source_hash: str,
    config: AuditConfig,
    cache: dict[str, str],
) -> tuple[Any, dict[str, Any], Any, Any]:
    timestamp = pd.to_datetime(frame.tran_date_time, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    route = frame.ngpt_route.str.extract(r"^(\d+) трамвай$", expand=False).fillna("")
    success = frame.validation_result.eq("1")
    canonical = frame.tran_date_time.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    invalid = timestamp.isna() | route.eq("") | ~canonical
    if bool((success & invalid).any()):
        raise ValueError("invalid successful timestamp/route: source mass cannot be bucketed")
    valid = ~invalid
    date = frame.tran_date_time.str[:10]
    scope = valid & date.ge(str(config.start)) & date.lt(str(config.end))
    full_key = ~frame[["device_no", "garage_number", "bus_exit_no"]].isin(SENTINELS).any(axis=1)
    prefix = hashlib.sha256((source_hash + ":" + member).encode()).hexdigest()[:24]
    ledger = pd.DataFrame(
        {
            "event_key": [f"{prefix}:{i}" for i in range(ordinal, ordinal + len(frame))],
            "event_at": frame.tran_date_time.where(valid, ""),
            "route": route,
            "success": success.astype(int),
            "in_range": scope.astype(int),
            "device_key": opaque(frame.device_no, "device", source_hash, cache),
            "vehicle_key": opaque(frame.garage_number, "vehicle", source_hash, cache),
            "exit_key": opaque(frame.bus_exit_no, "exit", source_hash, cache),
            "identity_status": "unverified",
            "assignment_status": "not_target",
            "reason": "rejected_or_outside_range",
            "stop_id": "",
        }
    )
    eligible = scope & success
    supported = route.isin(config.pattern_routes)
    ledger.loc[eligible & supported, ["assignment_status", "reason"]] = [
        "unassigned",
        "pattern_unverified_no_anchor",
    ]
    ledger.loc[eligible & ~supported, ["assignment_status", "reason"]] = [
        "out_of_scope",
        "pattern_missing",
    ]
    ledger.loc[eligible & ~full_key, "identity_status"] = "identity_missing"
    ledger.loc[eligible & supported & ~full_key, "reason"] = "identity_missing"
    stats: dict[str, Any] = {
        "raw_rows": len(frame),
        "successful_in_range": int(eligible.sum()),
        "rejected_in_range": int((scope & ~success).sum()),
        "outside_range_rows": int((valid & ~scope).sum()),
        "invalid_rejected_rows": int((invalid & ~success).sum()),
        "full_key_success": int((eligible & full_key).sum()),
        "minute_boundary_success": int((eligible & timestamp.dt.second.eq(0)).sum()),
        "assigned": 0,
        "ambiguous": 0,
        "unassigned": int((eligible & supported).sum()),
        "out_of_scope": int((eligible & ~supported).sum()),
        "missing_ids": {
            col: {
                "success": int((eligible & frame[col].isin(SENTINELS)).sum()),
                "rejected": int((scope & ~success & frame[col].isin(SENTINELS)).sum()),
            }
            for col in ["device_no", "garage_number", "bus_exit_no"]
        },
        "dates": sorted(date[scope].unique().tolist()),
    }
    a = ledger.loc[scope].copy()
    a["hour"] = a.event_at.str[:13]
    a["rejected"] = 1 - a.success
    hours = a.groupby(["route", "hour"], sort=True)[["success", "rejected"]].sum().reset_index()
    relations = (
        a.groupby(["hour", "device_key", "vehicle_key", "route", "exit_key"], sort=True)
        .agg(
            first=("event_at", "min"),
            last=("event_at", "max"),
            success=("success", "sum"),
            rejected=("rejected", "sum"),
        )
        .reset_index()
    )
    return ledger, stats, hours, relations


def audit(source: Path, out: Path, config: AuditConfig) -> dict[str, Any]:
    began = time.monotonic()
    if (out / "manifest.json").exists():
        raise ValueError("run is already complete; use a new directory")
    if out.exists() and any(out.iterdir()) and not (out / "checkpoint.json").exists():
        raise ValueError("nonempty output has no checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    signature = {
        "archive_sha256": source_hash,
        "config": config.model_dump(mode="json"),
        "implementation": {
            p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("records.py")]
        },
    }
    checkpoint = out / "checkpoint.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text()) != signature:
        raise ValueError("source/configuration changed; start a new run")
    write_json(checkpoint, signature)
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    db = connect(out / "audit.sqlite")
    cache: dict[str, str] = {}
    members = []
    with zipfile.ZipFile(source) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("duplicate archive member names")
        for member in ["train.csv", "test.csv"]:
            with z.open(member) as raw:
                reader = HashReader(raw)
                buffered = io.BufferedReader(reader)
                ordinal = 1
                for number, frame in enumerate(
                    pd.read_csv(
                        buffered,
                        sep=";",
                        dtype=str,
                        keep_default_na=False,
                        usecols=COLUMNS,
                        chunksize=config.chunk_rows,
                    )
                ):
                    name = f"{Path(member).stem}-{number:05d}.csv.gz"
                    prior = db.execute(
                        "SELECT hash,rows FROM chunks WHERE name=?", (name,)
                    ).fetchone()
                    if prior:
                        if prior != (digest(shards / name), len(frame)):
                            raise ValueError("checkpoint shard hash/length mismatch")
                        ordinal += len(frame)
                        continue
                    ledger, stats, hours, relations = process(
                        frame, member, ordinal, source_hash, config, cache
                    )
                    tmp = shards / (name + ".tmp")
                    with (
                        tmp.open("wb") as f,
                        gzip.GzipFile(
                            fileobj=f, mode="wb", filename="", mtime=0, compresslevel=1
                        ) as compressed,
                    ):
                        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                            ledger.to_csv(text, index=False, lineterminator="\n")
                    tmp.replace(shards / name)
                    with db:
                        db.executemany(
                            """INSERT INTO hours VALUES (?,?,?,?)
                            ON CONFLICT(route,hour) DO UPDATE SET
                            success=success+excluded.success,rejected=rejected+excluded.rejected""",
                            hours.itertuples(index=False, name=None),
                        )
                        db.executemany(
                            """INSERT INTO relations VALUES (?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(hour,device,vehicle,route,exit) DO UPDATE SET
                            first=min(first,excluded.first),last=max(last,excluded.last),
                            success=success+excluded.success,rejected=rejected+excluded.rejected""",
                            relations.itertuples(index=False, name=None),
                        )
                        db.execute(
                            "INSERT INTO chunks VALUES (?,?,?,?)",
                            (name, digest(shards / name), len(frame), json.dumps(stats)),
                        )
                    ordinal += len(frame)
                    if number % 20 == 0:
                        print(
                            json.dumps(
                                {
                                    "member": member,
                                    "rows": ordinal - 1,
                                    "seconds": round(time.monotonic() - began, 1),
                                }
                            ),
                            flush=True,
                        )
                members.append(
                    {
                        "name": member,
                        "sha256": reader.hash.hexdigest(),
                        "rows": ordinal - 1,
                        "bytes": z.getinfo(member).file_size,
                    }
                )
        labels: dict[tuple[str, str], int] = {}
        for member in ["labels/labels_day_train.csv", "labels/labels_day_test.csv"]:
            if member not in z.namelist():
                continue
            for row in csv.DictReader(
                io.StringIO(z.read(member).decode("utf-8-sig")), delimiter=";"
            ):
                if not str(config.start) <= row["date"] < str(config.end):
                    continue
                key = row["route"], row["date"] + f" {int(row['hour']):02}"
                if key in labels:
                    raise ValueError("duplicate label key")
                labels[key] = int(row["boardings"])
        report = finish(db, out, labels, signature, members, began)
    db.close()
    return report


def finish(
    db: sqlite3.Connection,
    out: Path,
    labels: dict[tuple[str, str], int],
    signature: dict[str, Any],
    members: list[dict[str, Any]],
    began: float,
) -> dict[str, Any]:
    chunk_rows = db.execute("SELECT name,hash,rows,stats FROM chunks ORDER BY name").fetchall()
    stats = [json.loads(row[3]) for row in chunk_rows]
    sums = (
        {key: sum(s[key] for s in stats) for key in stats[0] if isinstance(stats[0][key], int)}
        if stats
        else {}
    )
    sums.setdefault("successful_in_range", 0)
    counts = {
        (route, hour): n for route, hour, n in db.execute("SELECT route,hour,success FROM hours")
    }
    if (
        sums["successful_in_range"] != sum(counts.values())
        or sums.get("raw_rows", 0) != sum(r[2] for r in chunk_rows)
        or sums.get("raw_rows", 0) != sum(m["rows"] for m in members)
        or sums["successful_in_range"]
        != sum(sums.get(k, 0) for k in ["assigned", "ambiguous", "unassigned", "out_of_scope"])
    ):
        raise ValueError("G0 checkpoint/source/category mass mismatch")
    mismatches = sum(counts.get(k, 0) != labels.get(k, 0) for k in counts.keys() | labels.keys())
    if labels and mismatches:
        raise ValueError(f"G0: {mismatches} label mismatches")
    with (out / "mass_ledger.csv").open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "route",
                "event_hour",
                "source_success",
                "assigned",
                "ambiguous",
                "unassigned",
                "out_of_scope",
                "stop_id",
                "unresolved_weight",
            ]
        )
        for (route, hour), n in sorted(counts.items()):
            supported = route in signature["config"]["pattern_routes"]
            w.writerow([route, hour, n, 0, 0, n if supported else 0, 0 if supported else n, "", 1])
    conflict_device = db.execute("""SELECT count(*) FROM (SELECT hour,device FROM relations
        WHERE device<>'' AND vehicle<>'' GROUP BY hour,device
        HAVING count(DISTINCT vehicle)>1)""").fetchone()[0]
    conflict_vehicle = db.execute("""SELECT count(*) FROM (SELECT hour,vehicle FROM relations
        WHERE vehicle<>'' GROUP BY hour,vehicle
        HAVING count(DISTINCT route || ':' || exit)>1)""").fetchone()[0]
    device_migrations = db.execute("""SELECT count(*) FROM (SELECT device FROM relations
        WHERE device<>'' AND vehicle<>'' GROUP BY device
        HAVING count(DISTINCT vehicle)>1)""").fetchone()[0]
    missing = {
        col: {
            kind: sum(s["missing_ids"][col][kind] for s in stats)
            for kind in ["success", "rejected"]
        }
        for col in ["device_no", "garage_number", "bus_exit_no"]
    }
    write_json(
        out / "identity_quality.json",
        {
            "schema_version": "boarding-identity.v1",
            "pooling": "device_only",
            "identity_certified": False,
            "conflicting_device_hours": conflict_device,
            "conflicting_vehicle_hours": conflict_vehicle,
            "devices_with_multiple_vehicles": device_migrations,
            "conflict_resolution": "potential hour overlap; never majority assignment",
            "missing_ids": missing,
            "full_key_success": sums.get("full_key_success", 0),
            "denominator": sums["successful_in_range"],
            "intervals": "audit.sqlite relations: first/last per hour/device/vehicle/route/exit",
        },
    )
    write_json(
        out / "time_quality.json",
        {
            "schema_version": "boarding-time.v1",
            "resolution_seconds": 1,
            "minute_boundary_success": sums.get("minute_boundary_success", 0),
            "clock_correction_applied": False,
            "availability": "unknown",
            "detailed_ties_gaps_batching": (
                "pilot evaluation; global simultaneity is not boarding proof"
            ),
        },
    )
    write_json(
        out / "inventory.json",
        {
            **sums,
            "members": members,
            "label_keys": len(labels),
            "label_mismatches": mismatches if labels else None,
            "routes": sorted({k[0] for k in counts}),
            "source_range": signature["config"],
            "target": "validation_result == 1; no deduplication",
        },
    )
    result = {
        "source_success": sums["successful_in_range"],
        "label_mismatches": mismatches if labels else None,
        "G0": "passed",
        "G1": "device_only",
        "G3": "unverified",
        "real_stop_accuracy": "unverified",
        "stop_training_eligible": False,
    }
    write_json(out / "evaluation.json", result)
    files = {
        p.name: digest(p)
        for p in out.iterdir()
        if p.suffix in {".csv", ".json"} and p.name != "checkpoint.json"
    }
    manifest = {
        **signature,
        "schema_version": "boarding-run.v1",
        "inference_method": "B0-unresolved",
        "timezone": "Europe/Moscow",
        "feature_version": "boarding-device-bursts.v1",
        "target": "validation_count",
        "time_basis": "original_payment_time",
        "available_at": None,
        "mode": "retrospective_offline",
        "calibrated": False,
        "members": members,
        "shards": [{"name": r[0], "sha256": r[1], "rows": r[2]} for r in chunk_rows],
        "files": files,
        "source_success": sums["successful_in_range"],
        "code_sha": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "wall_seconds": time.monotonic() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
        },
        "leakage_policy": (
            "event_at < origin; availability unknown; offline-only; no future fitting"
        ),
        "stop_training_eligible": False,
        "complete": True,
    }
    write_json(out / "manifest.json", manifest)
    return result


def verify_run(out: Path) -> dict[str, Any]:
    if not (out / "manifest.json").exists():
        raise ValueError("incomplete run")
    m = json.loads((out / "manifest.json").read_text())
    if m.get("schema_version") != "boarding-run.v1" or m.get("complete") is not True:
        raise ValueError("invalid run manifest")
    for shard in m["shards"]:
        if digest(out / "shards" / shard["name"]) != shard["sha256"]:
            raise ValueError("shard hash mismatch")
    for name, expected in m["files"].items():
        if digest(out / name) != expected:
            raise ValueError("artifact hash mismatch")
    count = 0
    with (out / "mass_ledger.csv").open() as f:
        for row in csv.DictReader(f):
            source = int(row["source_success"])
            parts = [int(row[k]) for k in ["assigned", "ambiguous", "unassigned", "out_of_scope"]]
            if min(parts) < 0 or sum(parts) != source:
                raise ValueError("G0 mass mismatch")
            count += source
    if count != m["source_success"]:
        raise ValueError("G0 total mismatch")
    return json.loads((out / "evaluation.json").read_text())  # type: ignore[no-any-return]

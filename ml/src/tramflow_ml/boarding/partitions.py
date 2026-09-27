"""Hash-bound date shards for bounded-memory real-payment inference."""

import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, verify_run, write_json


def write_frame(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with (
        tmp.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as gz,
    ):
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            frame.to_csv(text, index=False, lineterminator="\n")
    tmp.replace(path)


def prepare(source: Path, out: Path) -> dict[str, Any]:
    verify_run(source)
    manifest = json.loads((source / "manifest.json").read_text())
    signature = {
        "schema_version": "boarding-date-shards.v1",
        "source_manifest_sha256": digest(source / "manifest.json"),
        "implementation_sha256": digest(Path(__file__)),
    }
    if (out / "manifest.json").exists():
        raise ValueError("partition run complete; use new output")
    if out.exists() and any(out.iterdir()) and not (out / "checkpoint.json").exists():
        raise ValueError("unrecognized partial partition run")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint.json").exists():
        if json.loads((out / "checkpoint.json").read_text()) != signature:
            raise ValueError("source/partition implementation changed")
    write_json(out / "checkpoint.json", signature)
    mass: Counter[tuple[str, str]] = Counter()
    parts = []
    total_events = 0
    for shard in manifest["shards"]:
        receipt_path = out / "receipts" / (shard["name"] + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            for part in receipt["parts"]:
                if digest(out / part["path"]) != part["sha256"]:
                    raise ValueError("date shard checksum changed")
        else:
            frame = pd.read_csv(source / "shards" / shard["name"], dtype=str, keep_default_na=False)
            frame = frame.loc[frame.in_range.eq("1")].copy()
            frame["date"] = frame.event_at.str[:10]
            frame["hour"] = frame.event_at.str[:13]
            frame["target"] = frame.success.astype(int)
            counts = frame.groupby(["route", "hour"]).target.sum()
            local_parts = []
            for day, block in frame.groupby("date", sort=True):
                relative = f"days/{day}/{shard['name']}"
                write_frame(out / relative, block.drop(columns=["date", "hour", "target"]))
                local_parts.append(
                    {
                        "date": day,
                        "path": relative,
                        "sha256": digest(out / relative),
                        "events": len(block),
                        "successful": int(block.target.sum()),
                    }
                )
            receipt = {
                "parts": local_parts,
                "mass": [[r, h, int(n)] for (r, h), n in counts.items()],
                "source_shard_sha256": shard["sha256"],
            }
            receipt_path.parent.mkdir(exist_ok=True)
            write_json(receipt_path, receipt)
        if receipt["source_shard_sha256"] != shard["sha256"]:
            raise ValueError("receipt source changed")
        parts.extend(receipt["parts"])
        total_events += sum(p["events"] for p in receipt["parts"])
        mass.update({(r, h): n for r, h, n in receipt["mass"]})
    with (source / "mass_ledger.csv").open() as f:
        expected = {
            (r["route"], r["event_hour"]): int(r["source_success"]) for r in csv.DictReader(f)
        }
    if any(mass[k] != expected.get(k, 0) for k in mass.keys() | expected.keys()):
        raise ValueError("source-hour conservation failed during partitioning")
    result = {
        **signature,
        "complete": True,
        "timezone": "Europe/Moscow",
        "target": "validation_count",
        "unit": "event_count",
        "successful": sum(mass.values()),
        "events": total_events,
        "date_range": manifest["config"],
        "parts": parts,
    }
    write_json(out / "manifest.json", result)
    return result


def read_day(source: Path, day: str, routes: tuple[str, ...] = ()) -> Any:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema_version") != "boarding-date-shards.v1" or not manifest.get("complete"):
        raise ValueError("incomplete date shards")
    blocks = []
    for part in manifest["parts"]:
        if part["date"] != day:
            continue
        path = source / part["path"]
        if digest(path) != part["sha256"]:
            raise ValueError("date shard checksum mismatch")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if routes:
            frame = frame.loc[frame.route.isin(routes)]
        blocks.append(frame)
    if not blocks:
        return pd.DataFrame()
    result = pd.concat(blocks, ignore_index=True)
    if result.event_key.duplicated().any():
        raise ValueError("duplicate source event")
    return result.sort_values(["event_at", "event_key"]).reset_index(drop=True)

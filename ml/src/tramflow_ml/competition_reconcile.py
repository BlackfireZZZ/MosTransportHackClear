"""Prove that missing organizer label keys have no successful raw validation."""

import csv
import io
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from tramflow_ml.competition import LABEL_ROUTES, load_labels

ROUTE_PATTERN = re.compile(r"^\s*(\d+)\b")
RAW_COLUMNS = ("tran_date_time", "validation_result", "ngpt_route")
START = "2025-01-01"
END = "2025-10-31"


def reconcile_archive(archive: Path) -> dict[str, object]:
    """Compare every successful Jan–Oct raw row with the positive-only label export."""
    labels = load_labels(archive)
    if labels.duplicated(["route", "date", "hour"]).any():
        raise ValueError("duplicate supplied label key")
    expected = {
        (int(row.route), row.date.isoformat(), int(row.hour)): int(row.boardings)
        for row in labels.itertuples(index=False)
    }
    raw: Counter[tuple[int, str, int]] = Counter()
    files: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(archive) as source:
        for name in ("train.csv", "test.csv"):
            rows = 0
            successful = 0
            outside = 0
            with source.open(name) as binary:
                reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8-sig"), delimiter=";")
                header = next(reader)
                if any(column not in header for column in RAW_COLUMNS):
                    raise ValueError(f"{name} missing required raw columns")
                time_idx, status_idx, route_idx = (header.index(column) for column in RAW_COLUMNS)
                for row in reader:
                    rows += 1
                    if len(row) != len(header):
                        raise ValueError(f"{name}: malformed row {rows}")
                    timestamp = row[time_idx]
                    if row[status_idx] != "1":
                        continue
                    try:
                        instant = datetime.fromisoformat(timestamp)
                    except ValueError as error:
                        raise ValueError(
                            f"{name}: invalid event timestamp at row {rows}"
                        ) from error
                    day = instant.date().isoformat()
                    if not START <= day <= END:
                        outside += 1
                        continue
                    match = ROUTE_PATTERN.match(row[route_idx])
                    if match is None or int(match.group(1)) not in LABEL_ROUTES:
                        raise ValueError(f"{name}: unknown route at row {rows}")
                    raw[(int(match.group(1)), day, instant.hour)] += 1
                    successful += 1
            files[name] = {
                "rows": rows,
                "successful_in_period": successful,
                "successful_outside_period": outside,
            }
    if raw != expected:
        missing_raw = expected.keys() - raw.keys()
        missing_labels = raw.keys() - expected.keys()
        changed = sum(raw[key] != expected[key] for key in raw.keys() & expected.keys())
        raise ValueError(
            f"raw/labels mismatch: {len(missing_raw)} labels-only, "
            f"{len(missing_labels)} raw-only, {changed} count mismatches"
        )
    return {
        "schema": "route-hour-reconciliation.v1",
        "period": [START, END],
        "files": files,
        "label_keys": len(expected),
        "raw_keys": len(raw),
        "successful_boardings": sum(raw.values()),
        "missing_keys_are_zero_in_supplied_raw": True,
    }

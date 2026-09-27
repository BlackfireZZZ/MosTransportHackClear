"""Organizer 2025 labels count successful rows, not certified service coverage."""

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
START = date(2025, 1, 1)
END = date(2025, 11, 1)
FEATURE_VERSION = "route-calendar-profiles.v1"
COVERAGE_POLICY = "zero-successful-rows-in-extract.v1"


@dataclass(frozen=True)
class History:
    values: FloatArray
    source_hash: str

    def __post_init__(self) -> None:
        if self.values.shape != (304, len(ROUTES), 24):
            raise ValueError("history must cover January through October 2025")
        if not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("history must contain finite nonnegative counts")


def read_labels(archive: Path) -> History:
    values = np.zeros((304, len(ROUTES), 24), dtype=np.float64)
    seen: set[tuple[int, int, int]] = set()
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as source:
        for part, first, last in (
            ("train", START, date(2025, 9, 1)),
            ("test", date(2025, 9, 1), END),
        ):
            name = f"labels/labels_day_{part}.csv"
            content = source.read(name)
            digest.update(name.encode() + b"\0" + content)
            reader = csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter=";")
            if reader.fieldnames != ["route", "date", "hour", "boardings"]:
                raise ValueError("unexpected organizer label columns")
            for row in reader:
                day = date.fromisoformat(row["date"])
                route, hour, target = (int(row[k]) for k in ("route", "hour", "boardings"))
                if not first <= day < last or route not in ROUTES or not 0 <= hour < 24:
                    raise ValueError("label key outside declared source scope")
                if target < 0:
                    raise ValueError("negative target")
                key = ((day - START).days, ROUTES.index(route), hour)
                if key in seen:
                    raise ValueError("duplicate route/date/hour")
                seen.add(key)
                values[key] = target
    if not seen:
        raise ValueError("empty label dataset")
    values.setflags(write=False)
    return History(values, digest.hexdigest())


def day_index(value: date) -> int:
    return (value - START).days


def civil_date(index: int) -> date:
    return START + timedelta(days=int(index))

import csv
import hashlib
from datetime import date, timedelta
from pathlib import Path

import pytest

from app.infrastructure.forecast_publication import load_scored_csv


def write_grid(path: Path, *, omit_last: bool = False, duplicate: bool = False) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["route", "date", "hour", "prediction"])
        rows = []
        for route in (1, 5, 7, 11, 12, 17, 25, 26, 28, 50):
            for day in range(61):
                stamp = date(2025, 11, 1) + timedelta(days=day)
                for hour in range(24):
                    rows.append((route, stamp.isoformat(), hour, route + hour))
        if omit_last:
            rows.pop()
        if duplicate:
            rows[-1] = rows[0]
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_grid_and_daily_totals(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    digest = write_grid(path)
    data = load_scored_csv(path, digest)
    assert len(data) == 14640
    assert data[(5, date(2025, 11, 1), 0)] == 5
    assert sum(data[(5, date(2025, 11, 1), hour)] for hour in range(24)) == 396


@pytest.mark.parametrize("change", ["omit_last", "duplicate"])
def test_incomplete_or_duplicate_grid_is_rejected(tmp_path: Path, change: str) -> None:
    path = tmp_path / "forecast.csv"
    digest = write_grid(path, **{change: True})
    with pytest.raises(ValueError):
        load_scored_csv(path, digest)


def test_wrong_checksum_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    write_grid(path)
    with pytest.raises(ValueError, match="checksum"):
        load_scored_csv(path, "0" * 64)


def test_oversized_csv_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="size limit"):
        load_scored_csv(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("bad_value", ["nan", "-1", "1.5", "1000000000"])
def test_invalid_prediction_is_rejected(tmp_path: Path, bad_value: str) -> None:
    path = tmp_path / "forecast.csv"
    write_grid(path)
    payload = path.read_text(encoding="utf-8").replace(
        "1;2025-11-01;0;1", f"1;2025-11-01;0;{bad_value}", 1
    )
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="prediction"):
        load_scored_csv(path, digest)

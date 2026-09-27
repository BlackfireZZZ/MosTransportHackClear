"""Exact stop forecasts inside their grid, explicitly qualitative repetition outside it."""

from collections import defaultdict
from datetime import date, timedelta
from typing import Any


class DirectStopForecast:
    def __init__(self, data: dict[str, Any], route: str):
        self.start = date.fromisoformat(data["start_date"])
        self.days = data["days"]
        self.identities = [row for row in data["identities"] if row["route"] == route]
        self.profiles: list[dict[tuple[int, int], float]] = []
        for row in self.identities:
            samples: dict[tuple[int, int], list[float]] = defaultdict(list)
            for index, value in enumerate(row["values"]):
                day = self.start + timedelta(days=index // 24)
                samples[(day.weekday(), index % 24)].append(value)
            self.profiles.append(
                {key: sum(values) / len(values) for key, values in samples.items()}
            )

    def cells(self, day: date, hour: int) -> list[tuple[str, str, float]]:
        offset = (day - self.start).days
        return [
            (
                row["stop_id"],
                row["direction"],
                row["values"][offset * 24 + hour]
                if 0 <= offset < self.days
                else self.profiles[index][(day.weekday(), hour)],
            )
            for index, row in enumerate(self.identities)
        ]

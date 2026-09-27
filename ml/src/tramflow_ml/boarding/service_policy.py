"""Source-bound calendar exclusions; restoration never certifies operated trips."""

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .audit import digest

MOSCOW = ZoneInfo("Europe/Moscow")
RECURRENCES = {"daily", "weekends", "continuous", "restoration", "night_after_23_30",
               "night_after_23_00", "night_after_00_20"}
RESTORATIONS = {
    "weekend_medvedkovo_closure_end_unknown": ("17", "telegram20825"),
    "weekend_entuziastov_diversion_end_unknown": ("12", "7177"),
}
INFORMATIONAL_WARNINGS = {
    "incomplete_timetable_coverage_clock_evidence_disabled",
    "short_turn_geometry_from_ordered_base_and_official_endpoints",
}


def _instant(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        if len(value) != 10:
            raise ValueError("service timestamps require timezone")
        return result.replace(tzinfo=MOSCOW)
    return result.astimezone(MOSCOW)


@dataclass(frozen=True)
class PolicyDecision:
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ServicePolicy:
    registry_sha256: str
    notices: tuple[dict[str, Any], ...]

    @property
    def source_urls(self) -> list[str]:
        return sorted({n["url"] for n in self.notices})

    def evaluate(
        self, route: str, hour: datetime, *, engine_applicable: bool,
        warnings: tuple[str, ...], soft: bool,
    ) -> PolicyDecision:
        """Whole original hours are excluded on any modeled notice-window overlap."""
        if hour.tzinfo is None or hour.minute or hour.second or hour.microsecond:
            raise ValueError("policy requires timezone-aware whole original hours")
        hour = hour.astimezone(MOSCOW)
        reasons = []
        eligible = engine_applicable
        if not engine_applicable:
            unresolved = []
            restored = []
            for warning in warnings:
                if warning in INFORMATIONAL_WARNINGS:
                    reasons.append("informational_warning:" + warning)
                    continue
                match = RESTORATIONS.get(warning)
                notice = next((n for n in self.notices if match and n["id"] == match[1]), None)
                if (soft and match and route == match[0] and notice is not None
                        and route in notice["routes"] and notice["recurrence"] == "restoration"
                        and hour >= _instant(notice["effective_start"])):
                    restored.append("restored_warning:" + notice["id"])
                else:
                    unresolved.append(warning)
            eligible = bool(restored) and not unresolved
            reasons.extend(restored)
            if not eligible:
                reasons.append("engine_pattern_unresolved")
        for notice in self.notices:
            if route not in notice["routes"]:
                continue
            recurrence = notice["recurrence"]
            if recurrence == "restoration":
                if notice["id"] in {"126404", "126575"}:
                    before = _instant("2025-01-01") <= hour < _instant(notice["effective_start"])
                    weekday = notice["id"] != "126575" or hour.weekday() < 5
                    if before and hour.hour < 5 and weekday:
                        eligible = False
                        reasons.append("unknown_start_precaution:" + notice["id"])
                continue
            if (notice["id"] == "7319" and route == "7"
                    and "short_turn_geometry_from_ordered_base_and_official_endpoints" in warnings):
                continue
            if _overlaps(notice, hour):
                eligible = False
                reasons.append("notice_exclusion:" + notice["id"])
                if notice["id"] == "7223":
                    reasons.append("approximate_end_extended_to_08")
        return PolicyDecision(eligible, tuple(sorted(set(reasons))))


def _overlaps(notice: dict[str, Any], hour: datetime) -> bool:
    start = _instant(notice["effective_start"])
    end = _instant(notice["effective_end_exclusive"]) if notice["effective_end_exclusive"] else None
    recurrence = notice["recurrence"]
    if recurrence.startswith("night_after_"):
        if recurrence == "night_after_00_20":
            if hour.hour >= 5:
                return False
            anchor = datetime.combine(hour.date(), time(0, 20), MOSCOW)
        else:
            if 5 <= hour.hour < 23:
                return False
            day = hour.date() - timedelta(days=int(hour.hour < 5))
            anchor = datetime.combine(day, time(23, 30 if recurrence.endswith("30") else 0), MOSCOW)
        return anchor >= start and (end is None or anchor < end)
    if recurrence == "weekends" and hour.weekday() < 5:
        return False
    if notice["id"] == "7223" and end is not None:
        end += timedelta(hours=1)
    return hour + timedelta(hours=1) > start and (end is None or hour < end)


def load_service_policy(path: Path, *, source_root: Path | None = None) -> ServicePolicy:
    """Relative snapshot paths resolve from repository cwd unless source_root is supplied."""
    registry = json.loads(path.read_text())
    if (registry.get("schema_version") != "boarding-service-notices-v1"
            or registry.get("timezone") != "Europe/Moscow"
            or registry.get("complete_historical_calendar") is not False):
        raise ValueError("unsupported service notice registry")
    notices = registry.get("notices", [])
    if not notices or len({n["id"] for n in notices}) != len(notices):
        raise ValueError("nonempty unique service notices required")
    for notice in notices:
        if notice["recurrence"] not in RECURRENCES:
            raise ValueError("unsupported service recurrence")
        if not notice["routes"] or any(not isinstance(r, str) or not r for r in notice["routes"]):
            raise ValueError("service notice routes required")
        url = urlsplit(notice["url"])
        if url.scheme != "https" or url.hostname not in {"transport.mos.ru", "www.mosmetro.ru",
                                                         "mosmetro.ru", "t.me"}:
            raise ValueError("official service source URL required")
        if url.hostname == "t.me" and not url.path.startswith("/DtOperativno/"):
            raise ValueError("official transport channel required")
        start = _instant(notice["effective_start"])
        end = notice.get("effective_end_exclusive")
        if end is not None and _instant(end) <= start:
            raise ValueError("service end must follow start")
        if end is None and notice["recurrence"] != "restoration" and not notice["end_unknown"]:
            raise ValueError("unbounded service exclusion must declare unknown end")
        snapshot = Path(notice["source_path"])
        if not snapshot.is_absolute():
            snapshot = (source_root or Path.cwd()) / snapshot
        if not snapshot.is_file() or digest(snapshot) != notice["source_sha256"]:
            raise ValueError("service notice snapshot missing or changed: " + notice["id"])
    return ServicePolicy(digest(path), tuple(notices))

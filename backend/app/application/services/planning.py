"""Scenario counts conserve the selected route total including unknown spatial mass."""

import hashlib
import json
from calendar import monthrange
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from app.application.services.stop_planning import DirectStopForecast


class PlanningUnavailable(RuntimeError):
    """A required versioned planning artifact cannot be safely served."""


class PlanningRepository(Protocol):
    def load(self) -> dict[str, Any]: ...

    def experimental(self) -> dict[str, Any]: ...

    def approved_sources(self) -> dict[str, Any]: ...

    def stop_model(self) -> dict[str, Any]: ...


def month_after(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


class PlanningService:
    def __init__(self, repository: PlanningRepository):
        self.repository = repository

    def forecast(self, request: dict[str, Any]) -> dict[str, Any]:
        data = self.repository.load()
        route, start, horizon = (
            request["route"],
            date.fromisoformat(str(request["start_date"])),
            request["horizon"],
        )
        if not date(2025, 11, 1) <= start <= date(2025, 12, 31):
            raise ValueError("Start date must be in November–December 2025")
        if horizon == "year" and start.day != 1:
            raise ValueError("Annual scenario starts on the first day of a month")
        routes = sorted({row["route"] for row in data["route_rows"]}, key=int)
        if route not in routes:
            raise ValueError("Unknown route number")
        rows = [row for row in data["route_rows"] if row["route"] == route]
        counts = {
            (date.fromisoformat(row["date"]), int(row["hour"])): float(row["prediction"])
            for row in rows
        }
        direct = None
        experimental_evaluation = None
        provenance = dict(data["provenance"])
        model_version = provenance["model_version"]
        generated_at = provenance["generated_at"]
        has_factor_override = request["forecast_mode"] == "approved" and any(
            control["enabled"] for control in request["factors"].values()
        )
        if (request["forecast_mode"] == "approved"
                and any(request["source_enabled"].values())
                and not has_factor_override):
            variants = self.repository.approved_sources()
            selected = {
                name: variants["weights"][name]
                for name, enabled in request["source_enabled"].items() if enabled
            }
            base_weight = 1 - sum(selected.values())
            counts = {
                (date.fromisoformat(day), hour): round(
                    base_weight * counts[(date.fromisoformat(day), hour)]
                    + sum(weight * variants["branches"][name][index]
                          for name, weight in selected.items())
                )
                for index, (number, day, hour) in enumerate(variants["keys"])
                if str(number) == route
            }
            provenance["approved_reference_model_version"] = provenance.pop("model_version")
            provenance["approved_reference_csv_sha256"] = provenance.pop("csv_sha256")
            provenance["approved_reference_generated_at"] = provenance.pop("generated_at")
            provenance["source_variant_sha256"] = variants["artifact_sha256"]
            provenance["source_mix_weights"] = json.dumps(
                {"incumbent": base_weight, **selected}, sort_keys=True,
            )
            provenance["source_availability"] = json.dumps(
                {name: variants["availability"][name] for name in selected},
                ensure_ascii=False, sort_keys=True,
            )
            generated_at = variants["generated_at"]
            model_version = "approved-source-variants.v1:unsubmitted"
            provenance["model_version"] = model_version
            provenance["generated_at"] = generated_at
        if request["forecast_mode"] == "external_experiment":
            experiment = self.repository.experimental()
            branch_names = ["base"] + [
                ("news" if name == "events" else name)
                for name, enabled in request["source_enabled"].items()
                if enabled
            ]
            counts = {
                (date.fromisoformat(day), hour): round(
                    sum(experiment["branches"][branch][i] for branch in branch_names)
                    / len(branch_names)
                )
                for i, (number, day, hour) in enumerate(experiment["keys"])
                if str(number) == route
            }
            experimental_evaluation = json.dumps(experiment["validation"], ensure_ascii=False)
            generated_at = experiment["generated_at"]
            provenance["approved_reference_model_version"] = provenance.pop("model_version")
            provenance["approved_reference_csv_sha256"] = provenance.pop("csv_sha256")
            provenance["experimental_generated_at"] = generated_at
            provenance["experimental_cutoff"] = experiment["cutoff"]
            provenance["experimental_sha256"] = experiment["artifact_sha256"]
            provenance["experimental_branches"] = ",".join(branch_names)
            model_version = "external-forecast-variants.v1:unpromoted"
        if request["forecast_mode"] == "stop_model":
            independent = self.repository.stop_model()
            direct = DirectStopForecast(independent, route)
            provenance = dict(independent["provenance"])
            model_version = provenance["model_version"]
            generated_at = provenance["generated_at"]
        raw_weights = {
            name: control["multiplier"]
            for name, control in request["factors"].items()
            if control["enabled"] and control["multiplier"] > 0
        }
        weight_total = 1 + sum(raw_weights.values())
        weights = {"base": 1 / weight_total} | {
            name: weight / weight_total for name, weight in raw_weights.items()
        }
        branch_counts: dict[str, dict[tuple[date, int], float]] = {}
        if raw_weights:
            variants = (
                self.repository.approved_sources()
                if request["forecast_mode"] == "approved"
                else self.repository.experimental()
            )
            for name in raw_weights:
                branch = (
                    name
                    if request["forecast_mode"] == "approved"
                    else "news" if name == "events" else name
                )
                branch_counts[name] = {
                    (date.fromisoformat(day), hour): float(variants["branches"][branch][i])
                    for i, (number, day, hour) in enumerate(variants["keys"])
                    if str(number) == route
                }
            provenance["scenario_source_sha256"] = variants["artifact_sha256"]
            provenance["scenario_source_version"] = variants["schema"]
            provenance["scenario_source_generated_at"] = variants["generated_at"]
            if request["forecast_mode"] == "approved":
                provenance["source_availability"] = json.dumps(
                    {name: variants["availability"][name] for name in raw_weights},
                    ensure_ascii=False, sort_keys=True,
                )
        provenance["normalized_weights"] = json.dumps(weights, sort_keys=True)
        profiles: dict[tuple[int, int], list[float]] = defaultdict(list)
        for (day, hour), value in counts.items():
            profiles[(day.weekday(), hour)].append(value)
        branch_profiles: dict[str, dict[tuple[int, int], list[float]]] = {}
        for name, values in branch_counts.items():
            profile: dict[tuple[int, int], list[float]] = defaultdict(list)
            for (day, hour), value in values.items():
                profile[(day.weekday(), hour)].append(value)
            branch_profiles[name] = profile
        stops = [
            {
                "stop_id": row["stop_id"],
                "name": row["name"],
                "direction": row["direction"],
                "latitude": float(row["lat"]) if row["lat"] else None,
                "longitude": float(row["lon"]) if row["lon"] else None,
            }
            for row in data["stops"]
            if row["route"] == route
        ]
        catalog = {(row["stop_id"], row["direction"]): row for row in stops}
        selected_stop, selected_direction = request.get("stop_id"), request.get("direction")
        if selected_stop is not None and not any(s["stop_id"] == selected_stop for s in stops):
            raise ValueError("Unknown GTFS stop for route")
        if selected_direction is not None and not any(
            s["direction"] == selected_direction for s in stops
        ):
            raise ValueError("Unknown estimated direction for route")
        if selected_stop is not None and selected_direction is not None:
            if (selected_stop, selected_direction) not in catalog:
                raise ValueError("GTFS stop does not belong to the selected direction")
        if horizon == "day":
            spans = [
                (
                    datetime.combine(start, datetime.min.time()) + timedelta(hours=h),
                    datetime.combine(start, datetime.min.time()) + timedelta(hours=h + 1),
                )
                for h in range(24)
            ]
        elif horizon == "month":
            spans = [
                (
                    datetime.combine(date(start.year, start.month, d), datetime.min.time()),
                    datetime.combine(date(start.year, start.month, d), datetime.min.time())
                    + timedelta(days=1),
                )
                for d in range(start.day, monthrange(start.year, start.month)[1] + 1)
            ]
        else:
            spans = []
            current = start
            for _ in range(12):
                following = month_after(current)
                spans.append(
                    (
                        datetime.combine(current, datetime.min.time()),
                        datetime.combine(following, datetime.min.time()),
                    )
                )
                current = following
        points = []
        for begin, end in spans:
            known: dict[tuple[str, str], float] = defaultdict(float)
            scenario_known: dict[tuple[str, str], float] = defaultdict(float)
            unknown = 0.0
            scenario_unknown = 0.0
            moment = begin
            while moment < end:
                hourly_known: dict[tuple[str, str], float] = defaultdict(float)
                hourly_unknown = 0.0
                if direct is not None:
                    for stop, direction, value in direct.cells(moment.date(), moment.hour):
                        if (stop, direction) in catalog:
                            hourly_known[(stop, direction)] += value
                        else:
                            hourly_unknown += value
                else:
                    key = (moment.date(), moment.hour)
                    samples = profiles[(moment.weekday(), moment.hour)]
                    value = counts[key] if key in counts else sum(samples) / len(samples)
                    cells = data["shares"].get(f"{route}|{moment.weekday()}|{moment.hour}", [])
                    allocated = 0.0
                    for stop, direction, share in cells:
                        if (stop, direction) in catalog:
                            hourly_known[(stop, direction)] += value * share
                            allocated += value * share
                    hourly_unknown = max(0.0, value - allocated)
                hourly_baseline = sum(hourly_known.values()) + hourly_unknown
                hourly_scenario = hourly_baseline * weights["base"]
                for name, branch in branch_counts.items():
                    sample_key = (moment.date(), moment.hour)
                    samples = branch_profiles[name][(moment.weekday(), moment.hour)]
                    prediction = (
                        branch[sample_key]
                        if sample_key in branch
                        else sum(samples) / len(samples)
                    )
                    hourly_scenario += weights[name] * prediction
                ratio = hourly_scenario / hourly_baseline if hourly_baseline else 0.0
                for cell_key, value in hourly_known.items():
                    known[cell_key] += value
                    scenario_known[cell_key] += value * ratio
                unknown += hourly_unknown
                scenario_unknown += (
                    hourly_unknown * ratio if hourly_baseline else hourly_scenario
                )
                moment += timedelta(hours=1)
            spatial = [
                {**catalog[key], "baseline": value, "scenario": scenario_known[key]}
                for key, value in sorted(known.items())
                if (selected_stop is None or key[0] == selected_stop)
                and (selected_direction is None or key[1] == selected_direction)
            ]
            route_unknown = unknown
            route_scenario_unknown = scenario_unknown
            route_baseline = sum(known.values()) + unknown
            route_scenario = sum(scenario_known.values()) + scenario_unknown
            if selected_stop is not None or selected_direction is not None:
                unknown = 0.0
                scenario_unknown = 0.0
            baseline = sum(row["baseline"] for row in spatial) + unknown
            scenario = sum(row["scenario"] for row in spatial) + scenario_unknown
            points.append(
                {
                    "timestamp": begin.replace(tzinfo=ZoneInfo("Europe/Moscow")).isoformat(),
                    "bucket_end": end.replace(tzinfo=ZoneInfo("Europe/Moscow")).isoformat(),
                    "basis": (
                        "competition_period"
                        if begin.date() < date(2026, 1, 1)
                        else "scenario_projection"
                    ),
                    "baseline": baseline,
                    "scenario": scenario,
                    "unallocated_baseline": unknown,
                    "unallocated_scenario": scenario_unknown,
                    "spatial": spatial,
                    "route_baseline": route_baseline,
                    "route_scenario": route_scenario,
                    "route_unallocated_baseline": route_unknown,
                    "route_unallocated_scenario": route_scenario_unknown,
                }
            )
        sources = [
            {
                "id": "calendar",
                "label": "Календарь",
                "status": "used_in_baseline",
                "detail": (
                    "Производственный календарь используется в выбранной модели; "
                    "отдельная сценарная ветка доступна по весу."
                ),
                "url": "https://government.ru/docs/52895/",
            },
            {
                "id": "weather",
                "label": "Погода",
                "status": "collected_not_selected",
                "detail": (
                    "Измеренное ухудшение; погода не включена в выбранную модель. "
                    "Отдельная сценарная ветка доступна по весу."
                ),
                "url": "https://open-meteo.com/en/docs/historical-weather-api",
            },
            {
                "id": "events",
                "label": "Новости и инциденты",
                "status": "evaluated_not_selected",
                "detail": (
                    "Исторические сообщения о перекрытиях и инцидентах; "
                    "устойчивый прирост точности не подтверждён."
                ),
                "url": "https://t.me/s/DtOperativno",
            },
            {
                "id": "traffic",
                "label": "Дорожный трафик",
                "status": "evaluated_not_selected",
                "detail": "Городские сводки Дептранса: эффект смешанный, модель не выбрана.",
                "url": "https://t.me/s/DtOperativno",
            },
        ]
        warnings = [
            (
                "Остановки и направления — оценки по косвенным данным, точность "
                "геопривязки не измерена; неизвестная масса сохранена."
            ),
            (
                "Веса нормированы вместе с базовым прогнозом (вес 1). "
                "Источник меняет сценарий, а не переобучает основную модель."
            ),
            "Число валидаций не является числом уникальных пассажиров или наполненностью салона.",
        ]
        if provenance.get("mapping_method"):
            warnings.append(
                "История восстановлена по первому событию рейса и скорректированному "
                "расписанию; для части дат использован перенос расписания с другой даты."
            )
        if selected_stop is not None or selected_direction is not None:
            warnings.append(
                "Неизвестная масса исключена из фильтра: её остановка/направление неизвестны. "
                "Полный маршрутный знаменатель сохранён отдельно."
            )
        if (request["forecast_mode"] == "approved"
                and any(request["source_enabled"].values())
                and not has_factor_override):
            warnings.append(
                "Включены отдельные обученные ветки источников; этот вариант не отправлялся "
                "на лидерборд, публичный score к нему не относится."
            )
            warnings.append(
                "Погода — архив выпущенных прогнозов до даты расчёта; события — "
                "сообщения об инцидентах и перекрытиях, не афиша."
            )
            if request["source_enabled"]["traffic"] or request["source_enabled"]["events"]:
                warnings.append(
                    "Исходная редакция сводок за 2025 год не архивирована: время публикации "
                    "и отсутствие отметки об изменении — проверяемый прокси доступности."
                )
            for source in sources:
                if request["source_enabled"][source["id"]]:
                    source["status"] = "approved_source_enabled"
                    source["detail"] = variants["availability"][source["id"]]
                    if source["id"] == "weather":
                        source["url"] = "https://open-meteo.com/en/docs/historical-forecast-api"
        if request["forecast_mode"] == "external_experiment":
            warnings.append("Экспериментальная модель не прошла отбор в основной прогноз.")
            warnings.append(
                "Источники меняют маршрутный прогноз, доли остановок остаются прежними."
            )
            for source in sources:
                source["status"] = (
                    "experimental_enabled"
                    if request["source_enabled"][source["id"]]
                    else "experimental_disabled"
                )
                source["detail"] = {
                    "calendar": "Праздники и сезонность. Эффект неодинаков на разных окнах.",
                    "weather": "Погода за прошлые 28 дней из уточнённого архива. Эффект смешанный.",
                    "traffic": "Редкие городские сводки за прошлые 56 дней. Эффект смешанный.",
                    "events": "Прошлые сообщения о перекрытиях и инцидентах. Эффект смешанный.",
                }[source["id"]]
                if source["id"] == "events":
                    source["label"] = "События: сообщения о перекрытиях и инцидентах"
                    source["url"] = "https://t.me/s/DtOperativno"
        if request["forecast_mode"] == "stop_model":
            warnings.append(
                "Показан самостоятельный остановочный прогноз: без нормировки "
                "к маршрутной модели; конкурсный score другой модели к нему не относится."
            )
            if any(request["source_enabled"].values()):
                warnings.append(
                    "Обученные переключатели источников не применяются к остановочной модели: "
                    "таких вариантов обучения нет. Сценарные веса смешивают отдельные прогнозы."
                )
            for source in sources:
                source["status"] = (
                    "stop_model_fixed" if source["id"] == "calendar" else "manual_only"
                )
                source["detail"] = (
                    "Календарный профиль — фиксированная часть остановочной модели."
                    if source["id"] == "calendar"
                    else "Источник не обучен в этой модели; доступна отдельная сценарная ветка."
                )
        if raw_weights:
            warnings.append(
                "Сценарные ветки источников не прошли отбор в основной прогноз; "
                "нормировка весов не гарантирует точность или отсутствие завышения."
            )
            if request["forecast_mode"] == "approved":
                warnings.append(
                    "Погода — архив выпущенных до отсечки прогнозов; новости — "
                    "сообщения об инцидентах и перекрытиях, не афиша."
                )
                if "traffic" in raw_weights or "events" in raw_weights:
                    warnings.append(
                        "Исходная редакция сводок за 2025 год не архивирована; "
                        "время публикации и отсутствие отметки об изменении — "
                        "прокси доступности."
                    )
            for source in sources:
                if source["id"] in raw_weights:
                    source["status"] = "scenario_weighted"
                    source["detail"] = (
                        (
                            variants["availability"][source["id"]]
                            if request["forecast_mode"] == "approved"
                            else "Отдельный прогноз источника смешан с выбранной основой."
                        ) + " "
                        f"Нормированный вес: {weights[source['id']]:.2%}."
                    )
                    if request["forecast_mode"] == "approved" and source["id"] == "weather":
                        source["url"] = "https://open-meteo.com/en/docs/historical-forecast-api"
        if horizon == "year":
            warnings.append(
                "Год — качественный сценарий: после декабря повторяется средний "
                "профиль дня недели/часа двух прогнозных месяцев "
                "(в остановочном режиме — отдельно для каждой остановки); "
                "годовая сезонность и точность не подтверждены."
            )
        return {
            "route": route,
            "start_date": start.isoformat(),
            "horizon": horizon,
            "stop_id": selected_stop,
            "direction": selected_direction,
            "run_id": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
            "model_version": model_version,
            "generated_at": generated_at,
            "unit": "event_count",
            "timezone": "Europe/Moscow",
            "qualitative": horizon == "year",
            "warnings": warnings,
            "sources": sources,
            "factors": request["factors"],
            "points": points,
            "stops": stops,
            "routes": routes,
            "provenance": provenance,
            "forecast_mode": request["forecast_mode"],
            "source_enabled": request["source_enabled"],
            "experimental_evaluation": experimental_evaluation,
        }

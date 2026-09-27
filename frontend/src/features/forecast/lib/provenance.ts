import type { ForecastResponse } from "@/features/forecast/types"

export function formatMoscowInstant(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "Недоступно"
  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return "Недоступно"
  return new Intl.DateTimeFormat("ru-RU", {
    timeZone: "Europe/Moscow", day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit", timeZoneName: "shortOffset",
  }).format(date)
}

export function snapshotFreshness(updatedAt: number, now: number, failed: boolean, pollInterval: number | false) {
  const limit = Math.max(120_000, (pollInterval || 0) * 2)
  const age = Math.max(0, now - updatedAt)
  return { stale: updatedAt <= 0 || failed || age > limit, age, limit }
}

export function sourceCutoff(forecast: ForecastResponse): string | null {
  const run = forecast.run
  if (!run || run.dataset_id.startsWith("legacy-") || run.calendar_version.startsWith("legacy-")) return null
  return run.data_cutoff
}

export function dataKind(forecast: ForecastResponse): string {
  return forecast.run?.synthetic === true ? "Синтетические демоданные"
    : forecast.run?.synthetic === false ? "Данные источника" : "Происхождение данных не указано"
}

export function forecastUnit(forecast: ForecastResponse): string {
  const run = forecast.run
  if (!run) return "Единица не указана"
  if (run.target === "synthetic_boardings" && run.unit === "event_count") return "Синтетические посадки"
  if (run.target === "validation_count" && run.unit === "event_count") return "Валидации"
  if (run.target === "boarding_count" && run.unit === "passengers") return "Посадки"
  if (run.target === "onboard_load" && run.unit === "passengers") return "Пассажиры в салоне"
  return `${run.target} · ${run.unit}`
}

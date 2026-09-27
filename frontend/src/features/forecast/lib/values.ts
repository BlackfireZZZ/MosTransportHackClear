import type { ForecastPoint, ForecastResponse } from "@/features/forecast/types"


export function intervalBounds(point: Pick<ForecastPoint, "lower_bound" | "upper_bound" | "predicted_passengers">): readonly [number, number] | null {
  const { lower_bound: lower, upper_bound: upper, predicted_passengers: value } = point
  return lower !== null && upper !== null && [lower, upper, value].every(Number.isFinite)
    && lower >= 0 && lower <= value && value <= upper ? [lower, upper] : null
}

export function formatForecastValue(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "Недоступно" : String(value).replace(".", ",")
}

export function intervalQuality(forecast: ForecastResponse): string {
  if (forecast.run?.synthetic) return "Демонстрационные границы; качество интервалов на реальных данных не оценено."
  const run = forecast.run
  const metadata = run?.interval_method && run.interval_level !== null
    ? `Метод: ${run.interval_method}; номинальный уровень: ${formatForecastValue(run.interval_level)}. ` : "Метод и номинальный уровень не указаны. "
  return `${metadata}Оценка качества интервалов недоступна.`
}

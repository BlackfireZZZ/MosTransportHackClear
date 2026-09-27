import type { ForecastHorizon } from "@/features/forecast/types"

export interface ForecastFilters {
  stop_id?: number
  start?: string
  end?: string
}

const limits: Record<ForecastHorizon, number> = { day: 1, month: 31, year: 366 }
const moscow = new Intl.DateTimeFormat("sv-SE", {
  timeZone: "Europe/Moscow", year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
})

function moscowStamp(instant: number): string {
  const parts = Object.fromEntries(moscow.formatToParts(new Date(instant)).map((part) => [part.type, part.value]))
  return `${parts.year.padStart(4, "0")}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}`
}

export function toMoscowInput(instant: string): string {
  return moscowStamp(Date.parse(instant)).slice(0, 16)
}

export function fromMoscowInput(value: string): string | null {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value) || Number(value.slice(0, 4)) < 1) return null
  const wall = Date.parse(`${value}:00Z`)
  if (!Number.isFinite(wall)) return null
  const offsets = new Set([-1, 0, 1].map((day) => {
    const sample = wall + day * 86_400_000
    return Date.parse(`${moscowStamp(sample)}Z`) - sample
  }))
  const candidates = [...offsets].map((offset) => wall - offset).filter((instant) =>
    Number.isFinite(instant) && new Date(instant).getUTCFullYear() >= 1 && moscowStamp(instant) === `${value}:00`,
  )
  return candidates.length === 1 ? new Date(candidates[0]).toISOString() : null
}

export function validateWindow(startInput: string, endInput: string, horizon: ForecastHorizon): {
  start?: string
  end?: string
  error?: string
} {
  if (!startInput && !endInput) return {}
  if (!startInput || !endInput) return { error: "Укажите начало и конец интервала." }
  const start = fromMoscowInput(startInput)
  const end = fromMoscowInput(endInput)
  if (!start || !end) return { error: "Укажите однозначные дату и время по Москве." }
  const duration = Date.parse(end) - Date.parse(start)
  if (duration <= 0) return { error: "Конец интервала должен быть позже начала." }
  if (duration > limits[horizon] * 86_400_000) return { error: `Максимальный интервал: ${limits[horizon]} дн.` }
  return { start, end }
}

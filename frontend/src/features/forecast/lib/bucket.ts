import type { ForecastResponse } from "@/features/forecast/types"

export function bucketScope(data: ForecastResponse): string {
  return JSON.stringify([data.route.id, data.horizon, data.selection?.start, data.selection?.end, data.selection?.stop_id, data.selection?.direction_id])
}

export function selectedBucket(data: ForecastResponse, selection: { scope: string; timestamp: string } | null): string | null {
  return selection?.scope === bucketScope(data) && data.points.some((point) => point.timestamp === selection.timestamp)
    ? selection.timestamp : data.points[0]?.timestamp ?? null
}

export function bucketStops(data: ForecastResponse, timestamp: string | null) {
  return (data.stop_points ?? []).filter((point) => point.timestamp === timestamp)
}

export function bucketLabel(timestamp: string): string {
  return new Intl.DateTimeFormat("ru-RU", {
    year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "Europe/Moscow",
  }).format(new Date(timestamp))
}

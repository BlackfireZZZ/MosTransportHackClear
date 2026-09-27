import type { PlanningPoint } from "@/api/planning"
import { HOT_RATE } from "./flow-color"

export interface HotSpot {
  timestamp: string
  stopId: string
  name: string
  direction: string
  count: number
  rate: number
}

export interface PlanningInsights {
  total: number
  peak: { timestamp: string; count: number; rate: number } | null
  busiest: { stopId: string; name: string; direction: string; count: number } | null
  hotSpots: HotSpot[]
}

export function planningInsights(points: readonly PlanningPoint[], limit = 5): PlanningInsights {
  let total = 0
  let peak: PlanningInsights["peak"] = null
  const stops = new Map<string, NonNullable<PlanningInsights["busiest"]>>()
  const hot: HotSpot[] = []
  for (const point of points) {
    total += point.scenario
    const hours = (Date.parse(point.bucket_end) - Date.parse(point.timestamp)) / 3_600_000
    if (!(hours > 0)) continue
    if (point.scenario > (peak?.count ?? 0)) peak = { timestamp: point.timestamp, count: point.scenario, rate: point.scenario / hours }
    for (const stop of point.spatial) {
      const key = `${stop.stop_id}|${stop.direction}`
      const entry = stops.get(key) ?? { stopId: stop.stop_id, name: stop.name, direction: stop.direction, count: 0 }
      entry.count += stop.scenario
      stops.set(key, entry)
      const rate = stop.scenario / hours
      if (rate >= HOT_RATE) hot.push({ timestamp: point.timestamp, stopId: stop.stop_id, name: stop.name, direction: stop.direction, count: stop.scenario, rate })
    }
  }
  let busiest: PlanningInsights["busiest"] = null
  for (const entry of stops.values()) if (entry.count > (busiest?.count ?? 0)) busiest = entry
  return { total, peak, busiest, hotSpots: hot.sort((a, b) => b.rate - a.rate).slice(0, limit) }
}

export interface RangeTotals {
  from: string
  to: string
  count: number
  baseline: number
  scenario: number
  unallocated: number
}

/** Sums buckets between two bucket starts, inclusive and in either order; unknown timestamps fall back to the period ends. */
export function rangeTotals(points: readonly PlanningPoint[], from: string | null, to: string | null): RangeTotals | null {
  if (points.length === 0) return null
  const find = (value: string | null, fallback: number) => {
    const index = points.findIndex((point) => point.timestamp === value)
    return index === -1 ? fallback : index
  }
  const [start, end] = [find(from, 0), find(to, points.length - 1)].sort((a, b) => a - b)
  const slice = points.slice(start, end + 1)
  const sum = (pick: (point: PlanningPoint) => number) => slice.reduce((total, point) => total + pick(point), 0)
  return { from: points[start].timestamp, to: points[end].timestamp, count: slice.length,
    baseline: sum((point) => point.baseline), scenario: sum((point) => point.scenario), unallocated: sum((point) => point.unallocated_scenario) }
}

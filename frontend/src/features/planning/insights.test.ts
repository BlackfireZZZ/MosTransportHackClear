import { describe, expect, it } from "vitest"
import type { PlanningPoint } from "@/api/planning"
import { planningInsights, rangeTotals } from "./insights"

const stop = (stop_id: string, direction: string, scenario: number) => ({ stop_id, name: `Остановка ${stop_id}`, direction, latitude: 55.7, longitude: 37.6, baseline: scenario, scenario })
const point = (timestamp: string, bucket_end: string, scenario: number, spatial: PlanningPoint["spatial"]): PlanningPoint => ({
  basis: "competition_period",
  timestamp, bucket_end, baseline: scenario, scenario, route_baseline: scenario, route_scenario: scenario,
  route_unallocated_baseline: 0, route_unallocated_scenario: 0, unallocated_baseline: 0, unallocated_scenario: 0, spatial,
})

const hourly = [
  point("2025-11-01T07:00:00+03:00", "2025-11-01T08:00:00+03:00", 300, [stop("a", "0", 120), stop("b", "1", 60), stop("c", "0", 9)]),
  point("2025-11-01T08:00:00+03:00", "2025-11-01T09:00:00+03:00", 500, [stop("a", "0", 40), stop("b", "1", 200), stop("c", "0", 50)]),
  point("2025-11-01T09:00:00+03:00", "2025-11-01T10:00:00+03:00", 100, [stop("a", "0", 30), stop("b", "1", 20), stop("c", "0", 49.9)]),
]

describe("planning insights", () => {
  it("finds the peak bucket, the busiest stop over the period and the total", () => {
    const result = planningInsights(hourly)
    expect(result.total).toBe(900)
    expect(result.peak).toEqual({ timestamp: "2025-11-01T08:00:00+03:00", count: 500, rate: 500 })
    expect(result.busiest).toMatchObject({ stopId: "b", direction: "1", count: 280 })
  })

  it("ranks stop-intervals at or above the red band by boardings per hour", () => {
    const hot = planningInsights(hourly).hotSpots
    expect(hot.map((row) => [row.stopId, row.timestamp.slice(11, 13), row.rate])).toEqual([
      ["b", "08", 200], ["a", "07", 120], ["b", "07", 60], ["c", "08", 50],
    ])
    expect(planningInsights(hourly, 2).hotSpots).toHaveLength(2)
  })

  it("compares daily buckets per hour, like the map colours", () => {
    const daily = [point("2025-11-01T00:00:00+03:00", "2025-11-02T00:00:00+03:00", 2000, [stop("a", "0", 1200), stop("b", "0", 1199)])]
    const result = planningInsights(daily)
    expect(result.peak?.rate).toBe(2000 / 24)
    expect(result.hotSpots.map((row) => row.stopId)).toEqual(["a"])
  })

  it("reports nothing instead of zeros when there are no boardings", () => {
    expect(planningInsights([])).toEqual({ total: 0, peak: null, busiest: null, hotSpots: [] })
    const idle = planningInsights([point("2025-11-01T02:00:00+03:00", "2025-11-01T03:00:00+03:00", 0, [stop("a", "0", 0)])])
    expect(idle).toEqual({ total: 0, peak: null, busiest: null, hotSpots: [] })
  })
})

describe("range totals", () => {
  it("sums an inclusive bucket range in either order", () => {
    const [a, , c] = hourly.map((point) => point.timestamp)
    expect(rangeTotals(hourly, a, c)).toMatchObject({ count: 3, baseline: 900, scenario: 900 })
    expect(rangeTotals(hourly, c, hourly[1].timestamp)).toMatchObject({ from: hourly[1].timestamp, to: c, count: 2, baseline: 600 })
  })

  it("falls back to the whole period for unknown ends and reports nothing without buckets", () => {
    expect(rangeTotals(hourly, null, "missing")).toMatchObject({ count: 3, baseline: 900 })
    expect(rangeTotals([], null, null)).toBeNull()
  })
})

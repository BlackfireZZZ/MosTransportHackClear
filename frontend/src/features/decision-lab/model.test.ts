import { describe, expect, it } from "vitest"
import type { PlanningResponse } from "@/api/planning"
import { response } from "@/features/planning/fixtures.test-support"
import { pressureTier, rankRoutes, scheduleWindowUsable, simulateService } from "./model"
import { serviceEstimate } from "./service-estimates"

const day = (route: string, values: number[]): PlanningResponse => ({ ...response, route, points: values.map((value, hour) => ({
  ...response.points[0], timestamp: `2025-11-01T${String(hour).padStart(2, "0")}:00:00+03:00`,
  bucket_end: `2025-11-${hour === 23 ? "02T00" : `01T${String(hour + 1).padStart(2, "0")}`}:00:00+03:00`,
  route_baseline: value, route_scenario: value, baseline: value, scenario: value,
})) })

describe("network decision lab", () => {
  it("ranks published routes by consecutive three-hour boardings, preserving route totals", () => {
    const rankings = rankRoutes([day("1", [10, 30, 40, 50, ...Array<number>(20).fill(0)]), day("7", [100, ...Array<number>(23).fill(0)])])
    expect(rankings.map((route) => route.route)).toEqual(["1", "7"])
    expect(rankings[0]).toMatchObject({ windowStart: 1, windowBoardings: 120, dayBoardings: 130, peakHour: 3 })
    expect(rankings[0].windowShare).toBeCloseTo(120 / 130)
  })

  it("redistributes the same boardings over assumed departures for one added vehicle", () => {
    const ranked = rankRoutes([day("1", Array<number>(24).fill(120))])[0]
    const result = simulateService(ranked.points, 7, 10, { headwayMinutes: 12, cycleMinutes: 60, extraVehicles: 1, directions: 2 })!
    expect(result.baselineDeparturesPerHour).toBe(10)
    expect(result.scenarioDeparturesPerHour).toBe(12)
    expect(result.scenarioHeadwayMinutes).toBe(10)
    expect(result.extraDeparturesInWindow).toBe(6)
    expect(result.baselineBoardingsPerDeparture).toBe(12)
    expect(result.scenarioBoardingsPerDeparture).toBe(10)
    expect(result.hours.reduce((sum, hour) => sum + hour.boardings, 0)).toBe(result.windowBoardings)
  })

  it("rejects unavailable or inconsistent model and operational inputs", () => {
    expect(rankRoutes([{ ...day("1", Array<number>(24).fill(2)), forecast_mode: "stop_model" }])).toEqual([])
    expect(rankRoutes([day("1", Array<number>(23).fill(2))])).toEqual([])
    const duplicate = day("1", Array<number>(24).fill(2))
    duplicate.points[1] = { ...duplicate.points[1], timestamp: duplicate.points[0].timestamp }
    expect(rankRoutes([duplicate])).toEqual([])
    const points = day("1", Array<number>(24).fill(2)).points
    expect(simulateService(points, 0, 3, { headwayMinutes: 0, cycleMinutes: 60, extraVehicles: 1, directions: 2 })).toBeNull()
    expect(simulateService(points, 22, 25, { headwayMinutes: 12, cycleMinutes: 60, extraVehicles: 1, directions: 2 })).toBeNull()
    expect(simulateService(points, 0, 3, { headwayMinutes: 12, cycleMinutes: 60, extraVehicles: 0.5, directions: 2 })).toBeNull()
    expect(simulateService(points, 0, 3, { headwayMinutes: 12, cycleMinutes: 60, extraVehicles: 1, directions: 1 })?.baselineBoardingsPerDeparture).toBe(2 / 5)
  })

  it("supports a one-hour or full-day window and ranks by scheduled demand pressure", () => {
    const first = day("1", [120, 120, ...Array<number>(22).fill(0)])
    const second = day("7", [80, 80, ...Array<number>(22).fill(0)])
    const departures = { "1": [12, 12, ...Array<number>(22).fill(12)], "7": [2, 2, ...Array<number>(22).fill(12)] }
    expect(rankRoutes([first, second], 1, departures).map((item) => item.route)).toEqual(["7", "1"])
    expect(rankRoutes([first, second], 2, departures)[0]).toMatchObject({ route: "7", windowStart: 0, windowEnd: 2, scheduledDepartures: 4, boardingsPerDeparture: 40 })
    expect(simulateService(first.points, 0, 24, { headwayMinutes: 12, cycleMinutes: 60, extraVehicles: 1, directions: 2 })?.hours).toHaveLength(24)
    expect(pressureTier(150, 100)).toBe("high")
    expect(pressureTier(115, 100)).toBe("elevated")
    expect(pressureTier(null, 100)).toBe("unknown")
    expect(scheduleWindowUsable([0, 0, 0, 0, 1, 1, ...Array<number>(18).fill(10)], 0, 6)).toBe(false)
    expect(scheduleWindowUsable(Array<number>(24).fill(2), 0, 24)).toBe(true)
  })

  it("resolves a dated schedule profile including previous service-day departures", () => {
    expect(serviceEstimate("17", "2025-11-03")).toMatchObject({ cycleMinutes: 87, pairedCycles: 456 })
    expect(serviceEstimate("17", "2025-11-01")).toMatchObject({ cycleMinutes: 82, pairedCycles: 399 })
    expect(serviceEstimate("5", "2025-11-01")).toBeNull()
    expect(serviceEstimate("17", "2025-11-03")?.departuresByHour).toHaveLength(24)
    expect(serviceEstimate("17", "2024-01-01")).toBeNull()
  })
})

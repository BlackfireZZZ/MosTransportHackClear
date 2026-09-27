import { describe, expect, it } from "vitest"
import { forecastResponse, routes } from "./forecast-fixtures"

describe("browser forecast fixture contracts", () => {
  for (const route of routes) {
    for (const horizon of ["day", "month", "year"] as const) {
      it(`reconciles route ${route.id} ${horizon} values and Moscow calendar buckets`, () => {
        const forecast = forecastResponse(route.id, horizon)
        expect(forecast).toEqual(forecastResponse(route.id, horizon))
        expect(forecast.route).toEqual(route)
        expect(forecast.peak_passengers).toBe(Math.max(...forecast.points.map((p) => p.predicted_passengers)))
        expect(forecast.peak_load_percent).toBe(Math.max(...forecast.points.map((p) => p.predicted_passengers / (p.capacity ?? NaN) * 100)))
        expect(forecast.points).toHaveLength(horizon === "day" ? 24 : horizon === "month" ? 31 : 12)
        forecast.points.forEach((point, index) => {
          expect(point.lower_bound).toBeLessThanOrEqual(point.predicted_passengers)
          expect(point.upper_bound).toBeGreaterThanOrEqual(point.predicted_passengers)
          expect(point.lower_bound).toBeGreaterThanOrEqual(0)
          const instant = new Date(point.timestamp).getTime()
          const moscow = new Date(instant + 3 * 60 * 60 * 1000)
          expect(moscow.getUTCMinutes()).toBe(0)
          expect(moscow.getUTCSeconds()).toBe(0)
          if (index) expect(instant).toBeGreaterThan(new Date(forecast.points[index - 1].timestamp).getTime())
          if (horizon === "day") {
            expect(moscow.getUTCDate()).toBe(1)
            expect(moscow.getUTCHours()).toBe(index)
          } else {
            expect(moscow.getUTCHours()).toBe(0)
            expect(moscow.getUTCDate()).toBe(horizon === "month" ? index + 1 : 1)
            expect(moscow.getUTCMonth()).toBe(horizon === "month" ? 9 : index)
          }
        })
      })
    }
  }
  it("rejects unknown routes instead of silently using another route", () => {
    expect(() => forecastResponse(999, "day")).toThrow("Unknown fixture route")
  })
})

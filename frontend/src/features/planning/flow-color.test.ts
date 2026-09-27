import { describe, expect, it } from "vitest"
import { flowAppearance, RATE_STOPS, rateColor, ZERO_COLOR } from "./flow-color"

const start = "2025-11-01T00:00:00+03:00"
const hour = "2025-11-01T01:00:00+03:00"
const lightness = (color: string) => [1, 3, 5].map((index) => Number.parseInt(color.slice(index, index + 2), 16)).reduce((sum, value, index) => sum + value * [0.299, 0.587, 0.114][index], 0)

describe("boarding rate gradient", () => {
  it("keeps zero gray and hits every stop colour exactly", () => {
    expect(rateColor(0)).toBe(ZERO_COLOR)
    for (const stop of RATE_STOPS.slice(1)) expect(rateColor(stop.rate)).toBe(stop.color)
    expect(rateColor(0.1)).toBe(RATE_STOPS[0].color)
    expect(rateColor(500)).toBe(RATE_STOPS.at(-1)?.color)
  })

  it("darkens monotonically as the rate grows, so neighbouring rates stay distinguishable", () => {
    const samples = [1, 5, 10, 15, 25, 40, 50, 70, 100].map((rate) => lightness(rateColor(rate)))
    for (let index = 1; index < samples.length; index += 1) expect(samples[index]).toBeLessThan(samples[index - 1])
  })

  it("compares hourly and daily buckets in the same units", () => {
    expect(flowAppearance(240, start, "2025-11-02T00:00:00+03:00")).toEqual(flowAppearance(10, start, hour))
    expect(flowAppearance(7200, start, "2025-12-01T00:00:00+03:00")).toEqual(flowAppearance(10, start, hour))
  })

  it("does not disguise invalid counts or intervals as observed zero", () => {
    for (const count of [-1, NaN, Infinity]) expect(flowAppearance(count, start, hour).rate).toBeNull()
    expect(flowAppearance(10, start, start).rate).toBeNull()
  })
})

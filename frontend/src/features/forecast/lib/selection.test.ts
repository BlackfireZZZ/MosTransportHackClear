import { describe, expect, it } from "vitest"
import { fromMoscowInput, toMoscowInput, validateWindow } from "./selection"

describe("explicit Moscow window", () => {
  it("is independent of browser timezone and handles year rollover", () => {
    expect(fromMoscowInput("2026-01-01T00:00")).toBe("2025-12-31T21:00:00.000Z")
    expect(toMoscowInput("2025-12-31T21:00:00Z")).toBe("2026-01-01T00:00")
  })
  it("uses the historical Moscow offset and rejects missing local time", () => {
    expect(fromMoscowInput("2012-01-01T00:00")).toBe("2011-12-31T20:00:00.000Z")
    expect(fromMoscowInput("2011-03-27T02:30")).toBeNull()
    expect(fromMoscowInput("2014-10-26T01:30")).toBeNull()
    expect(fromMoscowInput("0000-01-01T00:00")).toBeNull()
    expect(fromMoscowInput("0001-01-01T00:00")).toBeNull()
  })
  it.each(["invalid", "2026-02-30T12:00", "2026-01-01T25:00"])("rejects malformed calendar date %s", (value) => {
    expect(fromMoscowInput(value)).toBeNull()
  })
  it("accepts full defaults, forbids partial, reversed and overlong windows", () => {
    expect(validateWindow("", "", "day")).toEqual({})
    expect(validateWindow("2026-01-01T00:00", "", "day").error).toBeTruthy()
    expect(validateWindow("2026-01-02T00:00", "2026-01-01T00:00", "day").error).toBeTruthy()
    expect(validateWindow("2026-01-01T00:00", "2026-01-03T00:00", "day").error).toBeTruthy()
    expect(validateWindow("2026-01-01T00:00", "2026-01-02T00:00", "day")).toEqual({ start: "2025-12-31T21:00:00.000Z", end: "2026-01-01T21:00:00.000Z" })
  })
})

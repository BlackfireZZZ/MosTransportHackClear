import { describe, expect, it } from "vitest"
import type { ForecastResponse } from "@/features/forecast/types"
import { dataKind, forecastUnit, formatMoscowInstant, snapshotFreshness, sourceCutoff } from "./provenance"

describe("provenance and UI check freshness", () => {
  it("distinguishes expiry, failure and recovery without changing source time", () => {
    expect(snapshotFreshness(1_000, 121_000, false, 60_000).stale).toBe(false)
    expect(snapshotFreshness(1_000, 121_001, false, 60_000).stale).toBe(true)
    expect(snapshotFreshness(1_000, 2_000, true, 60_000).stale).toBe(true)
    expect(snapshotFreshness(2_000, 2_000, false, 60_000).stale).toBe(false)
    expect(snapshotFreshness(1_000, 601_000, false, 300_000).stale).toBe(false)
    expect(snapshotFreshness(0, 1_000, false, false).stale).toBe(true)
    expect(snapshotFreshness(1_000, 122_000, false, false).stale).toBe(true)
  })
  it("renders full Moscow date across a year boundary", () => {
    expect(formatMoscowInstant("2025-12-31T21:00:00Z")).toContain("01.01.2026")
    expect(formatMoscowInstant("2025-12-31T21:00:00Z")).toContain("00:00:00")
    expect(formatMoscowInstant(0)).toContain("01.01.1970")
    expect(formatMoscowInstant("bad")).toBe("Недоступно")
  })
  it("does not turn absent metadata or legacy placeholders into observations", () => {
    const missing = {} as ForecastResponse
    expect(dataKind(missing)).toBe("Происхождение данных не указано")
    expect(forecastUnit(missing)).toBe("Единица не указана")
    expect(sourceCutoff(missing)).toBeNull()
    const legacy = { run: { dataset_id: "legacy-demo", calendar_version: "v1", data_cutoff: "2026-01-01T00:00:00Z" } } as ForecastResponse
    expect(sourceCutoff(legacy)).toBeNull()
    const real = { run: { ...legacy.run, dataset_id: "observed-v1", synthetic: false, target: "validation_count", unit: "event_count" } } as ForecastResponse
    expect(sourceCutoff(real)).toBe("2026-01-01T00:00:00Z")
    expect(dataKind(real)).toBe("Данные источника")
    expect(forecastUnit(real)).toBe("Валидации")
  })
})

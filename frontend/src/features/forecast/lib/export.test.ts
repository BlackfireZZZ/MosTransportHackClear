import { expect, it } from "vitest"
import { mappedForecastResponse } from "../../../../e2e/forecast-fixtures"
import { readCsv } from "../../../../e2e/csv-reader"
import { forecastCsv, type ExportContext } from "./export"

const context: ExportContext = { updatedAt: 1000, exportedAt: 2000, failed: false, pollInterval: 60_000, selectedTimestamp: null, requestedFilters: {} }
it.each(["day", "month", "year"] as const)("exports exact %s snapshot values and identity", (horizon) => {
  const data = mappedForecastResponse(1, horizon)
  const rows = readCsv(forecastCsv(data, { ...context, selectedTimestamp: data.points[1].timestamp })!)
  expect(rows).toHaveLength(data.points.length)
  expect(rows.map((row) => Number(row.predicted_value))).toEqual(data.points.map((point) => point.predicted_passengers))
  expect(rows[0]).toMatchObject({ run_id: data.run!.run_id, timezone: "Europe/Moscow", target: "synthetic_boardings", unit: "event_count", synthetic: "true", stale: "false", capacity: "", capacity_status: "unavailable_no_compatible_metadata" })
  expect(rows[1].is_selected_bucket).toBe("true")
  expect(rows[0].is_selected_bucket).toBe("false")
})

it("preserves zero and fractional values, distinguishes unavailable, and marks stale legacy data", () => {
  const base = mappedForecastResponse(1, "day")
  const data = { ...base, run: { ...base.run!, dataset_id: "legacy-demo" }, points: [
    { ...base.points[0], predicted_passengers: 0, lower_bound: 0, upper_bound: 0 },
    { ...base.points[1], predicted_passengers: 12.345, lower_bound: null, upper_bound: null },
  ] }
  const rows = readCsv(forecastCsv(data, { ...context, failed: true })!)
  expect(rows[0]).toMatchObject({ predicted_value: "0", lower_bound: "0", upper_bound: "0", data_cutoff: "", stale: "true" })
  expect(rows[1]).toMatchObject({ predicted_value: "12.345", lower_bound: "", upper_bound: "" })
  expect(forecastCsv({ ...data, points: [] }, context)).toBeNull()
})

it.each(['=1+1', ' +SUM(1,2)', '@cmd', '\ttext', '\r\n=cmd', '＝SUM(1)', '-12', '"quoted",\r\nname'])("quotes and protects untrusted text %s", (name) => {
  const base = mappedForecastResponse(1, "day")
  const rows = readCsv(forecastCsv({ ...base, route: { ...base.route, name } }, context)!)
  const escaped = rows[0].escaped_text_columns.includes("route_name")
  expect(rows[0].route_name).toBe(escaped ? `text:${name}` : name)
  expect(escaped).toBe(!name.startsWith('"'))
  expect(rows[0].predicted_value).toBe(String(base.points[0].predicted_passengers))
  expect(rows).toHaveLength(base.points.length)
})


it("preserves unavailable mapping evidence and rejects a mismatched map envelope", () => {
  const base = mappedForecastResponse(1, "day")
  const staleMap = { ...base.map!, mapping_version: "old-mapping", reason: "graph_version_mismatch" }
  const rows = readCsv(forecastCsv({ ...base, map: staleMap }, context)!)
  expect(rows[0]).toMatchObject({ mapping_version: "old-mapping", mapping_status: "unavailable", mapping_reason: "graph_version_mismatch" })
  const wrongRun = readCsv(forecastCsv({ ...base, map: { ...staleMap, run_id: "other-run" } }, context)!)
  expect(wrongRun[0]).toMatchObject({ mapping_version: "", mapping_status: "unavailable", mapping_reason: "missing_or_incompatible_envelope" })
})

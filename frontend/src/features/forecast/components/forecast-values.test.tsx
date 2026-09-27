import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { mappedForecastResponse } from "../../../../e2e/forecast-fixtures"
import { ForecastValues } from "./forecast-values"
import { formatForecastValue, intervalBounds, intervalQuality } from "../lib/values"

afterEach(cleanup)
it.each(["2025-12-31T21:00:00Z", "2024-02-28T21:00:00Z"])("exposes exact values and full Moscow date for %s", (timestamp) => {
  const base = mappedForecastResponse(1, "year")
  const snapshot = { ...base, points: [{ timestamp, predicted_passengers: 0, lower_bound: 0, upper_bound: 0, capacity: 0 },
    { timestamp: "2026-10-01T01:00:00Z", predicted_passengers: 12.345, lower_bound: null, upper_bound: null, capacity: null }] }
  const select = vi.fn()
  render(<ForecastValues snapshot={snapshot} selectedTimestamp={timestamp} onSelectTimestamp={select} />)
  fireEvent.click(screen.getByText(/Все значения прогноза —/))
  const table = screen.getByRole("table", { name: "Все значения прогноза" })
  const rows = within(table).getAllByRole("row")
  expect(rows[1]).toHaveTextContent(timestamp.startsWith("2025") ? "01.01.2026" : "29.02.2024")
  expect(within(rows[1]).getAllByRole("cell").map((cell) => cell.textContent)).toEqual(["0", "0", "0", "Недоступно"])
  expect(within(rows[2]).getAllByRole("cell").map((cell) => cell.textContent)).toEqual(["12,345", "Недоступно", "Недоступно", "Недоступно"])
  fireEvent.click(within(rows[2]).getByRole("button"))
  expect(select).toHaveBeenCalledWith(snapshot.points[1].timestamp)
  expect(screen.getByText(/качество интервалов на реальных данных не оценено/)).toBeVisible()
})

it("rejects incoherent bounds and never infers calibration from nominal metadata", () => {
  expect(intervalBounds({ lower_bound: 4, predicted_passengers: 3, upper_bound: 6 })).toBeNull()
  expect(intervalBounds({ lower_bound: 0, predicted_passengers: 3, upper_bound: Infinity })).toBeNull()
  expect(formatForecastValue(null)).toBe("Недоступно")
  expect(formatForecastValue(0)).toBe("0")
  const data = mappedForecastResponse(1, "month")
  expect(intervalQuality({ ...data, run: { ...data.run!, synthetic: false, interval_method: "conformal", interval_level: 0.9 } })).toBe("Метод: conformal; номинальный уровень: 0,9. Оценка качества интервалов недоступна.")
})

import { act, cleanup, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { forecastResponse } from "../../../../e2e/forecast-fixtures"
import { ProvenancePanel } from "./provenance-panel"

const forecast = forecastResponse(1, "day")
afterEach(() => { cleanup(); vi.useRealTimers() })

it("ages a successful UI check and recovers without rewriting source timestamps", () => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date("2026-10-01T00:00:00Z"))
  const updatedAt = Date.now()
  const props = { forecast, updatedAt, failed: false, fetching: false, pollInterval: 60_000 }
  const { rerender } = render(<ProvenancePanel {...props} />)
  expect(screen.getByTestId("forecast-freshness")).toHaveTextContent("проверка прогноза успешна")
  act(() => { vi.advanceTimersByTime(130_000) })
  expect(screen.getByTestId("forecast-freshness")).toHaveTextContent("Устаревший снимок")
  rerender(<ProvenancePanel {...props} failed />)
  expect(screen.getByTestId("forecast-freshness")).toHaveTextContent("Обновление не удалось")
  rerender(<ProvenancePanel {...props} updatedAt={Date.now()} />)
  expect(screen.getByTestId("forecast-freshness")).not.toHaveTextContent("Устаревший")
  expect(screen.getByText("Граница исходных данных · МСК").nextElementSibling).toHaveTextContent("Недоступно")
  expect(screen.getByText(/Поток оперативных наблюдений не подключён/)).toBeVisible()
})

it("describes manual mode and retains the UI-check expiry policy", () => {
  render(<ProvenancePanel forecast={forecast} updatedAt={Date.now()} failed={false} fetching={false} pollInterval={false} />)
  expect(screen.getByText(/Автопроверка отключена/)).toHaveTextContent("не гарантия актуальности источника")
})

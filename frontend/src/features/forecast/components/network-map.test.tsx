import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { mappedForecastResponse } from "../../../../e2e/forecast-fixtures"
import { NetworkMap } from "./network-map"

vi.mock("@/features/tram-network/hooks/use-tram-network", () => ({ useNetworkGeoJson: () => ({ data: undefined, isError: false, isLoading: false }) }))
vi.mock("@/features/tram-network/components/tram-map", () => ({ TramMap: (props: { forecastMarkers: readonly { stopId: number; value: number }[]; onForecastStopClick: (id: number) => void }) =>
  <div>{props.forecastMarkers.map((marker) => <button key={marker.stopId} onClick={() => props.onForecastStopClick(marker.stopId)}>Точка {marker.stopId}: {marker.value}</button>)}</div>,
}))
afterEach(cleanup)

it("uses one timestamp for visible values and serving-ID map/table callbacks", async () => {
  const data = mappedForecastResponse(1, "day")
  const onSelect = vi.fn()
  const client = new QueryClient()
  const view = render(<QueryClientProvider client={client}><NetworkMap snapshot={data} timestamp={data.points[0].timestamp} selectedStopId={null} onStopSelect={onSelect} /></QueryClientProvider>)
  expect(screen.getByRole("columnheader", { name: "Направление" })).toBeInTheDocument()
  expect(screen.getAllByRole("cell", { name: "out" })).toHaveLength(2)
  const value = data.stop_points![0].predicted_passengers
  fireEvent.click(await screen.findByRole("button", { name: `Точка 11: ${value}` }))
  expect(onSelect).toHaveBeenLastCalledWith(11)
  fireEvent.click(screen.getByRole("button", { name: "Бета прогноза" }))
  expect(onSelect).toHaveBeenLastCalledWith(12)
  expect(screen.getByText(/Демонстрационные координаты/)).toBeInTheDocument()
  view.rerender(<QueryClientProvider client={client}><NetworkMap snapshot={{ ...data, stop_points: [] }} timestamp={data.points[1].timestamp} selectedStopId={null} onStopSelect={onSelect} /></QueryClientProvider>)
  expect(screen.getByText(/Нет прогноза по остановкам/)).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Точка/ })).not.toBeInTheDocument()
})

it.each(["run_id", "entity_version", "graph_version"] as const)("refuses positions with mismatched %s while retaining values", async (field) => {
  const data = mappedForecastResponse(1, "day")
  const wrong = { ...data, map: { ...data.map!, [field]: "other-version" } }
  render(<QueryClientProvider client={new QueryClient()}><NetworkMap snapshot={wrong} timestamp={data.points[0].timestamp} selectedStopId={null} onStopSelect={vi.fn()} /></QueryClientProvider>)
  await screen.findByRole("table")
  expect(screen.queryByRole("button", { name: /Точка/ })).not.toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Альфа прогноза" })).toBeInTheDocument()
})

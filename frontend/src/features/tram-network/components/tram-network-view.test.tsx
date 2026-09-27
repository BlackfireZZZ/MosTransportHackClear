import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"

import { api } from "@/api/client"
import { TramNetworkView } from "./tram-network-view"

vi.mock("./tram-map", () => ({ TramMap: () => <p>Доступ к локальному графу</p> }))
afterEach(() => { cleanup(); vi.restoreAllMocks() })

function mount(networkFails = false) {
  vi.spyOn(api.tramGraph, "geojson").mockResolvedValue({ type: "FeatureCollection", metadata: { missing_geometry_edges: 0, synthetic: false }, features: [] })
  if (networkFails) vi.mocked(api.tramGraph.geojson).mockRejectedValue(new Error("graph offline"))
  vi.spyOn(api.tramGraph, "routes").mockResolvedValue([])
  vi.spyOn(api.tramGraph, "stats").mockRejectedValue(new Error("stats offline"))
  vi.spyOn(api, "overpassStatus").mockRejectedValue(new Error("status offline"))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  render(<QueryClientProvider client={client}><TramNetworkView /></QueryClientProvider>)
}

it("keeps empty local topology usable through independent failures and scoped retries", async () => {
  mount()
  await screen.findByText("Статистика сети: ошибка загрузки")
  await screen.findByText(/проверка недоступна/)
  expect(screen.getByText("В графе нет объектов.")).toBeInTheDocument()
  expect(screen.getByText("В графе нет маршрутов.")).toBeInTheDocument()
  expect(screen.getByText("Доступ к локальному графу")).toBeInTheDocument()
  vi.mocked(api.overpassStatus).mockResolvedValueOnce({ reachable: true, url: "https://fixture.invalid", detail: null })
  fireEvent.click(screen.getByRole("button", { name: "Повторить: Overpass" }))
  await screen.findByText("доступен", { exact: true })
  expect(api.tramGraph.geojson).toHaveBeenCalledTimes(1)
  expect(api.tramGraph.routes).toHaveBeenCalledTimes(1)
  expect(api.tramGraph.stats).toHaveBeenCalledTimes(1)
  fireEvent.click(screen.getByRole("button", { name: "Повторить: статистика сети" }))
  await waitFor(() => expect(api.tramGraph.stats).toHaveBeenCalledTimes(2))
  expect(api.overpassStatus).toHaveBeenCalledTimes(2)
})

it("does not report unselected route geometry as another graph failure", async () => {
  mount(true)
  await screen.findByText("Граф сети: ошибка загрузки")
  expect(screen.queryByText("Геометрия маршрута: ошибка загрузки")).not.toBeInTheDocument()
  expect(screen.getByRole("searchbox", { name: "Поиск остановки" })).toBeEnabled()
  vi.mocked(api.tramGraph.geojson).mockResolvedValueOnce({ type: "FeatureCollection", metadata: { missing_geometry_edges: 0, synthetic: false }, features: [] })
  fireEvent.click(screen.getByRole("button", { name: "Повторить: граф сети" }))
  await screen.findByText("В графе нет объектов.")
  expect(api.tramGraph.routes).toHaveBeenCalledTimes(1)
})

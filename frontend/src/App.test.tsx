import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { ApiError, api } from "@/api/client"
import { planningForecast } from "@/api/planning"
import App from "./App"
import { forecastResponse, routes } from "../e2e/forecast-fixtures"
import { response as planningResponse } from "./features/planning/fixtures.test-support"

vi.mock("@/api/client", () => ({ ApiError: class extends Error { status: number; constructor(message: string, status: number) { super(message); this.status = status } }, api: { routes: vi.fn(), routeStops: vi.fn(), forecast: vi.fn() } }))
vi.mock("@/api/planning", async (original) => ({ ...await original<object>(), planningForecast: vi.fn() }))
vi.mock("@/features/forecast/components/forecast-chart", () => ({ ForecastChart: () => <p>Тестовый график</p> }))
vi.mock("@/features/forecast/components/network-map", () => ({ NetworkMap: () => <p>Тестовая карта</p> }))
beforeEach(() => { window.history.replaceState(null, "", "/?view=published"); vi.mocked(api.routeStops).mockResolvedValue([]) })
afterEach(() => { cleanup(); vi.resetAllMocks() })
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>)
  return client
}

it("keeps the published audit available without a duplicate navigation item", async () => {
  window.history.replaceState(null, "", "/")
  vi.mocked(api.routes).mockResolvedValue([...routes])
  vi.mocked(planningForecast).mockResolvedValue(planningResponse)
  mount()
  await screen.findByRole("heading", { name: "Посадки на трамвайной сети" })
  expect(screen.getByRole("navigation", { name: "Основная навигация" })).not.toHaveTextContent("Опубликованный прогноз")
})

it("shows waiting then a scoped route error without requesting forecasts", async () => {
  let reject!: (error: Error) => void
  vi.mocked(api.routes).mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail }))
  mount()
  expect(screen.getByText("Ожидание данных")).toBeInTheDocument()
  expect(screen.getByRole("combobox", { name: "Маршрут" })).toBeDisabled()
  expect(screen.getByRole("button", { name: "1 год · качественный сценарий" })).toBeDisabled()
  expect(api.forecast).not.toHaveBeenCalled()
  await act(async () => { reject(new Error("offline")); await Promise.resolve() })
  await screen.findByText("Маршруты временно недоступны")
  expect(api.forecast).not.toHaveBeenCalled()
  vi.mocked(api.routes).mockResolvedValueOnce([])
  fireEvent.click(screen.getByRole("button", { name: "Повторить загрузку маршрутов" }))
  await screen.findByText("Маршруты не найдены")
  expect(api.forecast).not.toHaveBeenCalled()
  expect(screen.queryByText("Модель доступна")).not.toBeInTheDocument()
})

it("distinguishes initial failure from retained demo data and recovers", async () => {
  vi.mocked(api.routes).mockResolvedValue([...routes])
  vi.mocked(api.forecast).mockRejectedValueOnce(new Error("offline"))
  mount()
  await screen.findByText("Прогноз временно недоступен")
  expect(screen.queryByRole("region", { name: "Ключевые показатели" })).not.toBeInTheDocument()
  vi.mocked(api.forecast).mockResolvedValueOnce(forecastResponse(1, "day"))
  fireEvent.click(screen.getByRole("button", { name: "Повторить прогноз" }))
  await screen.findByText("Тестовый график")
  expect(screen.getAllByText(/Происхождение данных не указано/).length).toBeGreaterThan(0)
  expect(screen.getByText("Прогноз сформирован · МСК")).toBeInTheDocument()
  vi.mocked(api.forecast).mockRejectedValueOnce(new Error("offline"))
  fireEvent.click(screen.getByRole("button", { name: "Обновить прогноз" }))
  await screen.findByText("Показан сохранённый прогноз")
  expect(screen.getByText("Тестовый график")).toBeInTheDocument()
  expect(screen.getByRole("region", { name: "Ключевые показатели" })).toHaveTextContent("600")
  vi.mocked(api.forecast).mockResolvedValueOnce(forecastResponse(1, "day"))
  fireEvent.click(screen.getByRole("button", { name: "Повторить прогноз" }))
  await waitFor(() => expect(screen.queryByText("Показан сохранённый прогноз")).not.toBeInTheDocument())
})

it("shows unknown capacity without inventing a zero percentage", async () => {
  vi.mocked(api.routes).mockResolvedValue([...routes])
  const response = forecastResponse(1, "day")
  vi.mocked(api.forecast).mockResolvedValue({ ...response, peak_load_percent: null,
    points: response.points.map((point) => ({ ...point, capacity: null, lower_bound: null, upper_bound: null })),
    stops: response.stops.map((stop) => ({ ...stop, load_percent: null })),
  })
  mount()
  await screen.findByText("Тестовый график")
  const kpis = screen.getByRole("region", { name: "Ключевые показатели" })
  expect(kpis).toHaveTextContent("Пиковый поток")
  expect(kpis).not.toHaveTextContent("0%")
  expect(kpis).not.toHaveTextContent("в пределах вместимости")
})


it.each([[404, "Нет прогноза для выбранных параметров"], [422, "Интервал недоступен"], [409, "Данные прогноза несовместимы"]] as const)("explains selection error %i and hides retry on invalid draft", async (status, heading) => {
  vi.mocked(api.routes).mockResolvedValue([...routes])
  vi.mocked(api.forecast).mockRejectedValue(new ApiError("selection", status))
  mount()
  await screen.findByRole("heading", { name: heading })
  fireEvent.change(screen.getByLabelText("Начало · МСК"), { target: { value: "2026-10-01T00:00" } })
  expect(screen.queryByRole("button", { name: "Повторить прогноз" })).not.toBeInTheDocument()
  expect(screen.getByText("Запрос не выполнен: исправьте интервал.")).toBeInTheDocument()
})

it("recovers the stop catalog without losing the route forecast", async () => {
  vi.mocked(api.routes).mockResolvedValue([...routes])
  vi.mocked(api.forecast).mockResolvedValue(forecastResponse(1, "day"))
  vi.mocked(api.routeStops).mockRejectedValueOnce(new Error("offline"))
  mount()
  await screen.findByRole("button", { name: "Повторить остановки" })
  expect(screen.getByRole("combobox", { name: "Остановка" })).toBeDisabled()
  await screen.findByText("Тестовый график")
  vi.mocked(api.routeStops).mockResolvedValueOnce([...forecastResponse(1, "day").stops])
  fireEvent.click(screen.getByRole("button", { name: "Повторить остановки" }))
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Остановка" })).not.toBeDisabled())
  expect(screen.getByText("Тестовый график")).toBeInTheDocument()
})

it("explains the unavailable observed stop catalog without a pointless retry", async () => {
  vi.mocked(api.routes).mockResolvedValue([...routes])
  vi.mocked(api.forecast).mockResolvedValue(forecastResponse(1, "day"))
  vi.mocked(api.routeStops).mockRejectedValue(new ApiError("estimated only", 409))
  mount()
  await screen.findByText(/нет наблюдаемого каталога остановок/)
  expect(screen.getByRole("combobox", { name: "Остановка" })).toBeDisabled()
  expect(screen.queryByRole("button", { name: "Повторить остановки" })).not.toBeInTheDocument()
  expect(screen.getByText("Тестовый график")).toBeInTheDocument()
})

it("opens the qualitative year from the published view without requesting a published year", async () => {
  vi.mocked(api.routes).mockResolvedValue([...routes])
  vi.mocked(api.forecast).mockResolvedValue(forecastResponse(1, "day"))
  vi.mocked(planningForecast).mockResolvedValue({ ...planningResponse, horizon: "year", qualitative: true })
  mount()
  await screen.findByText("Тестовый график")
  fireEvent.click(screen.getByRole("button", { name: /1 год · качественный сценарий/ }))
  await waitFor(() => expect(planningForecast).toHaveBeenCalledWith(expect.objectContaining({ horizon: "year" }), expect.anything()))
  expect(api.forecast).toHaveBeenCalledTimes(1)
  expect(screen.getByRole("note")).toHaveTextContent("Точность на годовом горизонте не проверена")
})

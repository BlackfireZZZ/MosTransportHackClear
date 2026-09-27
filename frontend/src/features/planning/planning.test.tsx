import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { planningForecast } from "@/api/planning"
import { PlanningView } from "./planning-view"
import { networkPlanningCsv, planningCsv } from "./export"
import { response } from "./fixtures.test-support"

vi.mock("@/api/planning", async (original) => ({ ...await original<object>(), planningForecast: vi.fn() }))
vi.mock("@/features/tram-network/hooks/use-tram-network", () => ({
  useNetworkGeoJson: () => ({ data: undefined, isError: false }),
  useStopDetail: (id: number | null) => ({
    data: id === 101 ? { id: 101, name: "Альфа", latitude: 55.75, longitude: 37.6, routes: ["1", "5"] } : undefined,
    isPending: false, isError: false, refetch: vi.fn(),
  }),
}))
const mapState = vi.hoisted(() => ({
  props: null as null | { forecastMarkers: Array<{ label: string; stopId: number }>; onForecastStopClick: (index: number) => void; onStopClick: (id: number) => void; interactiveNetworkStops: boolean },
}))
vi.mock("@/features/tram-network/components/tram-map", () => ({
  TramMap: (props: NonNullable<typeof mapState.props>) => { mapState.props = props; return <div data-testid="test-map">Карта</div> },
}))
vi.mock("./planning-chart", () => ({ PlanningChart: () => <div>График</div> }))

afterEach(() => { cleanup(); vi.resetAllMocks(); mapState.props = null })
function mount() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><PlanningView /></QueryClientProvider>)
}
const open = (name: string) => fireEvent.click(screen.getByText(name, { selector: "summary" }))

describe("planning workbench", () => {
  it("leads with the map, keeps advanced controls and long lists collapsed, and uses one interval slider", async () => {
    vi.mocked(planningForecast).mockResolvedValue(response)
    mount()
    await screen.findByTestId("test-map")
    expect(screen.getByRole("region", { name: "Прогноз на карте" })).toBeInTheDocument()
    expect(screen.getByRole("slider", { name: /Интервал на карте/ })).toBeInTheDocument()
    expect(screen.queryByLabelText("Интервал сценария · МСК")).not.toBeInTheDocument()
    expect(screen.getByText("Модель и веса", { selector: "summary" }).closest("details")).not.toHaveAttribute("open")
    expect(screen.getByText(/Оценки остановок ·/, { selector: "summary" }).closest("details")).not.toHaveAttribute("open")
    expect(screen.getByText(/Динамика и таблица/, { selector: "summary" }).closest("details")).not.toHaveAttribute("open")
    expect(screen.getByRole("button", { name: "Вся сеть" })).toHaveAttribute("aria-pressed", "false")
    expect(screen.getByText("Опубликованный ряд и происхождение", { selector: "summary" }).closest("details")).not.toHaveAttribute("open")
    open("Опубликованный ряд и происхождение")
    expect(screen.getByRole("link", { name: "Открыть подробный опубликованный ряд" })).toHaveAttribute("href", "/?view=published")
  })

  it("opens model and manual adjustments and updates the map forecast", async () => {
    vi.mocked(planningForecast).mockImplementation((body) => Promise.resolve({ ...response, ...body,
      points: response.points.map((point) => { const weight = body.factors.weather.enabled ? body.factors.weather.multiplier : 0; const ratio = (1 + weight * 1.2) / (1 + weight); return { ...point, scenario: point.baseline * ratio,
        spatial: point.spatial.map((stop) => ({ ...stop, scenario: stop.baseline * ratio })) } }) }))
    mount()
    await screen.findByTestId("test-map")
    open("Модель и веса")
    expect(screen.getByLabelText("Основа расчёта")).toHaveValue("approved")
    expect(screen.queryByRole("option", { name: "Эксперимент с источниками" })).not.toBeInTheDocument()
    expect(screen.getAllByRole("checkbox", { name: "Погода" })).toHaveLength(1)
    expect(screen.queryByLabelText("Вес: Погода")).not.toBeInTheDocument()
    expect(screen.queryByRole("slider", { name: "Ползунок веса: Погода" })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("checkbox", { name: "Погода" }))
    expect(screen.getByLabelText("Вес: Погода")).toBeInTheDocument()
    expect(screen.getByRole("slider", { name: "Ползунок веса: Погода" })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("Вес: Погода"), { target: { value: "1.5" } })
    await waitFor(() => expect(vi.mocked(planningForecast).mock.calls.at(-1)?.[0].factors.weather).toEqual({ enabled: true, multiplier: 1.5 }))
    expect(vi.mocked(planningForecast).mock.calls.at(-1)?.[0].source_enabled.weather).toBe(false)
    expect(await screen.findByText("Сценарий с источниками")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("checkbox", { name: "Погода" }))
    expect(screen.queryByLabelText("Вес: Погода")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("checkbox", { name: "Погода" }))
    fireEvent.change(screen.getByLabelText("Вес: Погода"), { target: { value: "" } })
    expect(screen.getByRole("alert")).toHaveTextContent("Каждый вес")
    fireEvent.click(screen.getByRole("checkbox", { name: "Погода" }))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("Основа расчёта"), { target: { value: "stop_model" } })
    expect(screen.getAllByRole("checkbox", { name: "Трафик" })).toHaveLength(1)
    expect(screen.queryByLabelText("Вес: Трафик")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("checkbox", { name: "Трафик" }))
    expect(screen.getByLabelText("Вес: Трафик")).toBeInTheDocument()
    await waitFor(() => expect(vi.mocked(planningForecast).mock.calls.at(-1)?.[0].source_enabled).toEqual({ weather: false, calendar: false, events: false, traffic: false }))
  })

  it("uses the clicked marker's own stop when a preceding spatial row has no coordinates", async () => {
    const noCoordinates = { ...response.points[0].spatial[0], stop_id: "gtfs:missing", name: "Без координат", latitude: null, longitude: null }
    const target = { ...response.points[0].spatial[0], stop_id: "gtfs:target", name: "Целевая", scenario: 57 }
    vi.mocked(planningForecast).mockResolvedValue({ ...response, points: [{ ...response.points[0], spatial: [noCoordinates, target] }] })
    mount()
    await waitFor(() => expect(mapState.props?.forecastMarkers).toHaveLength(1))
    expect(mapState.props?.forecastMarkers[0].label).toContain("Целевая")
    act(() => mapState.props?.onForecastStopClick(0))
    expect(screen.getByText("Целевая", { selector: ".planning-selection h2" })).toBeInTheDocument()
    expect(screen.getByText("Целевая", { selector: ".planning-selection h2" }).closest("section")).toHaveTextContent("57 посадок")
  })

  it("requests all ten route forecasts only in network mode and explains the export", async () => {
    vi.mocked(planningForecast).mockImplementation((body) => Promise.resolve({ ...response, ...body, route: body.route }))
    mount()
    await screen.findByTestId("test-map")
    expect(vi.mocked(planningForecast)).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "Вся сеть" }))
    await waitFor(() => expect(new Set(vi.mocked(planningForecast).mock.calls.map(([body]) => body.route)).size).toBe(10))
    expect(screen.getByRole("button", { name: "Вся сеть" })).toHaveAttribute("aria-pressed", "true")
    await waitFor(() => expect(screen.getByText(/Загружено 10 из 10 маршрутов/)).toBeInTheDocument())
    open("Скачать CSV")
    expect(screen.getByText(/10 маршрутов · 2025-11-01 · все часы/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Скачать файл" })).toBeEnabled()
  })

  it("inspects an OSM network stop without presenting it as a GTFS boarding estimate", async () => {
    vi.mocked(planningForecast).mockImplementation((body) => Promise.resolve({ ...response, ...body, route: body.route }))
    mount()
    fireEvent.click(screen.getByRole("button", { name: "Вся сеть" }))
    await waitFor(() => expect(mapState.props?.interactiveNetworkStops).toBe(true))
    act(() => mapState.props?.onStopClick(101))
    expect(screen.getByText("Альфа", { selector: ".planning-selection h2" })).toBeInTheDocument()
    expect(screen.getByText(/Для этой точки OSM отдельный прогноз посадок не рассчитан/)).toBeInTheDocument()
  })

  it("exports multiple route snapshots with one CSV header and route identities intact", () => {
    const csv = networkPlanningCsv([{ ...response, route: "1" }, { ...response, route: "5" }], response.points[0].timestamp)
    expect(csv.match(/"run_id","model_version","route"/g)).toHaveLength(1)
    const lines = csv.trim().split("\r\n")
    expect(lines).toHaveLength(5)
    expect(lines.filter((line) => line.includes('"1"'))).toHaveLength(2)
    expect(lines.filter((line) => line.includes('"5"'))).toHaveLength(2)
  })

  it("keeps hot spots collapsed and updates them from scenario weights", async () => {
    vi.mocked(planningForecast).mockImplementation((body) => Promise.resolve({ ...response, ...body,
      points: response.points.map((point) => ({ ...point, spatial: point.spatial.map((stop) => ({ ...stop,
        scenario: 40 * (1 + (body.factors.weather.enabled ? body.factors.weather.multiplier : 0) * 1.5) / (1 + (body.factors.weather.enabled ? body.factors.weather.multiplier : 0)) })) })) }))
    mount()
    await screen.findByTestId("test-map")
    const summary = await screen.findByText(/Горячие точки ·/, { selector: "summary" })
    expect(summary.closest("details")).not.toHaveAttribute("open")
    open("Модель и веса")
    fireEvent.click(screen.getByRole("checkbox", { name: "Погода" }))
    fireEvent.change(screen.getByLabelText("Вес: Погода"), { target: { value: "1.5" } })
    await waitFor(() => expect(screen.getByText("Горячие точки · 1", { selector: "summary" })).toBeInTheDocument())
    fireEvent.click(screen.getByText("Горячие точки · 1", { selector: "summary" }))
    expect(screen.getByRole("region", { name: "Горячие точки" })).toHaveTextContent("Площадь")
    expect(screen.getByRole("region", { name: "Горячие точки" })).toHaveTextContent("52")
  })

  it("keeps the selected range total available behind a disclosure", async () => {
    const hours = ["2025-11-01T07:00:00+03:00", "2025-11-01T08:00:00+03:00", "2025-11-01T09:00:00+03:00"]
    vi.mocked(planningForecast).mockResolvedValue({ ...response, points: hours.map((timestamp, index) => ({ ...response.points[0],
      timestamp, bucket_end: hours[index + 1] ?? "2025-11-01T10:00:00+03:00", baseline: 100 * (index + 1), scenario: 100 * (index + 1) })) })
    mount()
    await screen.findByTestId("test-map")
    await screen.findByText("Сумма за выбранный период", { selector: "summary" })
    open("Сумма за выбранный период")
    const range = screen.getByRole("region", { name: "Сумма за выбранный период" })
    expect(range).toHaveTextContent("Прогноз модели: 600 посадок")
    fireEvent.change(screen.getByLabelText("С · МСК"), { target: { value: hours[1] } })
    expect(range).toHaveTextContent("Прогноз модели: 500 посадок")
    expect(range).toHaveTextContent("Интервалов: 2")
  })

  it("selects a qualitative year month through the timeline and states the CSV scope", async () => {
    const months = Array.from({ length: 12 }, (_, index) => {
      const month = (10 + index) % 12 + 1
      const year = 2025 + Math.floor((10 + index) / 12)
      const timestamp = `${year}-${String(month).padStart(2, "0")}-01T00:00:00+03:00`
      const next = month === 12 ? `${year + 1}-01` : `${year}-${String(month + 1).padStart(2, "0")}`
      return { ...response.points[0], timestamp, bucket_end: `${next}-01T00:00:00+03:00`, basis: index < 2 ? "competition_period" as const : "scenario_projection" as const, scenario: 100 + index * 100 }
    })
    vi.mocked(planningForecast).mockResolvedValue({ ...response, horizon: "year", qualitative: true, points: months })
    mount()
    await screen.findByTestId("test-map")
    fireEvent.change(screen.getByLabelText("Период"), { target: { value: "year" } })
    await waitFor(() => expect(screen.getByRole("slider", { name: /Интервал на карте/ })).toHaveAttribute("max", "11"))
    expect(screen.getByRole("note")).toHaveTextContent("Точность на годовом горизонте не проверена")
    fireEvent.change(screen.getByRole("slider", { name: /Интервал на карте/ }), { target: { value: "2" } })
    expect(screen.getByRole("slider", { name: /Интервал на карте/ })).toHaveAttribute("aria-valuetext", "январь 2026 г.")
    open("Скачать CSV")
    expect(screen.getByText(/Только выбранный на таймлайне месяц/)).toBeInTheDocument()
    expect(planningCsv({ ...response, horizon: "year", points: months }, months[2].timestamp)).not.toContain(months[0].timestamp)
  })
})

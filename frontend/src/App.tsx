import { Activity, GitBranch, Map, Network, TramFront, Waypoints } from "lucide-react"
import { lazy, Suspense, useState } from "react"

import { ApiError } from "@/api/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ForecastExport } from "@/features/forecast/components/forecast-export"
import { ProvenancePanel } from "@/features/forecast/components/provenance-panel"
import { formatForecastValue } from "@/features/forecast/lib/values"
import { dataKind, forecastUnit } from "@/features/forecast/lib/provenance"
import { NetworkMap } from "@/features/forecast/components/network-map"
import { useForecast, useRoutes, useRouteStops } from "@/features/forecast/hooks/use-forecast"
import { bucketLabel, bucketScope, bucketStops, selectedBucket } from "@/features/forecast/lib/bucket"
import { toMoscowInput, validateWindow } from "@/features/forecast/lib/selection"

const horizons: Array<{ value: "day" | "month"; label: string }> = [
  { value: "day", label: "1 день" },
  { value: "month", label: "1 месяц" },
]

const ForecastChart = lazy(() =>
  import("@/features/forecast/components/forecast-chart").then((module) => ({
    default: module.ForecastChart,
  })),
)

const TramNetworkView = lazy(() =>
  import("@/features/tram-network/components/tram-network-view").then((module) => ({
    default: module.TramNetworkView,
  })),
)

const PlanningView = lazy(() => import("@/features/planning/planning-view").then((module) => ({ default: module.PlanningView })))
const ArchitectureView = lazy(() => import("@/features/architecture/architecture-view").then((module) => ({ default: module.ArchitectureView })))
const DecisionLabView = lazy(() => import("@/features/decision-lab/decision-lab-view").then((module) => ({ default: module.DecisionLabView })))

type Section = "forecast" | "published" | "network" | "lab" | "architecture"

const sections: ReadonlyArray<{
  id: Section
  label: string
  icon: typeof Activity
}> = [
  { id: "forecast", label: "Прогноз", icon: Activity },
  { id: "lab", label: "Планирование сети", icon: Waypoints },
  { id: "network", label: "Граф сети", icon: GitBranch },
  { id: "architecture", label: "Как работает Тормоза", icon: Network },
]

function DashboardSkeleton() {
  return (
    <div className="dashboard-skeleton" aria-label="Загрузка прогноза">
      <div className="kpi-grid kpi-grid-pair">{Array.from({ length: 2 }).map((_, index) => <Skeleton key={index} className="h-28" />)}</div>
      <Skeleton className="h-[360px]" />
      <Skeleton className="h-[420px]" />
    </div>
  )
}

function App() {
  const routes = useRoutes()
  const [section, setSection] = useState<Section>(() => {
    const view = new URLSearchParams(globalThis.location.search).get("view")
    return view === "published" || view === "network" || view === "lab" || view === "architecture" ? view : "forecast"
  })
  const [planningYear, setPlanningYear] = useState(false)
  const mapView = section === "forecast"
  const [routeId, setRouteId] = useState<number | null>(null)
  const [horizon, setHorizon] = useState<"day" | "month">("day")
  const selectedRouteId = routeId === null
    ? (routes.data?.[0]?.id ?? null)
    : routes.data?.some((route) => route.id === routeId) ? routeId : null
  const [bucketSelection, setBucketSelection] = useState<{ scope: string; timestamp: string } | null>(null)
  const [pollInterval, setPollInterval] = useState<number | false>(60_000)
  const [stopId, setStopId] = useState<number | null>(null)
  const [startInput, setStartInput] = useState("")
  const [endInput, setEndInput] = useState("")
  const routeStops = useRouteStops(section === "published" ? selectedRouteId : null)
  const window = validateWindow(startInput, endInput, horizon)
  const validStop = stopId === null || Boolean(routeStops.data?.some((stop) => stop.id === stopId))
  const validSelection = !window.error && validStop
  const filters = { stop_id: stopId ?? undefined, start: window.start, end: window.end }
  const forecast = useForecast(selectedRouteId, horizon, filters, validSelection && section === "published", pollInterval)
  const resetWindow = () => { setStartInput(""); setEndInput("") }
  const filtered = stopId !== null || Boolean(startInput || endInput)
  const data = validSelection ? forecast.data : undefined
  const timestamp = data ? selectedBucket(data, bucketSelection) : null
  const currentPoint = data?.points.find((point) => point.timestamp === timestamp)
  const currentStops = data ? bucketStops(data, timestamp) : []
  const selectTimestamp = (value: string) => { if (data) setBucketSelection({ scope: bucketScope(data), timestamp: value }) }
  const queryStatus = forecast.error instanceof ApiError ? forecast.error.status : null
  const selectionError = queryStatus === 404 ? "Нет прогноза для выбранных параметров" : queryStatus === 422 ? "Интервал недоступен" : queryStatus === 409 ? "Данные прогноза несовместимы" : null
  const busiestRow = currentStops.length ? currentStops.reduce((max, row) => row.predicted_passengers > max.predicted_passengers ? row : max) : undefined
  const busiestStop = data?.stops.find((stop) => stop.id === busiestRow?.stop_id)
  const generatedAt = data
    ? new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "Europe/Moscow" }).format(new Date(data.generated_at))
    : "—"

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark"><TramFront /></span><span><b>Тормоза</b><small>ЕДЦ · Москва</small></span></div>
        <nav aria-label="Основная навигация" style={{ minWidth: 0, flexWrap: "wrap", flex: "1 1 auto" }}>
          {sections.map((item) => (
            <button
              key={item.id}
              type="button"
              className={section === item.id ? "active" : undefined}
              aria-current={section === item.id ? "page" : undefined}
              onClick={() => {
                setSection(item.id)
                if (item.id === "forecast") setPlanningYear(false)
                const url = new URL(globalThis.location.href)
                if (item.id !== "forecast") url.searchParams.set("view", item.id)
                else url.searchParams.delete("view")
                globalThis.history.replaceState(null, "", url)
              }}
            >
              <item.icon />
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot" style={section !== "published" ? { gridTemplateColumns: "1fr" } : undefined}>
          {section === "published" && data && <span className="status-dot" aria-hidden="true" />}
          <span>{section === "architecture" ? "Фактическая схема" : section === "network" ? "Граф сети OSM" : section === "lab" ? "Планирование сети" : mapView ? "Прогноз и сценарии" : data ? dataKind(data) : forecast.isError ? "Прогноз недоступен" : "Ожидание данных"}<small>{section === "architecture" ? "Замеры в разделе" : section === "lab" ? "Сценарий выпуска · допущения" : mapView ? "Маршруты и оценка остановок" : data ? `сформирован ${generatedAt} МСК` : "качество модели не подтверждено"}</small></span>
        </div>
      </aside>

      <main>
        {!mapView && <header className="topbar">
          <div><p className="eyebrow">ДИСПЕТЧЕРСКИЙ ЦЕНТР</p><h1>{section === "architecture" ? "Как работает Тормоза" : section === "network" ? "Граф трамвайной сети Москвы" : section === "lab" ? "Планирование сети" : "Пассажиропоток трамвайной сети"}</h1></div>
          <div className="topbar-actions"><Badge>{section === "architecture" ? "Фактическая архитектура" : section === "network" ? "Данные OpenStreetMap" : section === "lab" ? "Решение на основе прогноза" : mapView ? "Оценка и сценарий" : data ? dataKind(data) : "Нет данных"}</Badge></div>
        </header>}

        {section === "network" && (
          <Suspense fallback={<div className="workspace"><Skeleton className="h-[640px]" /></div>}>
            <TramNetworkView />
          </Suspense>
        )}
        {section === "architecture" && <Suspense fallback={<div className="workspace" role="status">Загружаем схему…</div>}><ArchitectureView /></Suspense>}

        {mapView && <Suspense fallback={<div className="workspace"><Skeleton className="h-[640px]" /></div>}><PlanningView key={planningYear ? "year" : "default"} initialHorizon={planningYear ? "year" : "day"} initialRoute={planningYear ? routes.data?.find((route) => route.id === selectedRouteId)?.number : undefined} /></Suspense>}

        {section === "lab" && <Suspense fallback={<div className="workspace"><Skeleton className="h-[640px]" /></div>}><DecisionLabView /></Suspense>}

        <div className="workspace" id="published-forecast" hidden={section !== "published"}>
          <section className="filters" aria-label="Параметры прогноза">
            <div className="filter-field">
              <label htmlFor="route-select">Маршрут</label>
              <Select value={selectedRouteId === null ? "" : String(selectedRouteId)} onValueChange={(value) => { setRouteId(Number(value)); setStopId(null); resetWindow() }}>
                <SelectTrigger id="route-select" disabled={!routes.data?.length}><SelectValue placeholder="Выберите маршрут" /></SelectTrigger>
                <SelectContent>
                  {routes.data?.map((route) => <SelectItem key={route.id} value={String(route.id)}>{route.number} · {route.name}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="filter-field horizon-field">
              <label>Горизонт планирования</label>
              <Tabs value={horizon} onValueChange={(value) => { setHorizon(value as "day" | "month"); resetWindow() }}>
                <TabsList aria-label="Горизонт планирования">{horizons.map((item) => <TabsTrigger key={item.value} value={item.value}>{item.label}</TabsTrigger>)}</TabsList>
                {horizons.map((item) => (
                  <TabsContent key={item.value} value={item.value} className="sr-only">
                    Выбран горизонт: {item.label}
                  </TabsContent>
                ))}
              </Tabs>
              <Button type="button" variant="secondary" disabled={selectedRouteId === null} onClick={() => {
                setPlanningYear(true)
                setSection("forecast")
                const url = new URL(globalThis.location.href)
                url.searchParams.delete("view")
                globalThis.history.replaceState(null, "", url)
              }}>1 год · качественный сценарий</Button>
            </div>
            <div className="filter-field">
              <label htmlFor="stop-select">Остановка</label>
              <Select value={stopId === null ? "all" : String(stopId)} onValueChange={(value) => setStopId(value === "all" ? null : Number(value))}>
                <SelectTrigger id="stop-select" disabled={!routeStops.data?.length}><SelectValue placeholder="Весь маршрут" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Весь маршрут</SelectItem>
                  {routeStops.data?.map((stop) => <SelectItem key={stop.id} value={String(stop.id)}>{stop.name}</SelectItem>)}
                </SelectContent>
              </Select>
              {routeStops.isLoading && <small role="status">Загрузка остановок…</small>}
              {routeStops.isError && <div role="alert">{routeStops.error instanceof ApiError && routeStops.error.status === 409 ? "Для опубликованного ряда нет наблюдаемого каталога остановок. Оценочные GTFS-остановки доступны на вкладке «Прогноз»." : <>Список остановок недоступен. <Button variant="secondary" onClick={() => void routeStops.refetch()}>Повторить остановки</Button></>}</div>}
            </div>
            <div className="filter-field">
              <label htmlFor="window-start">Начало · МСК</label>
              <input id="window-start" type="datetime-local" value={startInput} aria-describedby="window-help" aria-invalid={Boolean(window.error)} onChange={(event) => setStartInput(event.target.value)} />
            </div>
            <div className="filter-field">
              <label htmlFor="window-end">Конец (не включительно) · МСК</label>
              <input id="window-end" type="datetime-local" value={endInput} aria-describedby="window-help" aria-invalid={Boolean(window.error)} onChange={(event) => setEndInput(event.target.value)} />
            </div>
            <div className="filter-field"><label htmlFor="poll-interval">Автопроверка</label><select id="poll-interval" value={pollInterval === false ? "off" : String(pollInterval)} onChange={(event) => setPollInterval(event.target.value === "off" ? false : Number(event.target.value))}><option value="off">Вручную</option><option value="15000">Каждые 15 секунд</option><option value="60000">Каждую минуту</option><option value="300000">Каждые 5 минут</option></select></div>
            <div className="freshness"><span>Прогноз сформирован · МСК</span><strong>{generatedAt}</strong></div>
            <Button variant="secondary" disabled={selectedRouteId === null || !validSelection || forecast.isFetching} onClick={() => void forecast.refetch()}>Обновить прогноз</Button>
          </section>

          <ForecastExport snapshot={data} updatedAt={forecast.dataUpdatedAt} failed={forecast.isRefetchError} pollInterval={pollInterval} selectedTimestamp={timestamp} requestedFilters={filters} />
          <div className="window-help" id="window-help">
            <p>{window.error ?? "Время Europe/Moscow. Пустые даты — опубликованный интервал целиком; конец не включается."}</p>
            {window.error && <span role="alert">Запрос не выполнен: исправьте интервал.</span>}
            {data?.selection && <p>Показано: {toMoscowInput(data.selection.start).replace("T", " ")} — {toMoscowInput(data.selection.end).replace("T", " ")} МСК (конец не включён).</p>}
            {!validStop && !routeStops.isLoading && <p role="alert">Остановка недоступна в текущем списке. Сбросьте выбор или повторите загрузку остановок.</p>}
            {filtered && <Button variant="secondary" onClick={() => { setStopId(null); resetWindow() }}>Сбросить остановку и интервал</Button>}
          </div>
          {(routes.isLoading || forecast.isLoading) && <DashboardSkeleton />}
          {routes.isError && (
            <Card className="error-state"><CardContent><h2>{routes.data ? "Список маршрутов не обновлён" : "Маршруты временно недоступны"}</h2><p>Повторите загрузку списка маршрутов.</p><Button onClick={() => void routes.refetch()}>Повторить загрузку маршрутов</Button></CardContent></Card>
          )}
          {forecast.isError && validSelection && selectedRouteId !== null && (
            <Card className="error-state"><CardContent><h2>{data ? "Показан сохранённый прогноз" : selectionError ?? "Прогноз временно недоступен"}</h2><p>{data ? `Не удалось обновить данные. Прогноз сформирован ${generatedAt} МСК; данные могут быть устаревшими.` : selectionError ? "Измените остановку или сократите интервал; можно сбросить фильтры." : "Проверьте соединение с API и повторите запрос."}</p><Button onClick={() => void forecast.refetch()}>Повторить прогноз</Button></CardContent></Card>
          )}
          {routes.data?.length === 0 && !routes.isLoading && (
            <Card className="error-state"><CardContent><h2>Маршруты не найдены</h2><p>Загрузите сетевой граф и опубликуйте прогноз.</p></CardContent></Card>
          )}
          {data && data.points.length === 0 && (
            <Card className="error-state"><CardContent><h2>Нет точек прогноза</h2><p>Для выбранной остановки и временного интервала нет значений. Измените или сбросьте фильтры.</p></CardContent></Card>
          )}
          {data && data.points.length > 0 && (
            <>
              <section className="kpi-grid kpi-grid-pair" aria-label="Ключевые показатели">
                <article className="kpi"><span>Пиковый поток</span><strong>{formatForecastValue(data.peak_passengers)}</strong><small>{forecastUnit(data)} · значение интервала</small></article>
                {busiestStop && busiestRow && <article className="kpi"><span>Напряжённый узел</span><strong className="kpi-text">{busiestStop?.name ?? "—"}</strong><small>{busiestRow ? `${formatForecastValue(busiestRow.predicted_passengers)} в выбранном интервале` : "нет данных выбранного интервала"}</small></article>}
              </section>

              <section className="bucket-selection" aria-label="Выбранный интервал прогноза">
                <div>
                  <label htmlFor="bucket-select">Интервал на графике и карте · МСК</label>
                  <select id="bucket-select" value={timestamp ?? ""} onChange={(event) => selectTimestamp(event.target.value)}>
                    {data.points.map((point) => <option key={point.timestamp} value={point.timestamp}>{bucketLabel(point.timestamp)}</option>)}
                  </select>
                  <div className="bucket-step-controls">
                    <Button variant="secondary" disabled={!timestamp || data.points[0].timestamp === timestamp} onClick={() => {
                      const index = data.points.findIndex((point) => point.timestamp === timestamp)
                      if (index > 0) selectTimestamp(data.points[index - 1].timestamp)
                    }}>Предыдущий интервал</Button>
                    <Button variant="secondary" disabled={!timestamp || data.points.at(-1)?.timestamp === timestamp} onClick={() => {
                      const index = data.points.findIndex((point) => point.timestamp === timestamp)
                      if (index >= 0 && index + 1 < data.points.length) selectTimestamp(data.points[index + 1].timestamp)
                    }}>Следующий интервал</Button>
                  </div>
                </div>
                <div aria-live="polite" data-testid="current-forecast-value"><span>Прогноз выбранного интервала</span><strong>{currentPoint ? formatForecastValue(currentPoint.predicted_passengers) : "—"}</strong><span>{forecastUnit(data)}</span></div>
              </section>
              <Suspense fallback={<Skeleton className="h-[360px]" />}>
                <ForecastChart snapshot={data} selectedTimestamp={timestamp} onSelectTimestamp={selectTimestamp} />
              </Suspense>
              <NetworkMap snapshot={data} timestamp={timestamp} selectedStopId={stopId} onStopSelect={setStopId} />
              <ProvenancePanel forecast={data} updatedAt={forecast.dataUpdatedAt} failed={forecast.isRefetchError} fetching={forecast.isFetching} pollInterval={pollInterval} />
              <footer className="data-note"><Map />{dataKind(data)} · {data.run?.synthetic ? "демонстрационные данные; качество на реальных данных не подтверждено" : "оценка качества и ограничения — в происхождении прогноза"} · версия {data.model_version}</footer>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App

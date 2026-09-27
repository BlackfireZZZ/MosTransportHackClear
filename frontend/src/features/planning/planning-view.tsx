import { Download, MapPinned, SlidersHorizontal } from "lucide-react"
import { lazy, Suspense, useMemo, useState } from "react"
import { factorIds, type FactorId, type PlanningRequest, type PlanningResponse } from "@/api/planning"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { useNetworkGeoJson, useStopDetail } from "@/features/tram-network/hooks/use-tram-network"
import { stopLabel } from "@/features/tram-network/lib/network"
import { flowAppearance } from "./flow-color"
import { RateLegend } from "./rate-legend"
import { PlanningEvaluation } from "./evaluation"
import { Timeline } from "./timeline"
import { directionBearings, directionNames } from "./direction"
import { InsightsPanel } from "./insights-panel"
import { RangeTotalsPanel } from "./range-totals"
import { Applicability } from "./applicability"
import { usePlanning } from "./use-planning"
import { NETWORK_ROUTES, useNetworkPlanning } from "./use-network-planning"
import { useTweenedValues } from "./use-tween"
import { downloadPlanningCsv, networkPlanningCsv, planningCsv } from "./export"
import "./planning.css"

const PlanningChart = lazy(() => import("./planning-chart").then((module) => ({ default: module.PlanningChart })))
const TramMap = lazy(() => import("@/features/tram-network/components/tram-map").then((module) => ({ default: module.TramMap })))
const labels: Record<FactorId, string> = { weather: "Погода", calendar: "Календарь", events: "Новости и инциденты", traffic: "Трафик" }
const modeDetails = {
  stop_model: "Прямой прогноз по остановкам. Направление и место посадки восстановлены по расписанию.",
  approved: "Основа — конкурсный прогноз с лучшим публичным результатом. Включённые источники создают отдельный сценарий без публичного score.",
}
const number = (value: number) => new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 }).format(value)
const validMultiplier = (value: number) => Number.isFinite(value) && value >= 0 && value <= 3
const rateText = (count: number, start: string, end: string) => {
  const rate = flowAppearance(count, start, end).rate
  return rate === null ? "—" : number(rate)
}
const bucketFormats: Record<PlanningRequest["horizon"], Intl.DateTimeFormat> = {
  day: new Intl.DateTimeFormat("ru-RU", { timeZone: "Europe/Moscow", dateStyle: "short", timeStyle: "short" }),
  month: new Intl.DateTimeFormat("ru-RU", { timeZone: "Europe/Moscow", dateStyle: "short" }),
  year: new Intl.DateTimeFormat("ru-RU", { timeZone: "Europe/Moscow", month: "long", year: "numeric" }),
}
const initialRequest: PlanningRequest = {
  forecast_mode: "approved", source_enabled: { weather: false, calendar: false, events: false, traffic: false },
  route: "1", start_date: "2025-11-01", horizon: "day", stop_id: null, direction: null,
  factors: { weather: { enabled: false, multiplier: 1 }, calendar: { enabled: false, multiplier: 1 }, events: { enabled: false, multiplier: 1 }, traffic: { enabled: false, multiplier: 1 } },
}
type MapStop = { route: string; stop: PlanningResponse["points"][number]["spatial"][number]; timestamp: string; end: string; bearing: number | undefined }

export function PlanningView({ initialHorizon = "day", initialRoute }: { initialHorizon?: PlanningRequest["horizon"]; initialRoute?: string }) {
  const [request, setRequest] = useState<PlanningRequest>(() => ({ ...initialRequest, horizon: initialHorizon, route: initialRoute ?? initialRequest.route }))
  const [networkMode, setNetworkMode] = useState(false)
  const [bucket, setBucket] = useState("")
  const [inspectedKey, setInspectedKey] = useState("")
  const [osmStopId, setOsmStopId] = useState<number | null>(null)
  const dateValid = /^2025-(11|12)-\d{2}$/.test(request.start_date) && (request.horizon !== "year" || request.start_date.endsWith("-01"))
  const factorsValid = factorIds.every((id) => validMultiplier(request.factors[id].multiplier))
  const weightedScenario = factorIds.some((id) => request.factors[id].enabled && request.factors[id].multiplier > 0)
  const query = usePlanning(request, dateValid && factorsValid && !networkMode)
  const networkForecasts = useNetworkPlanning(request, networkMode && dateValid && factorsValid)
  const graph = useNetworkGeoJson()
  const osmDetail = useStopDetail(networkMode ? osmStopId : null)
  const snapshot = !networkMode && dateValid && factorsValid && !query.settling ? query.data : undefined
  const responses = networkMode ? networkForecasts.responses : snapshot ? [snapshot] : []
  const points = responses[0]?.points ?? []
  const point = points.find((item) => item.timestamp === bucket) ?? points.reduce((peak, item) => item.scenario > peak.scenario ? item : peak, points[0])
  const visiblePoints = snapshot?.horizon === "year" ? (point ? [point] : []) : (snapshot?.points ?? [])
  const directionName = directionNames(snapshot?.stops ?? [])
  const mapStops: MapStop[] = responses.flatMap((response) => {
    const current = response.points.find((item) => item.timestamp === point?.timestamp)
    if (!current) return []
    const bearing = directionBearings(response.stops)
    return current.spatial.flatMap((stop) => stop.latitude === null || stop.longitude === null ? [] : [{ route: response.route, stop, timestamp: current.timestamp, end: current.bucket_end, bearing: bearing(stop.stop_id, stop.direction) }])
  })
  const targetValues = useMemo(() => new Map(point?.spatial.map((stop) => [`${stop.stop_id}|${stop.direction}`, stop.scenario]) ?? []), [point])
  const shownValues = useTweenedValues(targetValues, 900)
  const markers = mapStops.map(({ route, stop, timestamp, end, bearing }, index) => {
    const value = networkMode ? stop.scenario : shownValues.get(`${stop.stop_id}|${stop.direction}`) ?? stop.scenario
    return { stopId: index, latitude: stop.latitude!, longitude: stop.longitude!, value, color: flowAppearance(value, timestamp, end).color,
      label: `Маршрут ${route} · ${stop.name} · ${number(stop.scenario)} посадок · оценка`,
      selected: inspectedKey === `${route}|${stop.stop_id}|${stop.direction}`, synthetic: false, bearing }
  })
  const inspected = mapStops.find(({ route, stop }) => inspectedKey === `${route}|${stop.stop_id}|${stop.direction}`)
  const time = (value: string) => bucketFormats[networkMode ? "day" : snapshot?.horizon ?? request.horizon].format(new Date(value))
  const update = (patch: Partial<PlanningRequest>) => setRequest((current) => ({ ...current, ...patch }))
  const loading = networkMode ? networkForecasts.loading : (query.settling || query.isLoading) && dateValid && factorsValid
  const failed = networkMode ? networkForecasts.failed : query.isError ? 1 : 0
  const exportScope = networkMode ? `10 маршрутов · ${request.start_date} · все часы`
    : request.horizon === "year" ? `маршрут ${request.route} · ${point ? time(point.timestamp) : "выбранный месяц"}`
      : `маршрут ${request.route} · ${request.horizon === "day" ? "все часы дня" : "все дни месяца"}`
  const exportCsv = () => {
    if (networkMode && responses.length === NETWORK_ROUTES.length) downloadPlanningCsv(networkPlanningCsv(responses, point?.timestamp ?? ""), "network")
    else if (snapshot) downloadPlanningCsv(planningCsv(snapshot, point?.timestamp ?? ""), snapshot.route)
  }

  return <div className="workspace planning" id="forecast">
    <div className="planning-heading">
      <div><p className="planning-overline">ПРОГНОЗ / МОСКВА</p><h1>Посадки на трамвайной сети</h1><p>Оценка спроса по времени и остановкам. Время — московское.</p></div>
      <div className="planning-scope" role="group" aria-label="Масштаб карты">
        <button type="button" aria-pressed={!networkMode} onClick={() => { setNetworkMode(false); setOsmStopId(null) }}>Маршрут</button>
        <button type="button" aria-pressed={networkMode} onClick={() => { setNetworkMode(true); setBucket(""); setInspectedKey(""); setOsmStopId(null); update({ horizon: "day", stop_id: null, direction: null }) }}>Вся сеть</button>
      </div>
    </div>
    <div className="planning-workbench">
      <section className="planning-map-panel" aria-label="Прогноз на карте">
        <div className="planning-map-topline"><div><MapPinned aria-hidden="true" /><span>{networkMode ? "Все маршруты · сеть OSM" : `Маршрут ${request.route} · ${request.horizon === "year" ? "годовой сценарий" : request.horizon === "month" ? "по дням" : "по часам"}`}</span></div><span className="planning-map-date">{point ? time(point.timestamp) : request.start_date}</span></div>
        {networkMode && <p className="planning-map-context">Линии и серые остановки — сеть OSM. Цветные точки — оценка посадок на маршрутах датасета. Идентификаторы OSM и GTFS не приравнены.</p>}
        {request.horizon === "year" && !networkMode && <p className="planning-year-note" role="note">2026 — качественный сценарий. Точность на годовом горизонте не проверена.</p>}
        {loading && <div className="planning-map-loading" role="status">Считаем прогноз…</div>}
        {failed > 0 && <div className="planning-map-error" role="alert"><span>{networkMode ? `Не загружено маршрутов: ${failed} из 10.` : "Прогноз недоступен."}</span><Button variant="secondary" onClick={() => void (networkMode ? networkForecasts.retry() : query.refetch())}>Повторить</Button></div>}
        {point && <Timeline points={points} value={point.timestamp} format={time} onChange={setBucket} />}
        <Suspense fallback={<Skeleton className="h-[500px]" />}><TramMap network={networkMode ? graph.data : undefined} routeGeoJson={undefined} selectedRoute={null} path={undefined} fromStop={null} toStop={null} selectedStop={networkMode ? osmDetail.data ?? null : null} onStopClick={(id) => { setInspectedKey(""); setOsmStopId(id) }} interactiveNetworkStops={networkMode} fitForecastMarkers forecastMarkers={markers} onForecastStopClick={(index) => { const row = mapStops[index]; if (row) { setOsmStopId(null); setInspectedKey(`${row.route}|${row.stop.stop_id}|${row.stop.direction}`) } }} /></Suspense>
        {networkMode && graph.isError && <p className="planning-map-context" role="status">Схема сети недоступна. Оценки GTFS показаны без линий OSM.</p>}
        <RateLegend />
      </section>
      <aside className="planning-side" aria-label="Параметры и выбранная остановка">
        <section className="planning-side-card">
          <h2>Показать на карте</h2>
          <div className="planning-primary-controls">
            {!networkMode && <label>Маршрут<select value={request.route} onChange={(event) => { setInspectedKey(""); update({ route: event.target.value, stop_id: null, direction: null }) }}>{(snapshot?.routes ?? [...NETWORK_ROUTES]).map((route) => <option key={route}>{route}</option>)}</select></label>}
            {!networkMode && <label>Период<select value={request.horizon} onChange={(event) => { setBucket(""); update({ horizon: event.target.value as PlanningRequest["horizon"], start_date: event.target.value === "year" ? request.start_date.slice(0, 7) + "-01" : request.start_date }) }}><option value="day">День · часы</option><option value="month">Месяц · дни</option><option value="year">Год · месяцы</option></select></label>}
            <label>Начальная дата · МСК<input type="date" min="2025-11-01" max="2025-12-31" value={request.start_date} aria-invalid={!dateValid} onChange={(event) => { setBucket(""); update({ start_date: event.target.value }) }} /></label>
          </div>
          {!dateValid && <p className="planning-warning" role="alert">Нужна дата ноября–декабря 2025. Год начинается с первого числа месяца.</p>}
          {!networkMode && <details className="planning-settings"><summary><SlidersHorizontal aria-hidden="true" /> Направление и остановка</summary><div className="planning-settings-body">
            <label>Направление · оценка<select value={request.direction ?? ""} onChange={(event) => update({ direction: event.target.value || null, stop_id: null })}><option value="">Все направления</option>{["0", "1"].map((direction) => <option key={direction} value={direction}>{directionName(direction)}</option>)}</select></label>
            <label>Остановка · оценка<select value={request.stop_id ?? ""} disabled={!snapshot?.stops.length} onChange={(event) => update({ stop_id: event.target.value || null })}><option value="">Все остановки</option>{Array.from(new Map(snapshot?.stops.filter((stop) => request.direction === null || stop.direction === request.direction).map((stop) => [stop.stop_id, stop])).values()).map((stop) => <option key={stop.stop_id} value={stop.stop_id}>{stop.name} {directionName(stop.direction)}</option>)}</select></label>
            <p>Привязка к остановке и направлению расчётная.</p>
          </div></details>}
        </section>
        <section className="planning-side-card planning-selection" aria-live="polite">
          <p className="planning-card-label">{inspected || osmStopId !== null ? "ВЫБРАННАЯ ОСТАНОВКА" : "ТЕКУЩИЙ ИНТЕРВАЛ"}</p>
          {inspected ? <><h2>{inspected.stop.name}</h2><p>Маршрут {inspected.route} · {directionNames(responses.find((item) => item.route === inspected.route)?.stops ?? [])(inspected.stop.direction)}</p><strong>{number(inspected.stop.scenario)} <small>посадок · оценка</small></strong><p>{rateText(inspected.stop.scenario, inspected.timestamp, inspected.end)} посадок/ч · {inspected.stop.stop_id}</p><Button variant="secondary" onClick={() => setInspectedKey("")}>К обзору интервала</Button></> : osmStopId !== null ? <>{osmDetail.isPending && <p role="status">Загрузка остановки OSM…</p>}{osmDetail.isError && <p role="alert">Остановка OSM недоступна. <Button variant="secondary" onClick={() => void osmDetail.refetch()}>Повторить</Button></p>}{osmDetail.data && <><h2>{stopLabel(osmDetail.data)}</h2><p>OSM {osmDetail.data.id} · маршруты {osmDetail.data.routes.join(", ") || "не указаны"}</p><p>Для этой точки OSM отдельный прогноз посадок не рассчитан. Цветные точки GTFS показаны отдельно.</p></>}<Button variant="secondary" onClick={() => setOsmStopId(null)}>К обзору интервала</Button></> : <><h2>{point ? time(point.timestamp) : "Выберите период"}</h2><strong>{number(networkMode ? responses.reduce((sum, item) => sum + (item.points.find((row) => row.timestamp === point?.timestamp)?.scenario ?? 0), 0) : point?.scenario ?? 0)} <small>посадок</small></strong><p>{networkMode ? `Загружено ${responses.length} из 10 маршрутов · ${mapStops.length} оценок остановок` : `Не распределено по остановкам: ${number(point?.route_unallocated_scenario ?? 0)}`}</p></>}
        </section>
        <section className="planning-side-card planning-adjustments"><details className="planning-settings"><summary><SlidersHorizontal aria-hidden="true" /> Модель и веса</summary><div className="planning-settings-body">
          <label>Основа расчёта<select value={request.forecast_mode} onChange={(event) => update({ forecast_mode: event.target.value as "approved" | "stop_model" })}><option value="approved">Опубликованный маршрутный прогноз</option><option value="stop_model">Оценка по остановкам</option></select></label>
          <p>{modeDetails[request.forecast_mode === "stop_model" ? "stop_model" : "approved"]}</p>
          <fieldset className="planning-factors"><legend>Веса сценарных моделей</legend>{factorIds.map((id) => <div key={id}>
            <label className="planning-switch"><input type="checkbox" checked={request.factors[id].enabled} onChange={(event) => update({ factors: { ...request.factors, [id]: { enabled: event.target.checked, multiplier: validMultiplier(request.factors[id].multiplier) ? request.factors[id].multiplier : 1 } } })} />{labels[id]}</label>
            {request.factors[id].enabled && <><label>Вес: {labels[id]}<input type="number" min="0" max="3" step="0.05" value={Number.isNaN(request.factors[id].multiplier) ? "" : request.factors[id].multiplier} onChange={(event) => update({ factors: { ...request.factors, [id]: { ...request.factors[id], multiplier: event.target.value === "" ? NaN : Number(event.target.value) } } })} /></label>
            <input type="range" className="planning-factor-slider" aria-label={`Ползунок веса: ${labels[id]}`} aria-valuetext={`${Number.isNaN(request.factors[id].multiplier) ? 1 : request.factors[id].multiplier}`} min="0" max="3" step="0.05" value={Number.isNaN(request.factors[id].multiplier) ? 1 : Math.min(3, Math.max(0, request.factors[id].multiplier))} onChange={(event) => update({ factors: { ...request.factors, [id]: { ...request.factors[id], multiplier: Number(event.target.value) } } })} /></>}
          </div>)}</fieldset>
          <p>Источники выключены, пока вы их не выбрали. У основы вес 1; при погоде 2 и новостях 1 доли станут 25%, 50% и 25%. Веса нормируются на сервере. Ветки источников не отправлялись на лидерборд.</p>
          {!factorsValid && <p className="planning-warning" role="alert">Каждый вес должен быть от 0 до 3.</p>}
        </div></details></section>
        <details className="planning-export"><summary><Download aria-hidden="true" /> Скачать CSV</summary><div><strong>{exportScope}</strong><p>{networkMode ? "Строки всех интервалов и остановок сгруппированы по маршрутам; доступно после загрузки всей сети." : request.horizon === "year" ? "Только выбранный на таймлайне месяц." : "Все интервалы выбранного периода."} В файле указаны прогноз с выбранными весами, сами веса, версия и происхождение.</p><Button disabled={networkMode ? responses.length !== NETWORK_ROUTES.length : !snapshot || query.isFetching} onClick={exportCsv}>Скачать файл</Button></div></details>
      </aside>
    </div>
    {snapshot && <section className="planning-results" aria-label="Подробности прогноза">
      <div className="planning-results-heading"><div><p className="planning-overline">ДАННЫЕ И ПРОВЕРКА</p><h2>Подробности прогноза</h2></div><span>{weightedScenario ? "Сценарий с источниками" : snapshot.forecast_mode === "approved" ? "Конкурсный снимок" : "Остановочная модель"}</span></div>
      {request.forecast_mode === "approved" && <details className="planning-detail planning-publication"><summary>Опубликованный ряд и происхождение</summary><div className="planning-publication-body">
        <dl><div><dt>Версия модели</dt><dd>{snapshot.model_version}</dd></div><div><dt>Сформирован · МСК</dt><dd>{new Intl.DateTimeFormat("ru-RU", { dateStyle: "medium", timeStyle: "short", timeZone: "Europe/Moscow" }).format(new Date(snapshot.generated_at))}</dd></div><div><dt>SHA-256 конкурсного CSV</dt><dd>{snapshot.provenance.csv_sha256 ?? "Недоступна"}</dd></div></dl>
        <p>Активная публикация в базе, интервалы неопределённости, время последней проверки и отдельный экспорт доступны в подробном ряду.</p>
        <a href="/?view=published">Открыть подробный опубликованный ряд</a>
      </div></details>}
      <InsightsPanel directionName={directionName} points={visiblePoints} selected={point?.timestamp} format={time} number={number} onSelect={setBucket} />
      <div className="planning-detail-grid"><details className="planning-detail"><summary>Сумма за выбранный период</summary><RangeTotalsPanel key={`${snapshot.route}-${snapshot.horizon}-${snapshot.start_date}-${snapshot.horizon === "year" ? point?.timestamp : "all"}`} points={visiblePoints} filtered={snapshot.stop_id !== null || snapshot.direction !== null} format={time} number={number} /></details>
      <details className="planning-detail"><summary>Оценки остановок · {point?.spatial.length ?? 0}</summary>{!point?.spatial.length ? <p>Для этого интервала нет оценок остановок. Маршрутный итог сохранён.</p> : <div className="forecast-stop-table" tabIndex={0} role="region" aria-label="Оценки остановок сценария"><table><thead><tr><th>Остановка GTFS</th><th>Направление · оценка</th><th>Прогноз модели</th><th>С весами</th><th>Посадок/ч</th></tr></thead><tbody>{point.spatial.map((stop) => <tr key={`${stop.stop_id}-${stop.direction}`}><th scope="row">{stop.name} <small>{stop.stop_id}</small></th><td>{directionName(stop.direction)}</td><td>{number(stop.baseline)}</td><td>{number(stop.scenario)}</td><td>{rateText(stop.scenario, point.timestamp, point.bucket_end)}</td></tr>)}</tbody></table></div>}</details></div>
      <details className="planning-detail"><summary>Динамика и таблица · интервалов: {visiblePoints.length}</summary>
        {snapshot.warnings.length > 0 && <p className="planning-warning">{snapshot.warnings.join(" ")}</p>}
        {visiblePoints.length === 0 ? <p>Нет прогноза для выбранных параметров. <Button onClick={() => update(initialRequest)}>Сбросить</Button></p> : <><Suspense fallback={<Skeleton className="h-64" />}><PlanningChart points={visiblePoints} /></Suspense><div className="forecast-stop-table" tabIndex={0} role="region" aria-label="Таблица прогноза модели и прогноза с весами"><table><caption>Конец интервала не включён</caption><thead><tr><th>Начало · МСК</th><th>Прогноз модели</th><th>С весами</th><th>Изменение</th><th>Не распределено</th></tr></thead><tbody>{visiblePoints.map((item) => <tr key={item.timestamp}><th scope="row"><button className="planning-bucket" aria-pressed={item.timestamp === point?.timestamp} onClick={() => setBucket(item.timestamp)}>{time(item.timestamp)}</button></th><td>{number(item.baseline)}</td><td>{number(item.scenario)}</td><td>{number(item.scenario - item.baseline)}</td><td>{number(item.unallocated_scenario)}</td></tr>)}</tbody></table></div></>}
      </details>
      <details className="planning-detail planning-method"><summary>Метод, ограничения и источники</summary><Applicability points={visiblePoints} route={snapshot.route} /><p>Направление и остановка — оценки по расписанию. Наличие источника или сценарный вес не доказывают прирост точности.</p><div className="planning-source-grid">{snapshot.sources.map((source) => <article key={source.id}><h3>{source.label}</h3><p>{source.detail}</p>{typeof source.url === "string" && /^https?:\/\//.test(source.url) && <a href={source.url} target="_blank" rel="noreferrer">Источник ↗</a>}</article>)}</div>{snapshot.experimental_evaluation && <PlanningEvaluation raw={snapshot.experimental_evaluation} />}<details><summary>Происхождение расчёта · {snapshot.run_id}</summary><dl>{Object.entries(snapshot.provenance).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl></details></details>
    </section>}
  </div>
}

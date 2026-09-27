import { ArrowDownRight, ArrowRight, ChevronDown, ChevronUp, CircleHelp, Download, TramFront } from "lucide-react"
import { useMemo, useRef, useState, type PointerEvent } from "react"
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import type { PlanningResponse } from "@/api/planning"
import { planningCsv, downloadPlanningCsv } from "@/features/planning/export"
import { DEFAULT_WINDOW_HOURS, pressureTier, rankRoutes, ROUTES, scheduleWindowUsable, simulateService, type RoutePriority } from "./model"
import { serviceEstimate, type ServiceEstimate } from "./service-estimates"
import { useDecisionLab } from "./use-decision-lab"
import "./decision-lab.css"

const integer = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 })
const decimal = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 })
const clock = (timestamp: string) => new Intl.DateTimeFormat("ru-RU", { timeZone: "Europe/Moscow", hour: "2-digit", minute: "2-digit" }).format(new Date(timestamp))
const dateLabel = (day: string) => new Intl.DateTimeFormat("ru-RU", { timeZone: "Europe/Moscow", day: "numeric", month: "long", year: "numeric" }).format(new Date(`${day}T12:00:00+03:00`))
const hourLabel = (hour: number) => `${String(hour).padStart(2, "0")}:00`
const span = (_route: RoutePriority, start: number, end: number) => `${hourLabel(start)}–${hourLabel(end)}`

function DemandChart({ route, start, end }: { route: RoutePriority; start: number; end: number }) {
  const data = route.points.map((point) => ({ timestamp: point.timestamp, label: clock(point.timestamp), boardings: point.route_baseline }))
  return <div className="lab-chart" role="img" aria-label={`Почасовой прогноз маршрута ${route.route}; выбранное окно ${span(route, start, end)}; точные значения в таблице ниже`}>
    <ResponsiveContainer width="100%" height="100%"><AreaChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: -12 }}>
      <defs><linearGradient id="labDemandFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#d9342b" stopOpacity={0.22} /><stop offset="100%" stopColor="#d9342b" stopOpacity={0.015} /></linearGradient></defs>
      <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="2 5" />
      <XAxis dataKey="label" axisLine={false} tickLine={false} interval={3} tickMargin={10} />
      <YAxis axisLine={false} tickLine={false} width={56} tickFormatter={(value: number) => integer.format(value)} />
      <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 6 }} formatter={(value) => [`${integer.format(Number(value))} посадок`, "Прогноз"]} />
      <ReferenceArea x1={data[start].label} x2={data[Math.min(end, 23)].label} fill="#d9342b" fillOpacity={0.07} strokeOpacity={0} />
      <Area type="monotone" dataKey="boardings" stroke="var(--primary)" strokeWidth={2.5} fill="url(#labDemandFill)" dot={false} activeDot={{ r: 5 }} isAnimationActive={false} />
    </AreaChart></ResponsiveContainer>
  </div>
}

function WindowTimeline({ route, start, end, onChange }: { route: RoutePriority; start: number; end: number; onChange: (start: number, end: number) => void }) {
  const rail = useRef<HTMLDivElement>(null)
  const dragging = useRef<"start" | "end" | null>(null)
  const move = (event: PointerEvent) => {
    if (!dragging.current || !rail.current) return
    const bounds = rail.current.getBoundingClientRect()
    const hour = Math.round((event.clientX - bounds.left) / bounds.width * 24)
    if (dragging.current === "start") onChange(Math.max(0, Math.min(hour, end - 1)), end)
    else onChange(start, Math.min(24, Math.max(hour, start + 1)))
  }
  const handleKey = (side: "start" | "end", key: string) => {
    if (key === "Home") { if (side === "start") onChange(0, end); else onChange(start, start + 1); return true }
    if (key === "End") { if (side === "start") onChange(end - 1, end); else onChange(start, 24); return true }
    const step = key === "ArrowLeft" ? -1 : key === "ArrowRight" ? 1 : 0
    if (!step) return false
    if (side === "start") onChange(Math.max(0, Math.min(start + step, end - 1)), end)
    else onChange(start, Math.min(24, Math.max(end + step, start + 1)))
    return true
  }
  return <div className="lab-timeline" aria-label="Выбор часового окна">
    <div className="lab-timeline-heading"><strong>Окно анализа</strong><span>{span(route, start, end)} · {end - start} ч</span></div>
    <div className="lab-time-rail" ref={rail}>
      {route.points.map((point) => <span key={point.timestamp} className="lab-time-hour" style={{ height: `${Math.max(8, point.route_baseline / Math.max(route.peakBoardings, 1) * 100)}%` }} aria-hidden="true" />)}
      <div className="lab-time-selection" style={{ left: `${start / 24 * 100}%`, width: `${(end - start) / 24 * 100}%` }} aria-hidden="true" />
      {(["start", "end"] as const).map((side) => <button key={side} type="button" role="slider" className={`lab-range-handle lab-range-handle--${side}`}
        style={{ left: `${(side === "start" ? start : end) / 24 * 100}%` }}
        aria-label={side === "start" ? "Начало окна" : "Конец окна"} aria-valuemin={side === "start" ? 0 : start + 1}
        aria-valuemax={side === "start" ? end - 1 : 24} aria-valuenow={side === "start" ? start : end}
        aria-valuetext={hourLabel(side === "start" ? start : end)}
        onPointerDown={(event) => { dragging.current = side; event.currentTarget.setPointerCapture(event.pointerId) }}
        onPointerMove={move} onPointerUp={() => { dragging.current = null }} onPointerCancel={() => { dragging.current = null }}
        onKeyDown={(event) => { if (handleKey(side, event.key)) event.preventDefault() }} />)}
    </div>
    <div className="lab-time-ticks"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>24:00</span></div>
    <p>Перетяните края выделения или используйте стрелки на клавиатуре · шаг 1 час</p>
  </div>
}

function ScenarioChart({ hours }: { hours: { timestamp: string; before: number; after: number }[] }) {
  const data = hours.map((hour) => ({ ...hour, label: clock(hour.timestamp) }))
  return <div className="lab-scenario-chart" role="img" aria-label="Расчётные посадки на отправление до и после изменения; точные значения в таблице ниже">
    <ResponsiveContainer width="100%" height="100%"><BarChart data={data} barGap={5} margin={{ top: 6, right: 10, bottom: 0, left: -14 }}>
      <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="2 5" />
      <XAxis dataKey="label" axisLine={false} tickLine={false} /><YAxis axisLine={false} tickLine={false} width={55} />
      <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 6 }} formatter={(value) => [decimal.format(Number(value)), "посадок на отправление"]} />
      <Bar dataKey="before" name="Сейчас, по допущению" fill="#59636a" radius={[3, 3, 0, 0]} isAnimationActive={false} />
      <Bar dataKey="after" name="С добавленным вагоном" fill="#246b88" radius={[3, 3, 0, 0]} isAnimationActive={false} />
    </BarChart></ResponsiveContainer>
  </div>
}

export function DecisionLabView() {
  const [day, setDay] = useState("2025-11-01")
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null)
  const [windowHours, setWindowHours] = useState(DEFAULT_WINDOW_HOURS)
  const [selectedWindow, setSelectedWindow] = useState<{ route: string; start: number; end: number } | null>(null)
  const [headwayOverride, setHeadwayOverride] = useState<number | null>(null)
  const [cycleOverride, setCycleOverride] = useState<number | null>(null)
  const [extraVehicles, setExtraVehicles] = useState(1)
  const [directions, setDirections] = useState<1 | 2>(2)
  const data = useDecisionLab(day)
  const schedules = useMemo<Record<string, ServiceEstimate | null>>(() => Object.fromEntries(ROUTES.map((number) => [number, serviceEstimate(number, day)])), [day])
  const departures = useMemo(() => Object.fromEntries(ROUTES.map((number) => [number, schedules[number]?.departuresByHour])), [schedules])
  const ranking = useMemo(() => rankRoutes(data.responses, windowHours, departures), [data.responses, windowHours, departures])
  const route = ranking.find((item) => item.route === selectedRoute) ?? ranking[0]
  const start = route && selectedWindow?.route === route.route ? selectedWindow.start : route?.windowStart ?? 0
  const end = route && selectedWindow?.route === route.route ? selectedWindow.end : route?.windowEnd ?? windowHours
  const schedule = route ? schedules[route.route] : null
  const scheduledDepartures = schedule?.departuresByHour.slice(start, end).reduce((sum, count) => sum + count, 0) ?? 0
  const scheduleUsable = scheduleWindowUsable(schedule?.departuresByHour, start, end)
  const estimatedHeadway = scheduleUsable ? Math.round(directions * (end - start) * 600 / scheduledDepartures) / 10 : Number.NaN
  const headwayMinutes = headwayOverride ?? estimatedHeadway
  const cycleMinutes = cycleOverride ?? schedule?.cycleMinutes ?? Number.NaN
  const scenario = route ? simulateService(route.points, start, end, { headwayMinutes, cycleMinutes, extraVehicles, directions }) : null
  const source = data.responses.find((item) => item.route === route?.route)
  const incomplete = ranking.length < 10
  const pressures = ranking.flatMap((item) => item.boardingsPerDeparture === null ? [] : [item.boardingsPerDeparture]).sort((a, b) => a - b)
  const medianPressure = pressures.length ? (pressures[Math.floor((pressures.length - 1) / 2)] + pressures[Math.floor(pressures.length / 2)]) / 2 : 0
  const maxPressure = ranking[0]?.boardingsPerDeparture ?? 0
  const top = ranking[0]

  const selectRoute = (number: string) => { setSelectedRoute(number); setSelectedWindow(null); setHeadwayOverride(null); setCycleOverride(null) }
  const changeWindow = (nextStart: number, nextEnd: number) => {
    if (!route) return
    setSelectedRoute(route.route)
    setSelectedWindow({ route: route.route, start: nextStart, end: nextEnd })
    setWindowHours(nextEnd - nextStart)
  }
  const exportForecast = (response: PlanningResponse | undefined) => {
    if (response) downloadPlanningCsv(planningCsv(response, route?.points[start].timestamp ?? ""), response.route)
  }

  return <div className="workspace decision-lab" id="decision-lab">
    <section className="lab-intro" aria-label="Планирование выпуска">
      <div><p className="lab-overline">АНАЛИТИКА ВЫПУСКА / НОЯБРЬ–ДЕКАБРЬ 2025</p><h2>Где проверить усиление выпуска?</h2><p>Сравните пиковые окна маршрутов и проверьте, как дополнительный вагон меняет расчётный интервал и число посадок на отправление.</p></div>
      <div className="lab-date"><label htmlFor="lab-date">Дата прогноза · МСК</label><input id="lab-date" type="date" min="2025-11-01" max="2025-12-31" value={day} aria-invalid={!data.valid} onChange={(event) => { setDay(event.target.value); setSelectedRoute(null); setSelectedWindow(null); setHeadwayOverride(null); setCycleOverride(null) }} /><small>Прогноз строится из данных до 31.10.2025</small></div>
    </section>

    {!data.valid && <div className="lab-message" role="alert">Выберите дату с 1 ноября по 31 декабря 2025 года.</div>}
    {data.valid && data.pending && ranking.length === 0 && <div className="lab-loading" role="status"><Skeleton className="h-24" /><Skeleton className="h-[450px]" />Загружаем маршрутные прогнозы…</div>}
    {data.valid && data.failed > 0 && <div className="lab-message" role="alert">Не удалось загрузить {data.failed} из 10 маршрутов. Рейтинг неполный. <Button variant="secondary" onClick={() => void data.retry()}>Повторить</Button></div>}
    {data.valid && !data.pending && ranking.length === 0 && <div className="lab-message" role="status">Для выбранной даты нет пригодных маршрутно-часовых прогнозов. Выберите другую дату или повторите загрузку.</div>}

    {data.valid && route && <>
      <div className="lab-summary" aria-label="Сводка сети">
        <div className="lab-summary-item"><span>МАРШРУТЫ В РЕЙТИНГЕ</span><strong>{ranking.length}<small> / 10</small></strong><p>{data.refreshing ? "Обновляем прогнозы…" : incomplete ? "Часть данных недоступна" : "Опубликованная маршрутная модель"}</p></div>
        <div className="lab-summary-item"><span>МАКСИМАЛЬНОЕ ДАВЛЕНИЕ СПРОСА · {windowHours} Ч</span><strong>{top.boardingsPerDeparture === null ? "—" : decimal.format(top.boardingsPerDeparture)}</strong><p>посадок на плановое отправление · маршрут {top.route} · {span(top, top.windowStart, top.windowEnd)}</p></div>
        <div className="lab-summary-item"><span>ДЕЙСТВИЕ ДИСПЕТЧЕРА</span><strong className="lab-summary-action">Проверить выпуск</strong><p>приоритет по плановым отправлениям, не по наполнению салона</p></div>
      </div>

      <div className="lab-grid">
        <section className="lab-rank-panel" aria-label="Рейтинг маршрутов для проверки">
          <div className="lab-section-heading"><div><p className="lab-overline">01 / СЕТЬ</p><h3>Куда смотреть сначала</h3></div><span className="lab-count">{dateLabel(day)}</span></div>
          <p className="lab-rank-explain">Рейтинг по максимуму посадок на плановое отправление за окно длиной {windowHours} ч. Красный — от 1,5 медианы сети, янтарный — от 1,15. Это относительный сигнал, не измеренная переполненность.</p>
          <ol className="lab-rank-list">{ranking.map((item, index) => <li key={item.route}>
            <button type="button" className={`lab-rank-button is-${pressureTier(item.boardingsPerDeparture, medianPressure)} ${item.route === route.route ? "is-selected" : ""}`} aria-pressed={item.route === route.route} onClick={() => selectRoute(item.route)}>
              <span className="lab-rank-position">{String(index + 1).padStart(2, "0")}</span>
              <span className="lab-rank-body"><span className="lab-rank-line"><b>Маршрут {item.route}</b><strong>{item.boardingsPerDeparture === null ? "—" : decimal.format(item.boardingsPerDeparture)} <small>на рейс</small></strong></span><span className="lab-rank-meta">{span(item, item.windowStart, item.windowEnd)} · {integer.format(item.windowBoardings)} посадок · {item.scheduledDepartures ?? "?"} отправлений</span><span className="lab-rank-track"><i style={{ width: `${maxPressure > 0 && item.boardingsPerDeparture !== null ? Math.max(2, item.boardingsPerDeparture / maxPressure * 100) : 0}%` }} /></span></span>
              <ArrowRight aria-hidden="true" size={17} />
            </button>
          </li>)}</ol>
          <p className="lab-rank-foot">Для сравнения нужны отправления не менее чем в 70% часов окна и суммарно от двух отправлений в час. Оценка по GTFS, захваченному 30.06.2026; фактический выпуск на дату не подтверждён. Для маршрута 5 расписания и успешных валидаций нет.</p>
        </section>

        <div className="lab-main-column">
          <section className="lab-panel lab-demand-panel" aria-label="Профиль выбранного маршрута">
            <div className="lab-section-heading"><div><p className="lab-overline">02 / ПРОГНОЗ</p><h3>Маршрут {route.route}: спрос по часам</h3></div><span className="lab-model-tag">{source?.model_version ?? "версия недоступна"}</span></div>
            <div className="lab-demand-stats"><div><span>За день</span><strong>{integer.format(route.dayBoardings)}</strong><small>посадок</small></div><div><span>Пиковый час</span><strong>{integer.format(route.peakBoardings)}</strong><small>{clock(route.points[route.peakHour].timestamp)} · посадок</small></div><div className="is-accent"><span>Выбранное окно</span><strong>{integer.format(route.points.slice(start, end).reduce((sum, point) => sum + point.route_baseline, 0))}</strong><small>{span(route, start, end)} · {end - start} ч</small></div></div>
            <DemandChart route={route} start={start} end={end} />
            <WindowTimeline route={route} start={start} end={end} onChange={changeWindow} />
            <details className="lab-details"><summary>Все 24 часа · точные прогнозы</summary><div className="lab-table-wrap" tabIndex={0} role="region" aria-label="Часовые прогнозы маршрута"><table><thead><tr><th>Начало · МСК</th><th>Посадок</th></tr></thead><tbody>{route.points.map((point) => <tr key={point.timestamp}><th>{clock(point.timestamp)}</th><td>{integer.format(point.route_baseline)}</td></tr>)}</tbody></table></div></details>
          </section>

          {route.dayBoardings === 0 ? <section className="lab-unknown" role="status"><h3>Для маршрута {route.route} нет основания рассчитывать выпуск</h3><p>Ноль в прогнозе связан с отсутствием успешных валидаций в истории. Он не доказывает отсутствие пассажиров. Для решения нужны отдельный замер спроса и фактический выпуск.</p></section> : <section className="lab-panel lab-scenario-panel" aria-label="Сценарий дополнительного вагона">
            <div className="lab-section-heading"><div><p className="lab-overline">03 / СЦЕНАРИЙ</p><h3>Добавить вагон на линию</h3></div><span className="lab-assumption-tag">Расчёт по допущениям</span></div>
            <p className="lab-panel-lead">Исходный интервал и оборот рассчитаны из плановых рейсов GTFS для выбранной даты и окна. Их можно исправить вручную; фактическое движение вагонов не наблюдается.</p>
            <div className="lab-controls"><label>Интервал в направлении, мин<input type="number" min="2" max="240" step="0.1" value={Number.isFinite(headwayMinutes) ? headwayMinutes : ""} aria-invalid={!scenario} onChange={(event) => setHeadwayOverride(event.target.value === "" ? NaN : Number(event.target.value))} /></label><label>Полный оборот вагона, мин<input type="number" min="20" max="240" step="1" value={Number.isFinite(cycleMinutes) ? cycleMinutes : ""} aria-invalid={!scenario} onChange={(event) => setCycleOverride(event.target.value === "" ? NaN : Number(event.target.value))} /></label><label>Схема движения<select value={directions} onChange={(event) => setDirections(Number(event.target.value) as 1 | 2)}><option value={2}>Два направления</option><option value={1}>Кольцо · один поток</option></select></label><div className="lab-extra-control"><span>Дополнительные вагоны</span><div><button type="button" aria-label="Убрать один вагон" disabled={extraVehicles === 0} onClick={() => setExtraVehicles((value) => value - 1)}><ChevronDown size={18} /></button><output aria-live="polite">+{extraVehicles}</output><button type="button" aria-label="Добавить один вагон" disabled={extraVehicles === 5} onClick={() => setExtraVehicles((value) => value + 1)}><ChevronUp size={18} /></button></div></div></div>
            <div className="lab-schedule-note"><strong>Время полного оборота маршрута по расписанию оценено: {schedule ? `${schedule.cycleMinutes} мин` : "нет оценки"}.</strong> {schedule ? `По ${schedule.pairedCycles} парам последовательных рейсов одного предполагаемого выхода; диапазон 10–90%: ${schedule.cycleP10}–${schedule.cycleP90} мин. В окне ${scheduledDepartures} плановых отправлений. ${scheduleUsable ? cycleOverride !== null || headwayOverride !== null ? "Есть ручная корректировка." : "Поля заполнены по расписанию." : "Покрытие окна отправлениями слишком низкое для автоматического интервала."}` : "Для этого маршрута нет проверяемой связки рейсов."} <a href={schedule?.sourceUrl ?? "https://gtfs.org/documentation/schedule/reference/"} target="_blank" rel="noreferrer">Источник расписания</a></div>
            {!scenario && <p role="status" className="lab-input-error">В выбранном окне недостаточно плановых отправлений или параметры вне диапазона: интервал 2–240 мин, оборот 20–240 мин, вагоны 0–5. Введите интервал вручную или измените окно.</p>}
            {scenario && <>
              <div className="lab-result-strip"><div><span>Расчётный интервал</span><strong>{decimal.format(headwayMinutes)} <small>→</small> {decimal.format(scenario.scenarioHeadwayMinutes)} <em>мин</em></strong></div><div><span>Посадок на отправление</span><strong>{decimal.format(scenario.baselineBoardingsPerDeparture)} <small>→</small> {decimal.format(scenario.scenarioBoardingsPerDeparture)}</strong></div><div className="lab-relief"><ArrowDownRight size={20} aria-hidden="true" /><strong>{extraVehicles === 0 ? "0%" : `−${decimal.format(scenario.reductionPercent)}%`}</strong><span>расчётных посадок на отправление</span></div></div>
              <div className="lab-chart-title"><h4>Распределение спроса между отправлениями</h4><div><span><i className="lab-legend-before" />до</span><span><i className="lab-legend-after" />после</span></div></div>
              <ScenarioChart hours={scenario.hours} />
              <p className="lab-scroll-hint">Таблицу можно прокрутить вправо →</p>
              <div className="lab-table-wrap" tabIndex={0} role="region" aria-label="Значения сценария по часам"><table><thead><tr><th>Час · МСК</th><th>Посадок всего</th><th>На отправление · до</th><th>После</th></tr></thead><tbody>{scenario.hours.map((hour) => <tr key={hour.timestamp}><th>{clock(hour.timestamp)}</th><td>{integer.format(hour.boardings)}</td><td>{decimal.format(hour.before)}</td><td>{decimal.format(hour.after)}</td></tr>)}</tbody></table></div>
              <p className="lab-equation">Расчёт для {directions === 2 ? "двух направлений" : "кольца"}: {directions} × 60 / интервал = {decimal.format(scenario.baselineDeparturesPerHour)} отправлений/ч на всём маршруте; +{extraVehicles} вагон(ов) при полном обороте {decimal.format(cycleMinutes)} мин ≈ {decimal.format(scenario.scenarioDeparturesPerHour)} отправлений/ч. За окно это ≈ {decimal.format(scenario.extraDeparturesInWindow)} дополнительных отправлений. Спрос {integer.format(scenario.windowBoardings)} посадок не меняется.</p>
            </>}
          </section>}

          <section className="lab-decision" aria-label="Вывод для диспетчера"><div className="lab-decision-icon"><TramFront size={24} /></div><div><p className="lab-overline">04 / РЕШЕНИЕ</p><h3>{route.dayBoardings === 0 ? `Проверить источник данных маршрута ${route.route}` : `Проверить усиление выпуска на маршруте ${route.route}`}</h3><p>{route.dayBoardings === 0 ? "Нет успешных валидаций в обучающей истории. Не меняйте выпуск на основании нулевого прогноза; сначала подтвердите спрос независимым источником." : `В окне ${span(route, start, end)} ожидается ${integer.format(scenario?.windowBoardings ?? route.windowBoardings)} посадок. Сверьте фактическое расписание, свободный подвижной состав, оборотное время и данные о наполнении вагонов перед изменением выпуска.`}</p><div className="lab-decision-actions"><Button variant="secondary" onClick={() => exportForecast(source)} disabled={!source}><Download size={16} />Скачать прогноз CSV</Button><span>Версия: {source?.model_version ?? "—"} · сформирован {source?.generated_at ? new Date(source.generated_at).toLocaleString("ru-RU", { timeZone: "Europe/Moscow" }) : "—"} МСК</span></div></div></section>

          <section className="lab-limits" aria-label="Границы решения"><CircleHelp size={20} aria-hidden="true" /><div><h3>Как читать этот сценарий</h3><p>Прогноз — число успешных валидаций по всему маршруту и часу. Цвет и рейтинг отражают относительное число посадок на плановое отправление, не фактическую наполненность салона: высадки, проверенная вместимость и фактический выпуск неизвестны. Интервал и полный оборот оценены по GTFS с календарями 2025 года, но снимок получен 30.06.2026, а поле block_id пусто: последовательность рейсов одного вагона предположена по коду выхода. Направления считаются симметричными только в сценарии. Влияние добавленного вагона на общий спрос и затраты не оценено.</p></div></section>
        </div>
      </div>
    </>}
  </div>
}

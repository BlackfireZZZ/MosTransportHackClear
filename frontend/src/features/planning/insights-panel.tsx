import type { PlanningPoint } from "@/api/planning"
import { HOT_RATE } from "./flow-color"
import { planningInsights } from "./insights"

interface InsightsPanelProps {
  points: readonly PlanningPoint[]
  selected: string | undefined
  format: (timestamp: string) => string
  number: (value: number) => string
  directionName: (direction: string) => string
  onSelect: (timestamp: string) => void
}

export function InsightsPanel({ points, selected, format, number, directionName, onSelect }: InsightsPanelProps) {
  const { total, peak, busiest, hotSpots } = planningInsights(points)
  return <section className="planning-insights" aria-label="Ключевые показатели прогноза">
    <div className="kpi-grid kpi-grid-triple">
      <article className="kpi"><span>Пиковый интервал</span><strong>{peak ? <>{number(peak.count)}<span className="kpi-unit">посадок</span></> : "—"}</strong><small>{peak ? `${format(peak.timestamp)} · ${number(peak.rate)} посадок/ч` : "нет посадок в периоде"}</small></article>
      <article className="kpi"><span>Самая загруженная остановка</span><strong className="kpi-text" title={busiest?.name}>{busiest?.name ?? "—"}</strong><small>{busiest ? `${number(busiest.count)} посадок за период · ${directionName(busiest.direction)} · оценка` : "нет оценок остановок"}</small></article>
      <article className="kpi"><span>Посадок за период</span><strong>{number(total)}<span className="kpi-unit">посадок</span></strong><small>сценарий с учётом весов</small></article>
    </div>
    <details className="planning-detail planning-hotspots"><summary>Горячие точки · {hotSpots.length}</summary>
    <p>Остановки от {HOT_RATE} посадок/ч — кандидаты для проверки выпуска. Это спрос, не заполненность вагона.</p>
    {hotSpots.length === 0
      ? <p role="status">В выбранном периоде нет остановок с интенсивностью от {HOT_RATE} посадок/ч.</p>
      : <div className="forecast-stop-table" tabIndex={0} role="region" aria-label="Горячие точки"><table>
        <thead><tr><th>Интервал · МСК</th><th>Остановка</th><th>Направление · оценка</th><th>Посадок/ч</th><th>За интервал, посадок</th></tr></thead>
        <tbody>{hotSpots.map((row) => <tr key={`${row.timestamp}-${row.stopId}-${row.direction}`}>
          <th scope="row"><button className="planning-bucket" aria-pressed={row.timestamp === selected} onClick={() => onSelect(row.timestamp)}>{format(row.timestamp)}</button></th>
          <td>{row.name}</td><td>{directionName(row.direction)}</td><td>{number(row.rate)}</td><td>{number(row.count)}</td>
        </tr>)}</tbody>
      </table></div>}
    </details>
  </section>
}

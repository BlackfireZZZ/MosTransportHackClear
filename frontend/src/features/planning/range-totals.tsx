import { useState } from "react"
import type { PlanningPoint } from "@/api/planning"
import { rangeTotals } from "./insights"

interface RangeTotalsPanelProps {
  points: readonly PlanningPoint[]
  filtered: boolean
  format: (timestamp: string) => string
  number: (value: number) => string
}

export function RangeTotalsPanel({ points, filtered, format, number }: RangeTotalsPanelProps) {
  const [from, setFrom] = useState<string | null>(null)
  const [to, setTo] = useState<string | null>(null)
  const totals = rangeTotals(points, from, to)
  if (!totals) return null
  const options = points.map((point) => <option key={point.timestamp} value={point.timestamp}>{format(point.timestamp)}</option>)
  return <section className="planning-range" aria-label="Сумма за выбранный период">
    <h3>Сумма за период</h3>
    <div className="planning-filters planning-range-controls">
      <label>С · МСК<select value={totals.from} onChange={(event) => setFrom(event.target.value)}>{options}</select></label>
      <label>По, включительно · МСК<select value={totals.to} onChange={(event) => setTo(event.target.value)}>{options}</select></label>
    </div>
    <div className="planning-range-totals" aria-live="polite">
      <p>Прогноз модели: <strong>{number(totals.baseline)} посадок</strong></p>
      <p>С учётом весов: <strong>{number(totals.scenario)} посадок</strong></p>
      {!filtered && <p>Не распределено по остановкам: <strong>{number(totals.unallocated)} посадок</strong></p>}
      <p>Интервалов: <strong>{totals.count}</strong></p>
    </div>
  </section>
}

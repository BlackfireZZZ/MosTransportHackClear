import { Button } from "@/components/ui/button"
import { forecastUnit, formatMoscowInstant } from "@/features/forecast/lib/provenance"
import { formatForecastValue, intervalBounds, intervalQuality } from "@/features/forecast/lib/values"
import type { ForecastResponse } from "@/features/forecast/types"

interface ForecastValuesProps {
  snapshot: ForecastResponse
  selectedTimestamp?: string | null
  onSelectTimestamp?: (timestamp: string) => void
}

export function ForecastValues({ snapshot, selectedTimestamp, onSelectTimestamp }: ForecastValuesProps) {
  return <details className="forecast-values">
    <summary>Все значения прогноза — {snapshot.points.length} интервалов</summary>
    <p>{forecastUnit(snapshot)} · {snapshot.run?.unit ?? "единица не указана"} · значение интервала. Время Europe/Moscow.</p>
    <p>{intervalQuality(snapshot)}</p>
    <div className="forecast-stop-table" role="region" aria-label="Прокручиваемые значения прогноза" tabIndex={0}>
      <table aria-label="Все значения прогноза">
        <caption>Начало интервала задано источником; конец отдельной корзины в этом API не указан. Выбор строки меняет срез карты.</caption>
        <thead><tr><th scope="col">Начало · МСК</th><th scope="col">Прогноз</th><th scope="col">Нижняя граница</th><th scope="col">Верхняя граница</th><th scope="col">Сопоставимая вместимость</th></tr></thead>
        <tbody>{snapshot.points.map((point) => {
          const bounds = intervalBounds(point)
          return <tr key={point.timestamp}>
            <th scope="row"><Button variant="ghost" aria-pressed={point.timestamp === selectedTimestamp} onClick={() => onSelectTimestamp?.(point.timestamp)}>{formatMoscowInstant(point.timestamp)}{point.timestamp === selectedTimestamp ? " · выбран" : ""}</Button></th>
            <td>{formatForecastValue(point.predicted_passengers)}</td>
            <td>{formatForecastValue(bounds?.[0] ?? null)}</td>
            <td>{formatForecastValue(bounds?.[1] ?? null)}</td>
            <td>Недоступно</td>
          </tr>
        })}</tbody>
      </table>
    </div>
  </details>
}

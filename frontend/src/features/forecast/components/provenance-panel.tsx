import { useEffect, useState } from "react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { dataKind, formatMoscowInstant, snapshotFreshness, sourceCutoff } from "@/features/forecast/lib/provenance"
import type { ForecastResponse } from "@/features/forecast/types"

interface ProvenancePanelProps {
  forecast: ForecastResponse
  updatedAt: number
  failed: boolean
  fetching: boolean
  pollInterval: number | false
}

export function ProvenancePanel({ forecast, updatedAt, failed, fetching, pollInterval }: ProvenancePanelProps) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 10_000)
    return () => window.clearInterval(timer)
  }, [])
  const freshness = snapshotFreshness(updatedAt, now, failed, pollInterval)
  const run = forecast.run
  const entries = [
    ["Запуск прогноза", run?.run_id],
    ["Версия модели", forecast.model_version],
    ["Целевая величина", run?.target],
    ["Единица измерения", run?.unit],
    ["Набор данных", run?.dataset_id],
    ["Версия источника", run?.source_version],
    ["Версия признаков", run?.feature_version],
    ["Версия сущностей", run?.entity_version],
    ["Версия календаря", run?.calendar_version],
    ["Версия графа прогноза", run?.graph_version],
    ["Версия соответствий карте", forecast.map?.mapping_version],
    ["Время формирования · МСК", formatMoscowInstant(forecast.generated_at)],
    ["Начало прогнозирования · МСК", formatMoscowInstant(run?.forecast_origin)],
    ["Граница исходных данных · МСК", formatMoscowInstant(sourceCutoff(forecast))],
    ["Последняя успешная проверка · МСК", formatMoscowInstant(updatedAt || null)],
  ]
  return (
    <Card aria-label="Происхождение и актуальность прогноза">
      <CardHeader>
        <CardTitle>Данные и обновление</CardTitle>
        <p className="text-sm">{dataKind(forecast)}</p>
        <p role="status" data-testid="forecast-freshness" className="text-sm">
          {freshness.stale ? "Устаревший снимок: последняя успешная загрузка сохранена."
            : "Последняя проверка прогноза успешна."}
          {failed && " Обновление не удалось."}
          {fetching && " Проверяем опубликованный прогноз…"}
        </p>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
          {entries.map(([label, value]) => <div key={label} className="min-w-0">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="break-words font-mono tabular-nums">{value || "Недоступно"}</dd>
          </div>)}
        </dl>
        <p className="mt-4 text-sm text-muted-foreground">
          Поток оперативных наблюдений не подключён. Покрытие источника и время последнего наблюдения недоступны.
          {run?.synthetic && " Данные демонстрационные; даты не подтверждают реальные наблюдения."}
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          {pollInterval === false ? "Автопроверка отключена." : `Автопроверка опубликованного прогноза каждые ${pollInterval / 1000} с, пока вкладка активна.`}
          {` Проверка старше ${freshness.limit / 60_000} мин помечается устаревшей. Это правило интерфейса, не гарантия актуальности источника.`}
        </p>
      </CardContent>
    </Card>
  )
}

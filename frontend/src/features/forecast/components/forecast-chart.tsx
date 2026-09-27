import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ForecastValues } from "./forecast-values"
import { forecastUnit, formatMoscowInstant } from "@/features/forecast/lib/provenance"
import { formatForecastValue, intervalBounds, intervalQuality } from "@/features/forecast/lib/values"
import type { ForecastHorizon, ForecastResponse } from "@/features/forecast/types"

const horizonTitles: Record<ForecastHorizon, string> = {
  day: "Почасовой прогноз",
  month: "Прогноз по дням",
  year: "Прогноз по месяцам",
}

function formatTick(timestamp: string, horizon: ForecastHorizon) {
  const date = new Date(timestamp)
  if (horizon === "day") {
    return new Intl.DateTimeFormat("ru-RU", { hour: "2-digit", timeZone: "Europe/Moscow" }).format(date)
  }
  if (horizon === "month") {
    return new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "short", timeZone: "Europe/Moscow" }).format(date)
  }
  return new Intl.DateTimeFormat("ru-RU", { month: "short", timeZone: "Europe/Moscow" }).format(date)
}

interface ForecastChartProps {
  snapshot: ForecastResponse
  selectedTimestamp?: string | null
  onSelectTimestamp?: (timestamp: string) => void
}

export function ForecastChart({ snapshot, selectedTimestamp, onSelectTimestamp }: ForecastChartProps) {
  const { points, horizon } = snapshot
  const data = points.map((point) => ({
    ...point,
    label: formatTick(point.timestamp, horizon),
    lower_bound: intervalBounds(point)?.[0] ?? null,
    upper_bound: intervalBounds(point)?.[1] ?? null,
    uncertainty: intervalBounds(point) ? point.upper_bound! - point.lower_bound! : null,
  }))
  const peak = Math.max(...points.map((point) => point.predicted_passengers))

  return (
    <Card className="chart-card">
      <CardHeader className="chart-header">
        <div>
          <CardTitle>{horizonTitles[horizon]}</CardTitle>
          <CardDescription>Прогноз для выбранного среза</CardDescription>
        </div>
        <div className="chart-legend" aria-label="Легенда графика">
          <span><i className="legend-line legend-forecast" />Прогноз</span>
          <span><i className="legend-band" />Интервал неопределённости</span>
        </div>
      </CardHeader>
      <CardContent>
        {points.some((point) => !intervalBounds(point)) && <p>Интервал неопределённости недоступен для части значений.</p>}
        <p>{intervalQuality(snapshot)}</p>
        <div className="chart-wrap" role="img" aria-label={`Пик прогноза ${formatForecastValue(peak)} · ${forecastUnit(snapshot)}`}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} onClick={(state) => {
              if (state?.activeTooltipIndex == null || state.activeTooltipIndex === "") return
              const index = Number(state.activeTooltipIndex)
              if (Number.isInteger(index) && points[index]) onSelectTimestamp?.(points[index].timestamp)
            }} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="2 4" />
              <XAxis dataKey="timestamp" tickFormatter={(value: string) => formatTick(value, horizon)} tickLine={false} axisLine={false} minTickGap={28} />
              {selectedTimestamp && <ReferenceLine x={selectedTimestamp} stroke="var(--primary)" strokeDasharray="3 3" />}
              <YAxis tickLine={false} axisLine={false} width={50} />
              <Tooltip
                labelFormatter={(value) => typeof value === "string" ? formatMoscowInstant(value) : ""}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  fontSize: 13,
                }}
                formatter={(value, name) => [formatForecastValue(Number(value)), name === "predicted_passengers" ? "Прогноз" : String(name)]}
              />
              <Area type="monotone" dataKey="lower_bound" stackId="uncertainty" stroke="none" fill="transparent" tooltipType="none" />
              <Area type="monotone" dataKey="uncertainty" stackId="uncertainty" stroke="none" fill="var(--chart-2)" fillOpacity={0.12} tooltipType="none" />
              <Line type="monotone" dataKey="lower_bound" name="Нижняя граница" stroke="transparent" dot={false} activeDot={false} />
              <Line type="monotone" dataKey="upper_bound" name="Верхняя граница" stroke="transparent" dot={false} activeDot={false} />
              <Line type="monotone" dataKey="predicted_passengers" name="Прогноз" stroke="var(--foreground)" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: "var(--primary)" }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <ForecastValues snapshot={snapshot} selectedTimestamp={selectedTimestamp} onSelectTimestamp={onSelectTimestamp} />
      </CardContent>
    </Card>
  )
}

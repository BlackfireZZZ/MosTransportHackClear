import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { PlanningPoint } from "@/api/planning"

export function PlanningChart({ points }: { points: PlanningPoint[] }) {
  const label = (value: string) => new Intl.DateTimeFormat("ru-RU", { timeZone: "Europe/Moscow", month: "short", day: "numeric", hour: "2-digit" }).format(new Date(value))
  return <div role="img" aria-label="Динамика прогноза модели и прогноза с весами, посадок; точные значения в таблице ниже" style={{ width: "100%", height: 280 }}>
    <ResponsiveContainer><LineChart data={points} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
      <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="timestamp" tickFormatter={label} minTickGap={48} /><YAxis width={64} label={{ value: "посадок", angle: -90, position: "insideLeft", offset: 4 }} />
      <Tooltip labelFormatter={(value) => typeof value === "string" ? label(value) : ""} formatter={(value) => `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 }).format(Number(value))} посадок`} /><Legend />
      <Line dataKey="baseline" name="Прогноз модели" stroke="var(--foreground)" dot={points.length === 1} isAnimationActive={false} />
      <Line dataKey="scenario" name="С учётом весов" stroke="var(--chart-2)" strokeDasharray="6 3" dot={points.length === 1} isAnimationActive={false} />
    </LineChart></ResponsiveContainer>
  </div>
}

import type { PlanningResponse } from "@/api/planning"

const quote = (value: string | number | boolean | null | undefined) => {
  const text = String(value ?? "")
  const safe = /^[\s\p{Cc}]*[=+\-@＝＋－＠]|^[\t\r\n]/u.test(text) ? `text:${text}` : text
  return `"${safe.replaceAll('"', '""')}"`
}
export function planningCsv(snapshot: PlanningResponse, timestamp: string): string {
  const columns = ["run_id", "model_version", "route", "horizon", "timezone", "unit", "qualitative", "filter_stop_id", "filter_direction", "timestamp", "bucket_end", "basis", "selected_bucket", "row_kind", "stop_id", "stop_name", "direction", "latitude", "longitude", "baseline", "scenario", "unallocated_baseline", "unallocated_scenario", "forecast_mode", "source_enabled", "scenario_weights", "sources", "provenance", "warnings", "generated_at", "experimental_evaluation", "route_baseline", "route_scenario", "route_unallocated_baseline", "route_unallocated_scenario"]
  const points = snapshot.horizon === "year" ? snapshot.points.filter((point) => point.timestamp === timestamp) : snapshot.points
  const rows = points.flatMap((point) => {
    const prefix = [snapshot.run_id, snapshot.model_version, snapshot.route, snapshot.horizon, snapshot.timezone, snapshot.unit, snapshot.qualitative, snapshot.stop_id, snapshot.direction, point.timestamp, point.bucket_end, point.basis, point.timestamp === timestamp]
    const suffix = [snapshot.forecast_mode, JSON.stringify(snapshot.source_enabled), JSON.stringify(snapshot.factors), JSON.stringify(snapshot.sources), JSON.stringify(snapshot.provenance), JSON.stringify(snapshot.warnings), snapshot.generated_at, snapshot.experimental_evaluation, point.route_baseline, point.route_scenario, point.route_unallocated_baseline, point.route_unallocated_scenario]
    return [[...prefix, "total", "", "", "", "", "", point.baseline, point.scenario, point.unallocated_baseline, point.unallocated_scenario, ...suffix], ...point.spatial.map((stop) => [...prefix, "estimated_stop", stop.stop_id, stop.name, stop.direction, stop.latitude, stop.longitude, stop.baseline, stop.scenario, "", "", ...suffix])]
  })
  return [columns, ...rows].map((row) => row.map(quote).join(",")).join("\r\n") + "\r\n"
}
export function networkPlanningCsv(snapshots: readonly PlanningResponse[], timestamp: string): string {
  const parts = snapshots.map((snapshot) => planningCsv(snapshot, timestamp).trimEnd().split("\r\n"))
  if (parts.length === 0) return ""
  return [parts[0][0], ...parts.flatMap((part) => part.slice(1))].join("\r\n") + "\r\n"
}
export function downloadPlanningCsv(csv: string, route: string) {
  const url = URL.createObjectURL(new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" }))
  const link = document.createElement("a")
  link.href = url
  link.download = `planning-${route}.csv`
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

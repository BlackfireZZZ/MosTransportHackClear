import type { ForecastResponse } from "@/features/forecast/types"
import type { ForecastFilters } from "./selection"
import { formatMoscowInstant, snapshotFreshness, sourceCutoff } from "./provenance"
import { intervalBounds, intervalQuality } from "./values"

export interface ExportContext {
  updatedAt: number
  exportedAt: number
  failed: boolean
  pollInterval: number | false
  selectedTimestamp: string | null
  requestedFilters: ForecastFilters
}

type Cell = string | number | boolean | null | undefined
const formulaLike = /^[\s\p{Cc}]*[=+\-@＝＋－＠]|^[\t\r\n]/u
const quote = (value: Cell) => `"${String(value ?? "").replaceAll('"', '""')}"`

export function forecastCsv(snapshot: ForecastResponse, context: ExportContext): string | null {
  if (snapshot.points.length === 0) return null
  const run = snapshot.run
  const map = run && snapshot.map?.run_id === run.run_id && snapshot.map.entity_version === run.entity_version && snapshot.map.graph_version === run.graph_version ? snapshot.map : null
  const selection = snapshot.selection
  const freshness = snapshotFreshness(context.updatedAt, context.exportedAt, context.failed, context.pollInterval)
  const rows = snapshot.points.map((point) => {
    const bounds = intervalBounds(point)
    const record: Record<string, Cell> = {
      report_schema: "tramflow.dispatcher.csv.v1", run_id: run?.run_id, horizon: snapshot.horizon,
      route_id: snapshot.route.id, route_number: snapshot.route.number, route_name: snapshot.route.name,
      stop_id: selection ? selection.stop_id : context.requestedFilters.stop_id,
      direction_id: selection?.direction_id,
      window_start: selection?.start ?? context.requestedFilters.start,
      window_end_exclusive: selection?.end ?? context.requestedFilters.end,
      selection_evidence: selection ? "server_selection" : "requested_filters_unconfirmed",
      aggregation_key: selection?.aggregation_key,
      timestamp: point.timestamp, timestamp_moscow: formatMoscowInstant(point.timestamp), timezone: "Europe/Moscow",
      bucket_end: null, target: run?.target, unit: run?.unit,
      predicted_value: point.predicted_passengers, lower_bound: bounds?.[0], upper_bound: bounds?.[1],
      capacity: null, capacity_status: "unavailable_no_compatible_metadata",
      interval_method: run?.interval_method, interval_nominal_level: run?.interval_level,
      interval_quality: intervalQuality(snapshot),
      model_version: snapshot.model_version, dataset_id: run?.dataset_id, source_version: run?.source_version,
      feature_version: run?.feature_version, entity_version: run?.entity_version, calendar_version: run?.calendar_version,
      graph_version: run?.graph_version, mapping_version: map?.mapping_version,
      mapping_status: map?.status ?? "unavailable", mapping_reason: map?.reason ?? "missing_or_incompatible_envelope",
      mapping_synthetic: map?.synthetic, mapping_matched_count: map?.matched_count,
      mapping_unmatched_count: map?.unmatched_count, mapping_ambiguous_count: map?.ambiguous_count,
      generated_at: snapshot.generated_at, forecast_origin: run?.forecast_origin, data_cutoff: sourceCutoff(snapshot),
      source_coverage: "unavailable", last_observation: null,
      ui_fetched_at: context.updatedAt > 0 ? new Date(context.updatedAt).toISOString() : null,
      exported_at: new Date(context.exportedAt).toISOString(),
      synthetic: run?.synthetic ?? "unknown", stale: freshness.stale, ui_check_expiry_ms: freshness.limit,
      selected_timestamp: context.selectedTimestamp, is_selected_bucket: point.timestamp === context.selectedTimestamp,
    }
    const escaped: string[] = []
    for (const [key, value] of Object.entries(record)) {
      if (typeof value === "string" && formulaLike.test(value)) {
        record[key] = `text:${value}`
        escaped.push(key)
      }
    }
    record.escaped_text_columns = escaped.join(";")
    return record
  })
  const columns = Object.keys(rows[0])
  return [columns.map(quote).join(","), ...rows.map((row) => columns.map((key) => quote(row[key])).join(","))].join("\r\n") + "\r\n"
}

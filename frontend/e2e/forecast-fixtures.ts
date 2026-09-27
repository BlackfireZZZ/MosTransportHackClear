import type { ForecastHorizon, ForecastResponse, RouteSummary } from "../src/features/forecast/types"

export const routes: readonly RouteSummary[] = [
  { id: 1, number: "Т1", name: "Белорусский вокзал — Каланчёвская", color: "#d9342b" },
  { id: 2, number: "Т2", name: "Черёмушки — Университет", color: "#246b88" },
]

export function forecastResponse(routeId: number, horizon: ForecastHorizon): ForecastResponse {
  const route = routes.find((candidate) => candidate.id === routeId)
  if (!route) throw new Error(`Unknown fixture route: ${routeId}`)
  const targetPeak = route.id === 2 ? 820 : horizon === "month" ? 710 : horizon === "year" ? 760 : 600
  const count = horizon === "day" ? 24 : horizon === "month" ? 31 : 12
  const points = Array.from({ length: count }, (_, index) => {
    const date = horizon === "day"
      ? new Date(Date.UTC(2026, 9, 1, index - 3))
      : horizon === "month"
        ? new Date(Date.UTC(2026, 9, index + 1, -3))
        : new Date(Date.UTC(2026, index, 1, -3))
    const predicted = targetPeak - Math.abs(index - Math.floor(count / 2)) * 10
    return {
      timestamp: date.toISOString(), capacity: 1000, predicted_passengers: predicted,
      lower_bound: predicted - 80, upper_bound: predicted + 90,
    }
  })
  const peak = Math.max(...points.map((point) => point.predicted_passengers))
  return {
    generated_at: "2025-12-31T21:00:00Z", horizon, model_version: "demo-e2e-fixture-v1",
    peak_load_percent: Math.max(...points.map((point) => point.predicted_passengers / point.capacity * 100)),
    peak_passengers: peak, points, route,
    stops: [{ id: route.id * 10 + 1, latitude: 55.776, longitude: 37.583,
      name: "Тестовая остановка", sequence: 1, load_percent: peak / 10, predicted_passengers: peak }],
  }
}

export function mappedForecastResponse(routeId: number, horizon: ForecastHorizon): ForecastResponse {
  const base = forecastResponse(routeId, horizon)
  const firstStop = { ...base.stops[0], name: "Альфа прогноза" }
  const secondStop = { ...firstStop, id: firstStop.id + 1, name: "Бета прогноза", sequence: 2, longitude: 37.603, latitude: 55.786 }
  const run = {
    run_id: `fixture-run-${horizon}`, dataset_id: "fixture", source_version: "fixture.v1",
    feature_version: "fixture.v1", entity_version: "fixture-entities", calendar_version: "calendar.v1",
    graph_version: null, target: "synthetic_boardings", unit: "event_count", synthetic: true,
    forecast_origin: base.generated_at, data_cutoff: base.generated_at,
    interval_level: null, interval_method: "synthetic_fixture", identity_namespace: "serving-surrogate-integer",
  }
  return { ...base, run, selection: { start: base.points[0].timestamp,
    end: new Date(Date.parse(base.points.at(-1)!.timestamp) + 3600000).toISOString(), stop_id: null, direction_id: null,
    aggregation_key: "route_bucket", interval_aggregation: "single_source_or_unavailable" }, stops: [firstStop, secondStop],
    stop_points: base.points.flatMap((point) => [firstStop, secondStop].map((stop, index) => {
      const value = Math.round(point.predicted_passengers * (index === 0 ? 0.4 : 0.6))
      return { stop_id: stop.id, timestamp: point.timestamp, bucket_end: null, direction_id: "out",
        predicted_passengers: value, lower_bound: value - 10, upper_bound: value + 10, capacity: null,
        aggregation_scope: "stop_bucket_direction" }
    })),
    map: { run_id: run.run_id, entity_version: run.entity_version, graph_version: null, mapping_version: null,
      synthetic: true, status: "unavailable", reason: "mapping_not_configured",
      matched_count: 0, unmatched_count: 2, ambiguous_count: 0,
      positions: [firstStop, secondStop].map((stop) => ({ stop_id: stop.id, direction_id: "out",
        status: "unmatched", reason: "mapping_not_configured", position_kind: "synthetic_demo",
        osm_stop_id: null, longitude: stop.longitude, latitude: stop.latitude })),
    },
  }
}

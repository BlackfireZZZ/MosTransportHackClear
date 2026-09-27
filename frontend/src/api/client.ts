import type { ForecastFilters } from "@/features/forecast/lib/selection"
import type {
  PublishedHorizon,
  ForecastResponse,
  RouteSummary,
  RouteStop,
} from "@/features/forecast/types"
import type {
  NetworkStats,
  OverpassStatus,
  RouteDetail,
  StopDetail,
  TramEdge,
  TramGraphGeoJson,
  TramPath,
  TramRoute,
  TramStop,
} from "@/features/tram-network/types"

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  })
  if (!response.ok) {
    throw new ApiError(`API request failed: ${response.status}`, response.status)
  }
  return (await response.json()) as T
}

type QueryValue = string | number | null | undefined

function search(params: Record<string, QueryValue>): string {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue
    query.set(key, String(value))
  }
  const rendered = query.toString()
  return rendered ? `?${rendered}` : ""
}

const TRAM_GRAPH = "/api/v1/tram-graph"

export const api = {
  routes: () => request<RouteSummary[]>("/api/v1/routes"),
  routeStops: (routeId: number) => request<RouteStop[]>(`/api/v1/routes/${routeId}/stops`),
  forecast: (routeId: number, horizon: PublishedHorizon, filters: ForecastFilters = {}, signal?: AbortSignal) =>
    request<ForecastResponse>(`/api/v1/forecasts${search({ route_id: routeId, horizon, ...filters })}`, { signal }),
  overpassStatus: () => request<OverpassStatus>("/api/v1/overpass/status"),
  tramGraph: {
    stats: () => request<NetworkStats>(`${TRAM_GRAPH}/stats`),
    routes: () => request<readonly TramRoute[]>(`${TRAM_GRAPH}/routes`),
    route: (ref: string) =>
      request<RouteDetail>(`${TRAM_GRAPH}/routes/${encodeURIComponent(ref)}`),
    stops: (query?: string, limit?: number) =>
      request<readonly TramStop[]>(`${TRAM_GRAPH}/stops${search({ q: query, limit })}`),
    stop: (stopId: number) => request<StopDetail>(`${TRAM_GRAPH}/stops/${stopId}`),
    edges: (route?: string | null) =>
      request<readonly TramEdge[]>(`${TRAM_GRAPH}/edges${search({ route })}`),
    geojson: (route?: string | null) =>
      request<TramGraphGeoJson>(`${TRAM_GRAPH}/geojson${search({ route })}`),
    path: (from: number, to: number) =>
      request<TramPath>(`${TRAM_GRAPH}/path${search({ from, to })}`),
  },
}

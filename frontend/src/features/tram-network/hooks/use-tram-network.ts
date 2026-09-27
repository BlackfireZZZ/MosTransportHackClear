import { useQuery } from "@tanstack/react-query"

import { api } from "@/api/client"

const GRAPH = "tram-graph"

/** The extract is a committed snapshot; refetching it during a session buys nothing. */
const SNAPSHOT = { staleTime: Infinity, gcTime: Infinity } as const

export function useNetworkStats() {
  return useQuery({ queryKey: [GRAPH, "stats"], queryFn: api.tramGraph.stats, ...SNAPSHOT })
}

export function useTramRoutes() {
  return useQuery({ queryKey: [GRAPH, "routes"], queryFn: api.tramGraph.routes, ...SNAPSHOT })
}

/**
 * ~660 KB. One query key for the whole network means every consumer shares a single
 * request for the session, however many components ask for it.
 */
export function useNetworkGeoJson() {
  return useQuery({
    queryKey: [GRAPH, "geojson", null],
    queryFn: () => api.tramGraph.geojson(),
    ...SNAPSHOT,
  })
}

export function useRouteGeoJson(ref: string | null) {
  return useQuery({
    queryKey: [GRAPH, "geojson", ref],
    queryFn: () => api.tramGraph.geojson(ref),
    enabled: ref !== null,
    ...SNAPSHOT,
  })
}

export function useStopSearch(query: string) {
  const trimmed = query.trim()
  return useQuery({
    queryKey: [GRAPH, "stops", trimmed],
    queryFn: () => api.tramGraph.stops(trimmed, 20),
    enabled: trimmed.length >= 2,
    staleTime: 5 * 60_000,
  })
}

export function useStopDetail(stopId: number | null) {
  return useQuery({
    queryKey: [GRAPH, "stop", stopId],
    queryFn: () => api.tramGraph.stop(stopId!),
    enabled: stopId !== null,
    ...SNAPSHOT,
  })
}

export function useTramPath(from: number | null, to: number | null) {
  return useQuery({
    queryKey: [GRAPH, "path", from, to],
    queryFn: () => api.tramGraph.path(from!, to!),
    enabled: from !== null && to !== null && from !== to,
    ...SNAPSHOT,
  })
}

export function useOverpassStatus() {
  return useQuery({
    queryKey: ["overpass", "status"],
    queryFn: api.overpassStatus,
    staleTime: 5 * 60_000,
  })
}

import { skipToken, useQuery } from "@tanstack/react-query"

import { api } from "@/api/client"
import type { ForecastFilters } from "@/features/forecast/lib/selection"
import type { PublishedHorizon } from "@/features/forecast/types"

export function useRoutes() {
  return useQuery({ queryKey: ["routes"], queryFn: api.routes, staleTime: 5 * 60_000 })
}

export function useRouteStops(routeId: number | null) {
  return useQuery({ queryKey: ["route-stops", routeId], queryFn: routeId === null ? skipToken : () => api.routeStops(routeId), staleTime: 5 * 60_000 })
}

export function useForecast(routeId: number | null, horizon: PublishedHorizon, filters: ForecastFilters = {}, valid = true, pollInterval: number | false = 60_000) {
  return useQuery({
    queryKey: ["forecast", routeId, horizon, filters],
    queryFn: valid && routeId !== null && Number.isInteger(routeId) && routeId > 0
      ? ({ signal }) => api.forecast(routeId, horizon, filters, signal)
      : skipToken,
    refetchInterval: pollInterval,
    refetchOnReconnect: pollInterval !== false,
  })
}

import { useQueries } from "@tanstack/react-query"
import { planningForecast, type PlanningRequest } from "@/api/planning"

export const NETWORK_ROUTES = ["1", "5", "7", "11", "12", "17", "25", "26", "28", "50"] as const

export function useNetworkPlanning(request: PlanningRequest, enabled: boolean) {
  const queries = useQueries({ queries: NETWORK_ROUTES.map((route) => {
    const body = { ...request, route, horizon: "day" as const, stop_id: null, direction: null }
    return {
      queryKey: ["planning-network", body],
      queryFn: ({ signal }: { signal: AbortSignal }) => planningForecast(body, signal),
      enabled,
      retry: false,
      staleTime: 5 * 60_000,
    }
  }) })
  return {
    responses: queries.flatMap((query) => query.data ? [query.data] : []),
    loading: enabled && queries.some((query) => query.isPending && query.fetchStatus === "fetching"),
    failed: queries.filter((query) => query.isError).length,
    retry: () => Promise.all(queries.filter((query) => query.isError).map((query) => query.refetch())),
  }
}

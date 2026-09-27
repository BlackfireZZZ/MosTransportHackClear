import { useQueries } from "@tanstack/react-query"
import { planningForecast, type PlanningRequest } from "@/api/planning"
import { ROUTES } from "./model"

const disabled = { weather: false, calendar: false, events: false, traffic: false }
const factors = { weather: { enabled: false, multiplier: 1 }, calendar: { enabled: false, multiplier: 1 }, events: { enabled: false, multiplier: 1 }, traffic: { enabled: false, multiplier: 1 } }

export function useDecisionLab(day: string) {
  const valid = /^2025-(11|12)-\d{2}$/.test(day) && day >= "2025-11-01" && day <= "2025-12-31" && !Number.isNaN(Date.parse(`${day}T00:00:00+03:00`))
  const queries = useQueries({ queries: ROUTES.map((route) => {
    const request: PlanningRequest = { route, start_date: day, horizon: "day", forecast_mode: "approved",
      stop_id: null, direction: null, source_enabled: disabled, factors }
    return { queryKey: ["decision-lab", day, route], queryFn: ({ signal }: { signal: AbortSignal }) => planningForecast(request, signal), enabled: valid,
      retry: false, staleTime: 5 * 60_000 }
  }) })
  return {
    valid,
    responses: queries.flatMap((query) => query.data ? [query.data] : []),
    pending: queries.some((query) => query.isPending && query.fetchStatus === "fetching"),
    refreshing: queries.some((query) => query.isFetching && Boolean(query.data)),
    failed: queries.filter((query) => query.isError).length,
    retry: () => Promise.all(queries.filter((query) => query.isError).map((query) => query.refetch())),
  }
}

import { useEffect, useState } from "react"
import { skipToken, useQuery } from "@tanstack/react-query"
import { planningForecast, type PlanningRequest } from "@/api/planning"

export function usePlanning(request: PlanningRequest, valid: boolean) {
  const key = JSON.stringify(request)
  const [debouncedKey, setDebouncedKey] = useState(key)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedKey(key), 300)
    return () => window.clearTimeout(timer)
  }, [key])
  const settled = key === debouncedKey
  const query = useQuery({
    queryKey: ["planning", key],
    queryFn: valid && settled ? ({ signal }) => planningForecast(request, signal) : skipToken,
    retry: false,
  })
  return { ...query, settling: !settled }
}

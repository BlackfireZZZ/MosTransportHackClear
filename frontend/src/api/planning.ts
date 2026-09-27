import { ApiError } from "@/api/client"

export const factorIds = ["weather", "calendar", "events", "traffic"] as const
export type FactorId = typeof factorIds[number]
export type Factors = Record<FactorId, { enabled: boolean; multiplier: number }>
export interface PlanningRequest {
  forecast_mode: "approved" | "external_experiment" | "stop_model"
  source_enabled: Record<FactorId, boolean>
  route: string
  start_date: string
  horizon: "day" | "month" | "year"
  stop_id: string | null
  direction: string | null
  factors: Factors
}
export interface PlanningStop {
  stop_id: string
  name: string
  direction: string
  latitude: number | null
  longitude: number | null
}
export interface PlanningPoint {
  timestamp: string
  bucket_end: string
  basis: "competition_period" | "scenario_projection"
  baseline: number
  scenario: number
  route_baseline: number
  route_scenario: number
  route_unallocated_baseline: number
  route_unallocated_scenario: number
  unallocated_baseline: number
  unallocated_scenario: number
  spatial: Array<PlanningStop & { baseline: number; scenario: number }>
}
export interface PlanningResponse {
  forecast_mode: "approved" | "external_experiment" | "stop_model"
  source_enabled: Record<FactorId, boolean>
  route: string
  start_date: string
  horizon: PlanningRequest["horizon"]
  stop_id: string | null
  direction: string | null
  generated_at: string
  experimental_evaluation: string | null
  run_id: string
  model_version: string
  unit: string
  timezone: string
  qualitative: boolean
  warnings: string[]
  sources: Array<{ id: string; label: string; status: string; detail: string; url: string | null }>
  factors: Factors
  points: PlanningPoint[]
  stops: PlanningStop[]
  routes: string[]
  provenance: Record<string, string>
}
export async function planningForecast(body: PlanningRequest, signal?: AbortSignal): Promise<PlanningResponse> {
  const response = await fetch("/api/v1/planning/forecast", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  })
  if (!response.ok) throw new ApiError(`Planning request failed: ${response.status}`, response.status)
  return response.json() as Promise<PlanningResponse>
}

import type { PlanningPoint, PlanningResponse } from "@/api/planning"

export const ROUTES = ["1", "5", "7", "11", "12", "17", "25", "26", "28", "50"] as const
export const DEFAULT_WINDOW_HOURS = 3

export interface RoutePriority {
  route: string
  points: PlanningPoint[]
  dayBoardings: number
  windowBoardings: number
  windowStart: number
  windowEnd: number
  scheduledDepartures: number | null
  boardingsPerDeparture: number | null
  peakHour: number
  peakBoardings: number
  windowShare: number
}

const count = (point: PlanningPoint) => point.route_baseline

export type PressureTier = "high" | "elevated" | "regular" | "unknown"

export function pressureTier(value: number | null, networkMedian: number): PressureTier {
  if (value === null || !Number.isFinite(value) || networkMedian <= 0) return "unknown"
  if (value >= 1.5 * networkMedian) return "high"
  if (value >= 1.15 * networkMedian) return "elevated"
  return "regular"
}

export function scheduleWindowUsable(departures: readonly number[] | undefined, start: number, end: number): boolean {
  if (departures?.length !== 24 || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > 24 || end <= start) return false
  const window = departures.slice(start, end)
  return window.every((value) => Number.isInteger(value) && value >= 0) &&
    window.filter((value) => value > 0).length >= Math.ceil(window.length * 0.7) &&
    window.reduce((sum, value) => sum + value, 0) >= 2 * window.length
}

export function rankRoutes(responses: readonly PlanningResponse[], windowHours = DEFAULT_WINDOW_HOURS,
  departuresByRoute: Readonly<Record<string, readonly number[] | undefined>> = {}): RoutePriority[] {
  if (!Number.isInteger(windowHours) || windowHours < 1 || windowHours > 24) return []
  return responses.flatMap((response) => {
    if (response.forecast_mode !== "approved" || response.horizon !== "day" || response.points.length !== 24 ||
      response.points.some((point) => !Number.isFinite(count(point)) || count(point) < 0 ||
        Date.parse(point.bucket_end) - Date.parse(point.timestamp) !== 3_600_000)) return []
    const points = [...response.points].sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    const midnight = Date.parse(`${response.start_date}T00:00:00+03:00`)
    if (!Number.isFinite(midnight) || points.some((point, index) => Date.parse(point.timestamp) !== midnight + index * 3_600_000)) return []
    const dayBoardings = points.reduce((sum, point) => sum + count(point), 0)
    const departures = departuresByRoute[response.route]
    const validSchedule = departures?.length === 24 && departures.every((value) => Number.isInteger(value) && value >= 0)
    let windowStart = 0
    let windowBoardings = -1
    let scheduledDepartures: number | null = null
    let boardingsPerDeparture: number | null = null
    for (let start = 0; start <= points.length - windowHours; start += 1) {
      const total = points.slice(start, start + windowHours).reduce((sum, point) => sum + count(point), 0)
      const scheduled = validSchedule ? departures.slice(start, start + windowHours).reduce((sum, value) => sum + value, 0) : null
      const pressure = validSchedule && scheduleWindowUsable(departures, start, start + windowHours) && scheduled !== null ? total / scheduled : null
      if (pressure !== null ? boardingsPerDeparture === null || pressure > boardingsPerDeparture ||
        (pressure === boardingsPerDeparture && total > windowBoardings) : boardingsPerDeparture === null && total > windowBoardings) {
        windowStart = start; windowBoardings = total; scheduledDepartures = scheduled; boardingsPerDeparture = pressure
      }
    }
    const peakHour = points.reduce((best, point, index) => count(point) > count(points[best]) ? index : best, 0)
    return [{ route: response.route, points, dayBoardings, windowBoardings, windowStart, windowEnd: windowStart + windowHours,
      scheduledDepartures, boardingsPerDeparture,
      peakHour, peakBoardings: count(points[peakHour]), windowShare: dayBoardings > 0 ? windowBoardings / dayBoardings : 0 }]
  }).sort((a, b) => (b.boardingsPerDeparture ?? -1) - (a.boardingsPerDeparture ?? -1) ||
    b.windowBoardings - a.windowBoardings || Number(a.route) - Number(b.route))
}

export interface ServiceAssumptions {
  headwayMinutes: number
  cycleMinutes: number
  extraVehicles: number
  directions: 1 | 2
}

export interface HourlyScenario {
  timestamp: string
  boardings: number
  before: number
  after: number
}

export interface ServiceScenario {
  baselineDeparturesPerHour: number
  scenarioDeparturesPerHour: number
  scenarioHeadwayMinutes: number
  extraDeparturesInWindow: number
  baselineBoardingsPerDeparture: number
  scenarioBoardingsPerDeparture: number
  reductionPercent: number
  windowBoardings: number
  hours: HourlyScenario[]
}

export function simulateService(points: readonly PlanningPoint[], start: number, end: number, assumptions: ServiceAssumptions): ServiceScenario | null {
  const { headwayMinutes, cycleMinutes, extraVehicles, directions } = assumptions
  if (!Number.isFinite(headwayMinutes) || headwayMinutes < 2 || headwayMinutes > 240 ||
    !Number.isFinite(cycleMinutes) || cycleMinutes < 20 || cycleMinutes > 240 ||
    !Number.isInteger(extraVehicles) || extraVehicles < 0 || extraVehicles > 5 ||
    (directions !== 1 && directions !== 2) ||
    !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > points.length || end <= start) return null
  const window = points.slice(start, end)
  if (window.some((point) => !Number.isFinite(count(point)) || count(point) < 0)) return null
  const baselineDeparturesPerHour = directions * 60 / headwayMinutes
  const scenarioDeparturesPerHour = baselineDeparturesPerHour + directions * extraVehicles * 60 / cycleMinutes
  const windowBoardings = window.reduce((sum, point) => sum + count(point), 0)
  const baselineBoardingsPerDeparture = windowBoardings / (window.length * baselineDeparturesPerHour)
  const scenarioBoardingsPerDeparture = windowBoardings / (window.length * scenarioDeparturesPerHour)
  return {
    baselineDeparturesPerHour, scenarioDeparturesPerHour,
    scenarioHeadwayMinutes: directions * 60 / scenarioDeparturesPerHour,
    extraDeparturesInWindow: window.length * (scenarioDeparturesPerHour - baselineDeparturesPerHour),
    baselineBoardingsPerDeparture, scenarioBoardingsPerDeparture,
    reductionPercent: 100 * (1 - baselineDeparturesPerHour / scenarioDeparturesPerHour),
    windowBoardings,
    hours: window.map((point) => ({ timestamp: point.timestamp, boardings: count(point),
      before: count(point) / baselineDeparturesPerHour, after: count(point) / scenarioDeparturesPerHour })),
  }
}

import type { PlanningStop } from "@/api/planning"

/** The catalog lists each direction's stops in GTFS stop_sequence order; the organizer data has no trip_headsign, so the last stop names the direction. */
export function directionNames(stops: readonly PlanningStop[]): (direction: string) => string {
  const terminals = new Map<string, string>()
  for (const stop of stops) terminals.set(stop.direction, stop.name)
  return (direction) => {
    const terminal = terminals.get(direction)
    return terminal ? `→ ${terminal}` : `Направление ${direction}`
  }
}

function bearingBetween(from: PlanningStop, to: PlanningStop): number {
  const latitude = (((from.latitude ?? 0) + (to.latitude ?? 0)) / 2) * (Math.PI / 180)
  const east = ((to.longitude ?? 0) - (from.longitude ?? 0)) * Math.cos(latitude)
  const north = (to.latitude ?? 0) - (from.latitude ?? 0)
  return ((Math.atan2(east, north) * 180) / Math.PI + 360) % 360
}

/** Clockwise degrees from north towards the next distinct stop of the same direction, in catalog order. */
export function directionBearings(stops: readonly PlanningStop[]): (stopId: string, direction: string) => number | undefined {
  const located = (stop: PlanningStop) => stop.latitude !== null && stop.longitude !== null
  const byDirection = new Map<string, PlanningStop[]>()
  for (const stop of stops) if (located(stop)) byDirection.set(stop.direction, [...(byDirection.get(stop.direction) ?? []), stop])
  const bearings = new Map<string, number>()
  for (const [direction, ordered] of byDirection) {
    ordered.forEach((stop, index) => {
      const distinct = (other: PlanningStop) => other.latitude !== stop.latitude || other.longitude !== stop.longitude
      const next = ordered.slice(index + 1).find(distinct)
      const previous = ordered.slice(0, index).reverse().find(distinct)
      if (next) bearings.set(`${stop.stop_id}|${direction}`, bearingBetween(stop, next))
      else if (previous) bearings.set(`${stop.stop_id}|${direction}`, bearingBetween(previous, stop))
    })
  }
  return (stopId, direction) => bearings.get(`${stopId}|${direction}`)
}

import type { FeatureCollection, Point } from "geojson"

import type { LngLat, StopRef, TramGraphGeoJson, TramPath, TramRoute } from "@/features/tram-network/types"

/** OSM leaves 10 stops nameless; the extract falls back to this spelling. */
const UNNAMED_PREFIX = "node/"

const UNREACHABLE_FALLBACK =
  "Между этими остановками нет рельсовой связи."

export const OSM_ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0"

export function isUnnamedStop(name: string): boolean {
  return name.startsWith(UNNAMED_PREFIX)
}

export function stopLabel(stop: { readonly id: number; readonly name: string }): string {
  return isUnnamedStop(stop.name) ? `Без названия · OSM ${stop.id}` : stop.name
}

export interface RouteGroup {
  readonly component: number
  readonly routes: readonly TramRoute[]
}

/**
 * Groups routes by network component while preserving the order the API returned.
 * Refs are not all numeric ("А", "1а", "т1"); re-sorting them here would break that order.
 */
export function groupRoutesByComponent(routes: readonly TramRoute[]): readonly RouteGroup[] {
  const groups = new Map<number, TramRoute[]>()
  for (const route of routes) {
    const bucket = groups.get(route.component)
    if (bucket) bucket.push(route)
    else groups.set(route.component, [route])
  }
  return [...groups.entries()]
    .sort(([left], [right]) => left - right)
    .map(([component, grouped]) => ({ component, routes: grouped }))
}

export function componentLabel(component: number): string {
  return component === 0 ? "Основная сеть" : "Северная сеть"
}

export type PathAbsence = NonNullable<TramPath["reason_code"]>

export type PathSummary =
  | { readonly status: "found"; readonly stopCount: number; readonly lengthM: number; readonly routes: readonly string[] }
  | {
      readonly status: "unreachable"
      readonly code: PathAbsence | null
      readonly explanation: string
      readonly reason: string
    }

const ABSENCE_EXPLANATION: Record<PathAbsence, string> = {
  different_components:
    "Сеть физически разделена на две части: северная сеть вокруг Тимирязевской не связана рельсами с остальной Москвой.",
  wrong_direction:
    "Рельсы между этими остановками есть, но только во встречном направлении — попробуйте поменять точки местами.",
  unknown_stop: "Одной из выбранных остановок нет в графе.",
}

/**
 * `found: false` arrives as a plain 200: the absence is an answer, not a failure.
 *
 * Branch on `reason_code`, never on `reason` — the latter is English prose meant for
 * someone reading the API directly, and only one of the three causes is the split
 * network. Explaining a wrong-direction result as two disconnected components is wrong.
 */
export function summarisePath(path: TramPath): PathSummary {
  if (!path.found) {
    const code = path.reason_code ?? null
    return {
      status: "unreachable",
      code,
      explanation: code ? ABSENCE_EXPLANATION[code] : UNREACHABLE_FALLBACK,
      reason: path.reason ?? UNREACHABLE_FALLBACK,
    }
  }
  return {
    status: "found",
    stopCount: path.stops.length,
    lengthM: path.total_length_m,
    routes: path.routes,
  }
}

export function formatKm(metres: number): string {
  return `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1, minimumFractionDigits: 1 }).format(metres / 1000)} км`
}

export function formatMetres(metres: number): string {
  return `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 }).format(metres)} м`
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 }).format(value)
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—"
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Europe/Moscow",
  }).format(parsed)
}

/** [west, south, east, north] — longitude first, as GeoJSON stores it. */
export type Bounds = readonly [number, number, number, number]

/** Grow a bounding box by a fraction of its own span, clamped to valid lat/lon. */
export function padBounds(bounds: Bounds, fraction: number): Bounds {
  const [west, south, east, north] = bounds
  // Off the longer side, so the narrow axis gets the same room rather than a
  // proportionally tighter one.
  const margin = Math.max(east - west, north - south) * fraction
  return [
    Math.max(-180, west - margin),
    Math.max(-90, south - margin),
    Math.min(180, east + margin),
    Math.min(90, north + margin),
  ]
}

export function boundsOfCoordinates(coordinates: readonly LngLat[]): Bounds | null {
  if (coordinates.length === 0) return null
  let west = Infinity
  let south = Infinity
  let east = -Infinity
  let north = -Infinity
  for (const [longitude, latitude] of coordinates) {
    if (longitude < west) west = longitude
    if (latitude < south) south = latitude
    if (longitude > east) east = longitude
    if (latitude > north) north = latitude
  }
  return Number.isFinite(west) ? [west, south, east, north] : null
}

export function boundsOfGraph(graph: TramGraphGeoJson | undefined): Bounds | null {
  if (!graph) return null
  const coordinates: LngLat[] = []
  for (const feature of graph.features) {
    if (feature.geometry.type === "Point") coordinates.push(feature.geometry.coordinates)
    else coordinates.push(...feature.geometry.coordinates)
  }
  return boundsOfCoordinates(coordinates)
}

/**
 * The generated contract marks every payload readonly; the parsed JSON underneath is an
 * ordinary mutable object, which is what maplibre's source API is typed against.
 */
export function asFeatureCollection(graph: TramGraphGeoJson): FeatureCollection {
  return graph as unknown as FeatureCollection
}

export const EMPTY_COLLECTION: FeatureCollection = { type: "FeatureCollection", features: [] }

export type MarkerRole = "from" | "to" | "selected"

export interface MarkerProperties {
  readonly role: MarkerRole
  readonly label: string
}

/**
 * from / to / selected frequently point at the same stop; the first role wins so the
 * endpoint colours are not overpainted by the generic selection ring.
 */
export function markerCollection(
  from: StopRef | null,
  to: StopRef | null,
  selected: StopRef | null,
): FeatureCollection<Point, MarkerProperties> {
  const seen = new Set<number>()
  const features: FeatureCollection<Point, MarkerProperties>["features"] = []
  const push = (stop: StopRef | null, role: MarkerRole) => {
    if (!stop || seen.has(stop.id)) return
    seen.add(stop.id)
    features.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: [stop.longitude, stop.latitude] },
      properties: { role, label: stopLabel(stop) },
    })
  }
  push(from, "from")
  push(to, "to")
  push(selected, "selected")
  return { type: "FeatureCollection", features }
}

export function pathCollection(path: TramPath | undefined): FeatureCollection {
  if (!path?.found || path.geometry_quality === "inferred" || path.geometry.length < 2) return EMPTY_COLLECTION
  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: { type: "LineString", coordinates: path.geometry as unknown as number[][] },
        properties: {},
      },
    ],
  }
}

export function featureStopId(properties: unknown): number | null {
  if (typeof properties !== "object" || properties === null) return null
  const value = (properties as Record<string, unknown>).id
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

export function featureStopLabel(properties: unknown): string | null {
  if (typeof properties !== "object" || properties === null) return null
  const record = properties as Record<string, unknown>
  const name = record.name
  const id = record.id
  if (typeof name !== "string" || typeof id !== "number") return null
  return stopLabel({ id, name })
}

import type { Page, Route } from "@playwright/test"
import type { NetworkStats, OverpassStatus, StopDetail, TramGraphGeoJson, TramPath, TramRoute, TramStop } from "../src/features/tram-network/types"

export const stops: readonly TramStop[] = [
  { id: 101, name: "Альфа", latitude: 55.75, longitude: 37.6, routes: ["1"] },
  { id: 102, name: "Бета", latitude: 55.76, longitude: 37.62, routes: ["1"] },
  { id: 201, name: "Северная", latitude: 55.85, longitude: 37.55, routes: ["6"] },
]
const routes: readonly TramRoute[] = [
  { ref: "1", component: 0, length_m: 1700, stop_count: 2 },
  { ref: "6", component: 1, length_m: 0, stop_count: 1 },
]
const stats: NetworkStats = {
  stops: 3, edges: 1, routes: 2, route_relations: 2, total_length_km: 1.7,
  components: [{ size: 2, routes: ["1"] }, { size: 1, routes: ["6"] }],
  degree_histogram: { "0": 1, "1": 2 },
  segment_length_m: { min: 1700, max: 1700, mean: 1700, median: 1700 },
}
export function graph(ref: string | null = null): TramGraphGeoJson {
  return {
    type: "FeatureCollection", metadata: { missing_geometry_edges: 0, synthetic: true, source: "Deterministic E2E fixture", filtered_to_route: ref },
    features: [
      ...stops.filter((stop) => !ref || stop.routes.includes(ref)).map((stop) => ({
        type: "Feature" as const, geometry: { type: "Point" as const, coordinates: [stop.longitude, stop.latitude] as [number, number] },
        properties: { id: stop.id, name: stop.name, routes: stop.routes },
      })),
      ...(!ref || ref === "1" ? [{ type: "Feature" as const,
        geometry: { type: "LineString" as const, coordinates: [[37.6, 55.75], [37.62, 55.76]] as [number, number][] },
        properties: { geometry_quality: "synthetic" as const, source: 101, target: 102, length_m: 1700, routes: ["1"] },
      }] : []),
    ],
  }
}
export async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) })
}
export async function mockNetwork(page: Page, failures = new Set<string>()) {
  const requests: string[] = []
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url())
    const key = url.pathname.replace("/api/v1/", "") + decodeURIComponent(url.search)
    requests.push(key)
    if (failures.has(key)) return json(route, { detail: "Fixture outage" }, 503)
    if (key === "routes") return json(route, [])
    if (key === "overpass/status") return json(route, { reachable: false, url: "https://fixture.invalid", detail: "Offline fixture" } satisfies OverpassStatus)
    if (key === "tram-graph/stats") return json(route, stats)
    if (key === "tram-graph/routes") return json(route, routes)
    if (url.pathname.endsWith("/geojson")) return json(route, graph(url.searchParams.get("route")))
    if (url.pathname.endsWith("/stops")) return json(route, stops.filter((stop) => stop.name.toLowerCase().includes((url.searchParams.get("q") ?? "").toLowerCase())))
    if (/\/stops\/\d+$/.test(url.pathname)) {
      const stop = stops.find((candidate) => url.pathname.endsWith(`/${candidate.id}`))!
      const detail: StopDetail = { ...stop, neighbours: stop.id === 201 ? [] : [{ id: stop.id === 101 ? 102 : 101, name: stop.id === 101 ? "Бета" : "Альфа", direction: stop.id === 101 ? "out" : "in", length_m: 1700, routes: ["1"] }] }
      return json(route, detail)
    }
    if (url.pathname.endsWith("/path")) {
      const from = Number(url.searchParams.get("from")), to = Number(url.searchParams.get("to"))
      const found = from === 101 && to === 102
      const path: TramPath = { geometry_quality: found ? "synthetic" : null, missing_geometry_edges: 0, found, stops: found ? stops.slice(0, 2) : [], geometry: found ? [[37.6, 55.75], [37.62, 55.76]] : [], routes: found ? ["1"] : [], total_length_m: found ? 1700 : 0, reason_code: found ? null : from === 201 || to === 201 ? "different_components" : "wrong_direction", reason: found ? null : "Fixture has no directed connection" }
      return json(route, path)
    }
    return json(route, { detail: `Unmocked endpoint: ${key}` }, 500)
  })
  return requests
}

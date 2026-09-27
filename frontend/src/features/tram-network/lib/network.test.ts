import { describe, expect, it } from "vitest"

import {
  boundsOfCoordinates,
  componentLabel,
  formatKm,
  groupRoutesByComponent,
  isUnnamedStop,
  markerCollection,
  padBounds,
  type PathAbsence,
  pathCollection,
  stopLabel,
  summarisePath,
} from "./network"
import type { TramPath, TramRoute } from "@/features/tram-network/types"

function route(ref: string, component = 0): TramRoute {
  return { ref, stop_count: 10, length_m: 1000, component }
}

describe("route refs", () => {
  it("keeps the order the API returned instead of sorting numerically", () => {
    const refs = ["1", "2", "7", "10", "16", "50", "1а", "39а", "47а", "А", "т1", "т2"]
    const groups = groupRoutesByComponent(refs.map((ref) => route(ref)))

    expect(groups).toHaveLength(1)
    expect(groups[0].routes.map((item) => item.ref)).toEqual(refs)
  })

  it("groups by component, largest network first, without reordering refs", () => {
    const groups = groupRoutesByComponent([
      route("1", 0),
      route("6", 1),
      route("10", 1),
      route("т1", 0),
    ])

    expect(groups.map((group) => group.component)).toEqual([0, 1])
    expect(groups[0].routes.map((item) => item.ref)).toEqual(["1", "т1"])
    expect(groups[1].routes.map((item) => item.ref)).toEqual(["6", "10"])
  })

  it("names both components", () => {
    expect(componentLabel(0)).toBe("Основная сеть")
    expect(componentLabel(1)).toBe("Северная сеть")
  })
})

describe("unnamed stops", () => {
  it("recognises the node/<id> fallback", () => {
    expect(isUnnamedStop("node/535844771")).toBe(true)
    expect(isUnnamedStop("Курский вокзал")).toBe(false)
  })

  it("renders a readable label instead of the raw fallback", () => {
    expect(stopLabel({ id: 535844771, name: "node/535844771" })).toBe("Без названия · OSM 535844771")
  })

  it("leaves a real name untouched", () => {
    expect(stopLabel({ id: 472373305, name: "Курский вокзал" })).toBe("Курский вокзал")
  })
})

describe("path summary", () => {
  const unreachable: TramPath = {
    missing_geometry_edges: 0, found: false,
    reason:
      "Даниловская мануфактура and Тимирязевская академия are in different parts of the tram network; there is no track connecting them",
    stops: [],
    total_length_m: 0,
    geometry: [],
    routes: [],
  }

  it("carries the backend explanation for a disconnected pair", () => {
    const summary = summarisePath(unreachable)

    expect(summary.status).toBe("unreachable")
    expect(summary.status === "unreachable" && summary.reason).toContain("different parts")
  })

  it("falls back to its own wording when the backend sends no reason", () => {
    const summary = summarisePath({ ...unreachable, reason: null })

    expect(summary.status === "unreachable" && summary.reason).toBe(
      "Между этими остановками нет рельсовой связи.",
    )
  })

  it("draws nothing for an unreachable pair", () => {
    expect(pathCollection(unreachable).features).toHaveLength(0)
  })

  it("reports stop count and length for a found path", () => {
    const summary = summarisePath({
      missing_geometry_edges: 0, found: true,
      reason: null,
      stops: [
        { id: 1, name: "A", latitude: 55.7, longitude: 37.6, routes: ["16"] },
        { id: 2, name: "B", latitude: 55.75, longitude: 37.66, routes: ["16"] },
      ],
      total_length_m: 32014.2,
      geometry: [
        [37.6, 55.7],
        [37.66, 55.75],
      ],
      routes: ["16", "т2"],
    })

    expect(summary).toEqual({
      status: "found",
      stopCount: 2,
      lengthM: 32014.2,
      routes: ["16", "т2"],
    })
    expect(formatKm(32014.2)).toBe("32,0 км")
  })
})

describe("coordinate order", () => {
  it("reads bounds as [west, south, east, north] from [lon, lat] pairs", () => {
    expect(
      boundsOfCoordinates([
        [37.62, 55.7],
        [37.55, 55.83],
      ]),
    ).toEqual([37.55, 55.7, 37.62, 55.83])
  })

  it("has no bounds for an empty geometry", () => {
    expect(boundsOfCoordinates([])).toBeNull()
  })

  it("emits markers as [longitude, latitude]", () => {
    const collection = markerCollection(
      { id: 1, name: "Курский вокзал", latitude: 55.7555891, longitude: 37.6601427 },
      null,
      { id: 1, name: "Курский вокзал", latitude: 55.7555891, longitude: 37.6601427 },
    )

    expect(collection.features).toHaveLength(1)
    expect(collection.features[0].geometry.coordinates).toEqual([37.6601427, 55.7555891])
    expect(collection.features[0].properties.role).toBe("from")
  })
})

describe("summarisePath explanations", () => {
  const absent = (reason_code: PathAbsence | null, reason: string) =>
    summarisePath({
      missing_geometry_edges: 0, found: false,
      reason_code,
      reason,
      stops: [],
      total_length_m: 0,
      geometry: [],
      routes: [],
    })

  it("explains a one-way pair as direction, not as a split network", () => {
    const summary = absent("wrong_direction", "the track exists but only the other way round")

    expect(summary.status).toBe("unreachable")
    if (summary.status !== "unreachable") return
    expect(summary.explanation).toContain("встречном направлении")
    // The split-network wording would be plainly false here.
    expect(summary.explanation).not.toContain("Тимирязевской")
  })

  it("explains disconnected components as the split network", () => {
    const summary = absent("different_components", "are in different parts")

    if (summary.status !== "unreachable") throw new Error("expected unreachable")
    expect(summary.explanation).toContain("Тимирязевской")
  })

  it("keeps the backend prose verbatim alongside the explanation", () => {
    const summary = absent("unknown_stop", "unknown stop id 999 in 'from'")

    if (summary.status !== "unreachable") throw new Error("expected unreachable")
    expect(summary.reason).toBe("unknown stop id 999 in 'from'")
    expect(summary.explanation).not.toBe(summary.reason)
  })

  it("falls back when the contract sends no code", () => {
    const summary = absent(null, "")

    if (summary.status !== "unreachable") throw new Error("expected unreachable")
    expect(summary.explanation.length).toBeGreaterThan(0)
  })
})

describe("padBounds", () => {
  it("takes the margin off the longer side for both axes", () => {
    expect(padBounds([37.0, 55.0, 38.0, 55.5], 0.1)).toEqual([36.9, 54.9, 38.1, 55.6])
  })

  it("never leaves the valid lat/lon range", () => {
    const [west, south, east, north] = padBounds([-179.0, -89.0, 179.0, 89.0], 0.5)
    expect(west).toBe(-180)
    expect(south).toBe(-90)
    expect(east).toBe(180)
    expect(north).toBe(90)
  })

  it("leaves the committed graph's extent comfortably inside", () => {
    // Measured from the API over all 856 stops.
    const [west, south, east, north] = padBounds([37.391, 55.596, 37.821, 55.888], 0.12)
    expect(west).toBeCloseTo(37.339, 3)
    expect(south).toBeCloseTo(55.544, 3)
    expect(east).toBeCloseTo(37.873, 3)
    expect(north).toBeCloseTo(55.940, 3)
  })

  it("is a no-op at zero", () => {
    expect(padBounds([37.0, 55.0, 38.0, 56.0], 0)).toEqual([37.0, 55.0, 38.0, 56.0])
  })
})

it("does not draw inferred path geometry as rails", () => {
  const path: TramPath = { missing_geometry_edges: 0, found: true, stops: [], total_length_m: 100, routes: ["1"],
    geometry: [[37, 55], [38, 56]], geometry_quality: "inferred" }
  expect(pathCollection(path).features).toEqual([])
  expect(pathCollection({ ...path, geometry_quality: "provided" }).features).toHaveLength(1)
  expect(pathCollection({ ...path, geometry_quality: "synthetic" }).features).toHaveLength(1)
})

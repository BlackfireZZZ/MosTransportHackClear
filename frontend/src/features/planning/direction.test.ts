import { describe, expect, it } from "vitest"
import { directionBearings, directionNames } from "./direction"

const stop = (direction: string, name: string) => ({ stop_id: `${direction}-${name}`, name, direction, latitude: null, longitude: null })

describe("direction names", () => {
  it("names each direction after its last stop in catalog order", () => {
    const name = directionNames([stop("0", "Чертаново Южное"), stop("0", "Каховская"), stop("0", "Москворецкий рынок"), stop("1", "Москворецкий рынок"), stop("1", "Чертаново Южное")])
    expect(name("0")).toBe("→ Москворецкий рынок")
    expect(name("1")).toBe("→ Чертаново Южное")
  })

  it("falls back to the raw code when the direction has no stops", () => {
    expect(directionNames([])("0")).toBe("Направление 0")
  })
})

describe("direction bearings", () => {
  const at = (stop_id: string, direction: string, latitude: number | null, longitude: number | null) => ({ stop_id, name: stop_id, direction, latitude, longitude })

  it("points each stop towards the next stop of its direction, the last one continuing from the previous", () => {
    const bearing = directionBearings([at("a", "0", 55.6, 37.6), at("b", "0", 55.7, 37.6), at("c", "0", 55.7, 37.8), at("c", "1", 55.7, 37.8), at("a", "1", 55.6, 37.6)])
    expect(bearing("a", "0")).toBeCloseTo(0)
    expect(bearing("b", "0")).toBeCloseTo(90)
    expect(bearing("c", "0")).toBeCloseTo(90)
    expect(bearing("c", "1")).toBeGreaterThan(180)
    expect(bearing("c", "1")).toBeLessThan(270)
  })

  it("skips coincident or missing coordinates and gives no bearing to a lone stop", () => {
    const bearing = directionBearings([at("a", "0", 55.6, 37.6), at("a2", "0", 55.6, 37.6), at("x", "0", null, null), at("b", "0", 55.5, 37.6), at("z", "1", 55.6, 37.6)])
    expect(bearing("a", "0")).toBeCloseTo(180)
    expect(bearing("a2", "0")).toBeCloseTo(180)
    expect(bearing("x", "0")).toBeUndefined()
    expect(bearing("z", "1")).toBeUndefined()
  })
})

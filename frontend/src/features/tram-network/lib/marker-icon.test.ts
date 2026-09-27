import { describe, expect, it } from "vitest"
import { DIRECTED_ICON_SIZE, directedMarkerIcon, quantizeColor } from "./marker-icon"

const pixel = (icon: ReturnType<typeof directedMarkerIcon>, x: number, y: number) => Array.from(icon.data.slice((y * icon.width + x) * 4, (y * icon.width + x) * 4 + 4))

describe("directed forecast marker icon", () => {
  it("draws a filled circle with an up-pointing white arrow on a transparent square", () => {
    const icon = directedMarkerIcon("#d9342b", false)
    const size = DIRECTED_ICON_SIZE * 2
    expect([icon.width, icon.height, icon.data.length]).toEqual([size, size, size * size * 4])
    expect(pixel(icon, 0, 0)[3]).toBe(0)
    expect(pixel(icon, size / 2, size / 2 - 6).slice(0, 3)).toEqual([255, 255, 255])
    const fill = pixel(icon, size / 2 - 15, size / 2)
    expect(fill.slice(0, 3)).toEqual([0xd9, 0x34, 0x2b])
    expect(pixel(icon, size / 2, 1).slice(0, 3)).toEqual([0x1f, 0x25, 0x29])
  })

  it("thickens the outline for the selected stop and tolerates unknown colours", () => {
    const plain = directedMarkerIcon("#246b88", false)
    const selected = directedMarkerIcon("#246b88", true)
    const size = DIRECTED_ICON_SIZE * 2
    expect(pixel(plain, size / 2, 6).slice(0, 3)).toEqual([0x24, 0x6b, 0x88])
    expect(pixel(selected, size / 2, 6).slice(0, 3)).toEqual([0x1f, 0x25, 0x29])
    expect(pixel(directedMarkerIcon("tomato", false), size / 2 - 15, size / 2).slice(0, 3)).toEqual([0x6b, 0x75, 0x80])
  })
})

describe("colour quantization", () => {
  it("snaps channels to steps of 8 and leaves non-hex colours alone", () => {
    expect(quantizeColor("#d9342b")).toBe("#d83828")
    expect(quantizeColor("#ffffff")).toBe("#f8f8f8")
    expect(quantizeColor("tomato")).toBe("tomato")
  })
})

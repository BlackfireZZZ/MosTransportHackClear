/** Logical diameter in CSS pixels; the raster is drawn at pixel ratio 2. */
export const DIRECTED_ICON_SIZE = 24

const STROKE = [0x1f, 0x25, 0x29] as const
const FALLBACK = [0x6b, 0x75, 0x80] as const
const SAMPLES = 4

/** Snaps each channel to a step of 8 so animated colours reuse a bounded set of map images. */
export function quantizeColor(color: string): string {
  const match = /^#([0-9a-f]{6})$/i.exec(color)
  if (!match) return color
  const value = Number.parseInt(match[1], 16)
  return `#${[value >> 16, (value >> 8) & 0xff, value & 0xff].map((channel) => Math.min(248, Math.round(channel / 8) * 8).toString(16).padStart(2, "0")).join("")}`
}

function rgb(color: string): readonly [number, number, number] {
  const match = /^#([0-9a-f]{6})$/i.exec(color)
  if (!match) return FALLBACK
  const value = Number.parseInt(match[1], 16)
  return [value >> 16, (value >> 8) & 0xff, value & 0xff]
}

function inArrow(x: number, y: number): boolean {
  const edges: Array<[number, number, number, number]> = [[0, -13, 9, 9], [9, 9, 0, 3], [0, 3, -9, 9], [-9, 9, 0, -13]]
  let inside = false
  for (const [x1, y1, x2, y2] of edges) if ((y1 > y) !== (y2 > y) && x < ((x2 - x1) * (y - y1)) / (y2 - y1) + x1) inside = !inside
  return inside
}

/** Raw RGBA raster for MapLibre `addImage`: a stop circle with a north-pointing arrow, rotated on the map by bearing. */
export function directedMarkerIcon(color: string, selected: boolean): { width: number; height: number; data: Uint8Array } {
  const size = DIRECTED_ICON_SIZE * 2
  const radius = size / 2 - 1
  const stroke = selected ? 8 : 3
  const fill = rgb(color)
  const data = new Uint8Array(size * size * 4)
  for (let py = 0; py < size; py += 1) {
    for (let px = 0; px < size; px += 1) {
      let r = 0, g = 0, b = 0, a = 0
      for (let sy = 0; sy < SAMPLES; sy += 1) {
        for (let sx = 0; sx < SAMPLES; sx += 1) {
          const x = px + (sx + 0.5) / SAMPLES - size / 2
          const y = py + (sy + 0.5) / SAMPLES - size / 2
          const distance = Math.hypot(x, y)
          if (distance > radius) continue
          const [cr, cg, cb] = inArrow(x, y) ? [255, 255, 255] : distance > radius - stroke ? STROKE : fill
          r += cr; g += cg; b += cb; a += 1
        }
      }
      const offset = (py * size + px) * 4
      if (a === 0) continue
      data[offset] = Math.round(r / a)
      data[offset + 1] = Math.round(g / a)
      data[offset + 2] = Math.round(b / a)
      data[offset + 3] = Math.round((255 * a) / (SAMPLES * SAMPLES))
    }
  }
  return { width: size, height: size, data }
}

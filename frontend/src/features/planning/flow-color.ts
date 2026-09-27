export const HOT_RATE = 50
export const ZERO_COLOR = "#8a9299"

/** Sequential ramp for boardings per hour: lightness falls monotonically, so order survives without hue. */
export const RATE_STOPS = [
  { rate: 0, color: "#f6cf5a" },
  { rate: 15, color: "#ee8a2a" },
  { rate: HOT_RATE, color: "#c8321f" },
  { rate: 100, color: "#7d0f24" },
] as const

const channels = (color: string) => [1, 3, 5].map((index) => Number.parseInt(color.slice(index, index + 2), 16))
const hex = (values: number[]) => `#${values.map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`

export function rateColor(rate: number): string {
  if (!(rate > 0)) return ZERO_COLOR
  const upper = RATE_STOPS.findIndex((stop) => stop.rate >= rate)
  if (upper === -1) return RATE_STOPS.at(-1)!.color
  if (upper === 0) return RATE_STOPS[0].color
  const low = RATE_STOPS[upper - 1]
  const high = RATE_STOPS[upper]
  const t = (rate - low.rate) / (high.rate - low.rate)
  const [a, b] = [channels(low.color), channels(high.color)]
  return hex(a.map((value, index) => value + (b[index] - value) * t))
}

export function flowAppearance(count: number, timestamp: string, bucketEnd: string) {
  const hours = (Date.parse(bucketEnd) - Date.parse(timestamp)) / 3_600_000
  if (!Number.isFinite(count) || count < 0 || !Number.isFinite(hours) || hours <= 0) return { color: ZERO_COLOR, rate: null }
  const rate = count / hours
  return { color: rateColor(rate), rate }
}

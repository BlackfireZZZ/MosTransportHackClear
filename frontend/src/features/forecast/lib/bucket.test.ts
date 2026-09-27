import { describe, expect, it } from "vitest"
import { mappedForecastResponse } from "../../../../e2e/forecast-fixtures"
import { bucketScope, bucketStops, selectedBucket } from "./bucket"

describe.each(["day", "month", "year"] as const)("%s snapshot selection", (horizon) => {
  it("joins exact stop values and falls back only to a present chart bucket", () => {
    const data = mappedForecastResponse(1, horizon)
    const timestamp = data.points[1].timestamp
    expect(selectedBucket(data, { scope: bucketScope(data), timestamp })).toBe(timestamp)
    expect(bucketStops(data, timestamp).map((row) => row.predicted_passengers)).toEqual([
      Math.round(data.points[1].predicted_passengers * 0.4), Math.round(data.points[1].predicted_passengers * 0.6),
    ])
    expect(bucketStops({ ...data, stop_points: data.stop_points?.filter((row) => row.timestamp !== timestamp) }, timestamp)).toEqual([])
    expect(selectedBucket(data, { scope: "other-route", timestamp })).toBe(data.points[0].timestamp)
    expect(selectedBucket(data, { scope: bucketScope(data), timestamp: "absent" })).toBe(data.points[0].timestamp)
  })
})

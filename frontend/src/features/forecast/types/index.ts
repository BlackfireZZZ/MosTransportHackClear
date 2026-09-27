import type { components } from "@/api/schema.generated"

type Schemas = components["schemas"]

export type ForecastHorizon = Schemas["ForecastHorizon"]
export type PublishedHorizon = Extract<ForecastHorizon, "day" | "month">
export type ForecastPoint = Schemas["ForecastPointResponse"]
export type ForecastResponse = Schemas["ForecastResponse"]
export type RouteSummary = Schemas["RouteResponse"]
export type RouteStop = Schemas["RouteStopResponse"]

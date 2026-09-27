import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

const renderer = vi.hoisted(() => ({ fail: false, instances: [] as Array<Record<string, unknown>> }))
vi.mock("maplibre-gl", () => {
  class Map {
    listeners: Record<string, () => void> = {}
    canvas = document.createElement("canvas")
    sources: Record<string, { setData: ReturnType<typeof vi.fn> }> = {}
    layers: Record<string, unknown> = {}
    hits: unknown[] = []
    addSource(id: string) { this.sources[id] = { setData: vi.fn() } }
    addLayer(layer: { id: string }) { this.layers[layer.id] = layer }
    getSource(id: string) { return this.sources[id] }
    getLayer(id: string) { return this.layers[id] }
    images: Record<string, { width: number; height: number }> = {}
    hasImage(id: string) { return id in this.images }
    addImage(id: string, image: { width: number; height: number }) { this.images[id] = image }
    setPaintProperty() {}
    queryRenderedFeatures() { return this.hits }
    fitBounds = vi.fn()
    cameraForBounds = vi.fn(() => ({ center: [37.62, 55.75], zoom: 9 }))
    getCenter = vi.fn(() => [37.62, 55.75])
    getZoom = vi.fn(() => 10)
    getBounds = vi.fn(() => ({ getWest: () => 37.35, getSouth: () => 55.52, getEast: () => 37.93, getNorth: () => 55.96 }))
    setMaxBounds = vi.fn()
    setMinZoom = vi.fn()
    jumpTo = vi.fn()
    off() {}
    constructor() {
      if (renderer.fail) throw new Error("WebGL unavailable")
      renderer.instances.push(this as unknown as Record<string, unknown>)
    }
    on(name: string, callback: () => void) { this.listeners[name] = callback }
    addControl() {}
    getCanvas() { return this.canvas }
    remove() {}
  }
  class Control {}
  class Popup {
    text = ""
    remove() {}
    setLngLat() { return this }
    setText(text: string) { this.text = text; return this }
    addTo() { return this }
  }
  return { MapLibreMap: Map, NavigationControl: Control, ScaleControl: Control,
    AttributionControl: Control, Popup, setWorkerUrl: vi.fn() }
})

import { TramMap } from "./tram-map"
import type { TramGraphGeoJson } from "../types"

const network: TramGraphGeoJson = { type: "FeatureCollection", metadata: { missing_geometry_edges: 0, synthetic: false }, features: [
  { type: "Feature", geometry: { type: "Point", coordinates: [37.6, 55.7] },
    properties: { id: 101, name: "Остановка А", routes: ["1"] } },
  { type: "Feature", geometry: { type: "LineString", coordinates: [[37.6, 55.7], [37.7, 55.8]] },
    properties: { geometry_quality: "provided", source: 101, target: 102, length_m: 400, routes: ["1"] } },
] }
const defaults = { network, routeGeoJson: undefined, selectedRoute: null, path: undefined,
  fromStop: null, toStop: null, selectedStop: null }
beforeEach(() => { renderer.fail = false; renderer.instances = [] })
afterEach(cleanup)

it("keeps a selectable committed graph when WebGL initialization fails and retries", async () => {
  renderer.fail = true
  const onStopClick = vi.fn()
  render(<TramMap {...defaults} onStopClick={onStopClick} />)
  await screen.findByText(/Карта недоступна/)
  expect(screen.queryByText(/сеть отрисована/)).not.toBeInTheDocument()
  fireEvent.click(screen.getByText("Остановки и участки сети"))
  fireEvent.click(screen.getByRole("button", { name: "Остановка А · OSM 101" }))
  expect(onStopClick).toHaveBeenCalledWith(101)
  expect(screen.getByText(/OSM 101 → OSM 102/)).toBeInTheDocument()
  renderer.fail = false
  fireEvent.click(screen.getByRole("button", { name: "Повторить загрузку карты" }))
  await waitFor(() => expect(screen.getByTestId("tram-map")).toHaveAttribute("data-state", "loading"))
  expect(renderer.instances).toHaveLength(1)
})

it("reports every renderer error truthfully, including generic style network failures", () => {
  render(<TramMap {...defaults} onStopClick={vi.fn()} />)
  const listeners = renderer.instances[0].listeners as Record<string, () => void>
  act(() => listeners.error())
  expect(screen.getByTestId("tram-map")).toHaveAttribute("data-state", "unavailable")
  expect(screen.getByText(/Карта недоступна/)).toBeInTheDocument()
})

it("exposes missing and synthetic geometry without hover", () => {
  renderer.fail = true
  const partial: TramGraphGeoJson = { ...network,
    metadata: { missing_geometry_edges: 1, synthetic: true },
    features: network.features.map((feature) => "source" in feature.properties
      ? { ...feature, properties: { ...feature.properties, geometry_quality: "inferred" as const } }
      : feature) as TramGraphGeoJson["features"] }
  render(<TramMap {...defaults} network={partial} onStopClick={vi.fn()} />)
  expect(screen.getByText(/Прямые соединения не показаны как рельсы/)).toBeInTheDocument()
  expect(screen.getByText(/Синтетическая геометрия/)).toBeInTheDocument()
  expect(screen.getByText(/OSM 101 → OSM 102.*геометрия отсутствует/)).toBeInTheDocument()
})

it("keeps the map available when graph metadata is absent", () => {
  const incomplete = { type: "FeatureCollection", features: network.features } as TramGraphGeoJson
  render(<TramMap {...defaults} network={incomplete} onStopClick={vi.fn()} />)
  expect(screen.getByTestId("tram-map")).toHaveAttribute("data-state", "loading")
  expect(screen.queryByText(/Синтетическая геометрия/)).not.toBeInTheDocument()
})


it("renders forecast points with separate serving identities and routes clicks correctly", async () => {
  const forecastClick = vi.fn()
  const osmClick = vi.fn()
  render(<TramMap {...defaults} network={undefined} onStopClick={osmClick}
    forecastMarkers={[{ stopId: 11, longitude: 37.6, latitude: 55.7, value: 42, label: "Демо", selected: true, synthetic: true }]}
    onForecastStopClick={forecastClick} />)
  const instance = renderer.instances[0]
  const listeners = instance.listeners as Record<string, (event?: unknown) => void>
  act(() => listeners.load())
  const sources = instance.sources as Record<string, { setData: ReturnType<typeof vi.fn> }>
  await waitFor(() => expect(sources.forecast.setData).toHaveBeenCalled())
  const payload = sources.forecast.setData.mock.calls.at(-1)?.[0] as { features: Array<{ geometry: { type: string }; properties: Record<string, unknown> }> }
  expect(payload.features[0].geometry.type).toBe("Point")
  expect(payload.features[0].properties).toMatchObject({ forecast_stop_id: 11, value: 42, selected: true, synthetic: true })
  expect(payload.features[0].properties.id).toBeUndefined()
  instance.hits = [{ layer: { id: "forecast-points" }, properties: { forecast_stop_id: 11 } }]
  act(() => listeners.click({ point: { x: 0, y: 0 } }))
  expect(forecastClick).toHaveBeenCalledWith(11)
  expect(osmClick).not.toHaveBeenCalled()
})


it("draws markers with a bearing as rotated direction icons that stay clickable", async () => {
  const forecastClick = vi.fn()
  render(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} interactiveNetworkStops={false}
    forecastMarkers={[{ stopId: 3, longitude: 37.6, latitude: 55.7, value: 10, label: "Туда", selected: false, synthetic: false, color: "#d9342b", bearing: 90 },
      { stopId: 4, longitude: 37.6, latitude: 55.7, value: 10, label: "Без направления", selected: false, synthetic: false }]}
    onForecastStopClick={forecastClick} />)
  const instance = renderer.instances.at(-1) as Record<string, unknown> & { listeners: Record<string, (event?: unknown) => void>; hits: unknown[]; sources: Record<string, { setData: ReturnType<typeof vi.fn> }>; layers: Record<string, { filter?: unknown; layout?: Record<string, unknown> }>; images: Record<string, unknown> }
  act(() => instance.listeners.load())
  await waitFor(() => expect(instance.sources.forecast.setData).toHaveBeenCalled())
  const payload = instance.sources.forecast.setData.mock.calls.at(-1)?.[0] as { features: Array<{ properties: Record<string, unknown> }> }
  expect(payload.features[0].properties).toMatchObject({ bearing: 90, icon: "forecast-directed-#d83828-plain" })
  expect(payload.features[1].properties).not.toHaveProperty("bearing")
  expect(Object.keys(instance.images)).toEqual(["forecast-directed-#d83828-plain"])
  expect(instance.layers["forecast-points"].filter).toEqual(["!", ["has", "bearing"]])
  expect(instance.layers["forecast-directed"].layout).toMatchObject({ "icon-rotate": ["get", "bearing"], "icon-rotation-alignment": "map" })
  instance.hits = [{ layer: { id: "forecast-directed" }, properties: { forecast_stop_id: 3 } }]
  act(() => instance.listeners.click({ point: { x: 1, y: 1 } }))
  expect(forecastClick).toHaveBeenCalledWith(3)
})

it("fits forecast coordinates only when opted in and preserves camera for value-only changes", () => {
  const marker = { stopId: 1, longitude: 37.59, latitude: 55.60, value: 42, label: "GTFS", selected: false, synthetic: false }
  const { rerender } = render(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} forecastMarkers={[marker]} />)
  const instance = renderer.instances[0]
  const listeners = instance.listeners as Record<string, () => void>
  const fit = instance.fitBounds as ReturnType<typeof vi.fn>
  act(() => listeners.load())
  expect(fit).not.toHaveBeenCalled()
  rerender(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} fitForecastMarkers forecastMarkers={[marker]} />)
  expect(fit).toHaveBeenCalledExactlyOnceWith([37.59, 55.60, 37.59, 55.60], { padding: 56, maxZoom: 14, duration: 0 })
  rerender(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} fitForecastMarkers forecastMarkers={[{ ...marker, value: 84 }]} />)
  expect(fit).toHaveBeenCalledTimes(1)
  rerender(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} fitForecastMarkers forecastMarkers={[marker, { ...marker, stopId: 2, latitude: 55.64, longitude: 37.61 }]} />)
  expect(fit).toHaveBeenLastCalledWith([37.59, 55.60, 37.61, 55.64], { padding: 56, maxZoom: 14, duration: 0 })
  expect(fit).toHaveBeenCalledTimes(2)
})

it("bounds the forecast camera to Moscow and exposes marker labels on hover", async () => {
  render(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} interactiveNetworkStops={false}
    fitForecastMarkers forecastMarkers={[{ stopId: 1, longitude: 37.59, latitude: 55.60, value: 42, label: "Остановка А · 42 посадки", selected: false, synthetic: false }]} />)
  const instance = renderer.instances[0] as Record<string, unknown> & {
    listeners: Record<string, (event?: unknown) => void>; hits: unknown[]; canvas: HTMLCanvasElement
    cameraForBounds: ReturnType<typeof vi.fn>; setMinZoom: ReturnType<typeof vi.fn>; setMaxBounds: ReturnType<typeof vi.fn>
  }
  act(() => instance.listeners.load())
  await waitFor(() => expect(instance.setMinZoom).toHaveBeenCalledWith(9))
  expect(instance.cameraForBounds).toHaveBeenCalledWith([37.35, 55.52, 37.93, 55.96], { padding: 40 })
  expect(instance.setMaxBounds).toHaveBeenCalled()
  instance.hits = [{ layer: { id: "forecast-points" }, properties: { label: "Остановка А · 42 посадки" }, geometry: { type: "Point", coordinates: [37.59, 55.60] } }]
  act(() => instance.listeners.mousemove({ point: { x: 1, y: 1 } }))
  expect(instance.canvas.style.cursor).toBe("pointer")
  instance.hits = []
  act(() => instance.listeners.mousemove({ point: { x: 1, y: 1 } }))
  expect(instance.canvas.style.cursor).toBe("")
})

it("renders caller-supplied boarding color and updates it without altering exact counts", () => {
  const marker = { stopId: 7, latitude: 55.6, longitude: 37.6, value: 12.345, color: "#b86e13", selected: false, synthetic: false, label: "Остановка" }
  const { rerender } = render(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} forecastMarkers={[marker]} />)
  const instance = renderer.instances[0]
  act(() => (instance.listeners as Record<string, () => void>).load())
  const source = (instance.sources as Record<string, { setData: ReturnType<typeof vi.fn> }>).forecast
  expect((source.setData.mock.calls.at(-1)?.[0] as { features: Array<{ properties: Record<string, unknown> }> }).features[0].properties).toMatchObject({ value: 12.345, color: "#b86e13" })
  rerender(<TramMap {...defaults} network={undefined} onStopClick={vi.fn()} forecastMarkers={[{ ...marker, value: 60, color: "#d9342b" }]} />)
  expect((source.setData.mock.calls.at(-1)?.[0] as { features: Array<{ properties: Record<string, unknown> }> }).features[0].properties).toMatchObject({ value: 60, color: "#d9342b" })
  expect((instance.layers as Record<string, { paint: Record<string, unknown> }>)["forecast-points"].paint["circle-color"]).toEqual(["coalesce", ["get", "color"], ["case", ["get", "synthetic"], "#b86e13", "#246b88"]])
})

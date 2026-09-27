import {
  AttributionControl,
  MapLibreMap,
  NavigationControl,
  Popup,
  ScaleControl,
  setWorkerUrl,
  type GeoJSONSource,
  type LngLatBoundsLike,
  type MapMouseEvent,
  type PointLike,
} from "maplibre-gl"
import "maplibre-gl/dist/maplibre-gl.css"
import { useEffect, useRef, useState } from "react"

import { Button } from "@/components/ui/button"

import {
  asFeatureCollection,
  boundsOfCoordinates,
  boundsOfGraph,
  padBounds,
  EMPTY_COLLECTION,
  featureStopId,
  featureStopLabel,
  markerCollection,
  OSM_ATTRIBUTION,
  pathCollection,
  type Bounds,
} from "@/features/tram-network/lib/network"
import { DIRECTED_ICON_SIZE, directedMarkerIcon, quantizeColor } from "@/features/tram-network/lib/marker-icon"
import type { ForecastMarker, StopRef, TramGraphGeoJson, TramPath } from "@/features/tram-network/types"

const BASEMAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

/**
 * Bundled maplibre cannot find its own worker and fails silently when it cannot — a blank
 * canvas with working controls. `maplibreWorkerAssets` in vite.config.ts publishes the
 * worker at this fixed path in both dev and build.
 */
setWorkerUrl(`${import.meta.env.BASE_URL}maplibre/maplibre-gl-worker.mjs`)

const MAX_BOUNDS_MARGIN = 0.12
const FIT_PADDING = 40

/**
 * Pen the camera to the network without hiding any of it.
 *
 * maxBounds refuses to show anything outside itself, which makes it a second, hidden
 * floor on zoom: the graph's own box plus a margin is narrower than the viewport that
 * frames the graph whenever the container's aspect ratio differs from the box's, so the
 * network cannot be seen whole. The limit is therefore taken from the viewport that
 * frames the graph, not from the graph. Both it and minZoom depend on the container
 * size, so this runs again on every resize.
 */
function applyLimits(map: MapLibreMap, bounds: Bounds): void {
  const camera = map.cameraForBounds(bounds as LngLatBoundsLike, { padding: FIT_PADDING })
  if (camera?.zoom === undefined || camera.center === undefined) return

  const restore = { center: map.getCenter(), zoom: map.getZoom() }
  map.setMaxBounds(undefined)
  map.setMinZoom(undefined)

  map.jumpTo({ center: camera.center, zoom: camera.zoom })
  const framed = map.getBounds()
  map.setMinZoom(camera.zoom)
  map.setMaxBounds(
    padBounds(
      [framed.getWest(), framed.getSouth(), framed.getEast(), framed.getNorth()],
      MAX_BOUNDS_MARGIN,
    ) as LngLatBoundsLike,
  )
  map.jumpTo(restore)
}

const MOSCOW_CENTER: [number, number] = [37.62, 55.75]
const MOSCOW_ZOOM = 9.6
const MOSCOW_FORECAST_BOUNDS: Bounds = [37.35, 55.52, 37.93, 55.96]

const COLOR = {
  track: "#8d99a6",
  trackDim: "#c9ced4",
  route: "#d9342b",
  path: "#246b88",
  stop: "#ffffff",
  stopStroke: "#6b7580",
  from: "#1f7a4d",
  to: "#d9342b",
  selected: "#246b88",
}

interface TramMapProps {
  readonly network: TramGraphGeoJson | undefined
  readonly routeGeoJson: TramGraphGeoJson | undefined
  readonly selectedRoute: string | null
  readonly path: TramPath | undefined
  readonly fromStop: StopRef | null
  readonly toStop: StopRef | null
  readonly selectedStop: StopRef | null
  readonly onStopClick: (stopId: number) => void
  readonly interactiveNetworkStops?: boolean
  readonly forecastMarkers?: readonly ForecastMarker[]
  readonly fitForecastMarkers?: boolean
  readonly onForecastStopClick?: (stopId: number) => void
}

export function TramMap(props: TramMapProps) {
  const { network, routeGeoJson, selectedRoute, path, fromStop, toStop, selectedStop, onStopClick } =
    props

  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const framedRef = useRef(false)
  const forecastFrameRef = useRef("")
  const graphBoundsRef = useRef<Bounds | null>(null)
  const [ready, setReady] = useState(false)
  const [styleError, setStyleError] = useState(false)
  const [attempt, setAttempt] = useState(0)

  const onForecastStopClickRef = useRef(props.onForecastStopClick)
  useEffect(() => { onForecastStopClickRef.current = props.onForecastStopClick }, [props.onForecastStopClick])
  const onStopClickRef = useRef(onStopClick)
  useEffect(() => {
    onStopClickRef.current = onStopClick
  }, [onStopClick])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return undefined

    let map: MapLibreMap
    try {
      map = new MapLibreMap({
        container,
        style: BASEMAP_STYLE,
        center: MOSCOW_CENTER,
        zoom: MOSCOW_ZOOM,
        attributionControl: false,
      })
    } catch {
      const failure = window.setTimeout(() => setStyleError(true), 0)
      return () => window.clearTimeout(failure)
    }
    mapRef.current = map

    map.addControl(new NavigationControl({ showCompass: false }), "top-right")
    map.addControl(new ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left")
    map.addControl(
      new AttributionControl({
        compact: true,
        customAttribution: `<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">${OSM_ATTRIBUTION}</a>`,
      }),
      "bottom-right",
    )

    map.on("error", () => setStyleError(true))
    const handleContextLost = () => setStyleError(true)
    map.getCanvas().addEventListener("webglcontextlost", handleContextLost)

    map.on("load", () => {
      map.addSource("network", { type: "geojson", data: EMPTY_COLLECTION })
      map.addSource("route", { type: "geojson", data: EMPTY_COLLECTION })
      map.addSource("path", { type: "geojson", data: EMPTY_COLLECTION })
      map.addSource("markers", { type: "geojson", data: EMPTY_COLLECTION })
      map.addSource("forecast", { type: "geojson", data: EMPTY_COLLECTION })

      map.addLayer({
        id: "network-track",
        type: "line",
        source: "network",
        filter: ["all", ["==", ["geometry-type"], "LineString"], ["!=", ["get", "geometry_quality"], "inferred"]],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": COLOR.track, "line-width": 1.6, "line-opacity": 0.85 },
      })
      map.addLayer({
        id: "network-stops",
        type: "circle",
        source: "network",
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 1.8, 12, 3.2, 16, 6],
          "circle-color": COLOR.stop,
          "circle-stroke-color": COLOR.stopStroke,
          "circle-stroke-width": 1.2,
        },
      })
      map.addLayer({
        id: "route-track",
        type: "line",
        source: "route",
        filter: ["all", ["==", ["geometry-type"], "LineString"], ["!=", ["get", "geometry_quality"], "inferred"]],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": COLOR.route, "line-width": 3.4 },
      })
      map.addLayer({
        id: "path-track",
        type: "line",
        source: "path",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": COLOR.path, "line-width": 5, "line-opacity": 0.9 },
      })
      map.addLayer({
        id: "route-stops",
        type: "circle",
        source: "route",
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 2.6, 12, 4.2, 16, 7],
          "circle-color": COLOR.stop,
          "circle-stroke-color": COLOR.route,
          "circle-stroke-width": 2,
        },
      })
      map.addLayer({
        id: "marker-points",
        type: "circle",
        source: "markers",
        paint: {
          "circle-radius": 8,
          "circle-color": [
            "match",
            ["get", "role"],
            "from",
            COLOR.from,
            "to",
            COLOR.to,
            COLOR.selected,
          ],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 2.5,
        },
      })

      map.addLayer({
        id: "forecast-points", type: "circle", source: "forecast", filter: ["!", ["has", "bearing"]],
        paint: {
          "circle-radius": ["get", "radius"],
          "circle-color": ["coalesce", ["get", "color"], ["case", ["get", "synthetic"], "#b86e13", "#246b88"]],
          "circle-stroke-color": "#1f2529",
          "circle-stroke-width": ["case", ["get", "selected"], 4, 1.5],
          "circle-opacity": 0.9,
        },
      })
      map.addLayer({
        id: "forecast-directed", type: "symbol", source: "forecast", filter: ["has", "bearing"],
        layout: {
          "icon-image": ["get", "icon"],
          "icon-size": ["/", ["*", 2, ["get", "radius"]], DIRECTED_ICON_SIZE],
          "icon-rotate": ["get", "bearing"],
          "icon-rotation-alignment": "map",
          "icon-offset": [DIRECTED_ICON_SIZE / 2 + 1, 0],
          "icon-allow-overlap": true,
          "icon-ignore-placement": true,
        },
      })

      setReady(true)
    })

    const pickLayers = props.interactiveNetworkStops === false ? ["forecast-points", "forecast-directed"] : ["forecast-points", "forecast-directed", "marker-points", "route-stops", "network-stops"]
    const boxAround = (event: MapMouseEvent, slop: number): [PointLike, PointLike] => [
      [event.point.x - slop, event.point.y - slop],
      [event.point.x + slop, event.point.y + slop],
    ]

    const handleClick = (event: MapMouseEvent) => {
      const layers = pickLayers.filter((id) => map.getLayer(id))
      if (layers.length === 0) return
      const hits = map.queryRenderedFeatures(boxAround(event, 6), { layers })
      for (const hit of hits) {
        const servingId: unknown = hit.properties?.forecast_stop_id
        if ((hit.layer.id === "forecast-points" || hit.layer.id === "forecast-directed") && typeof servingId === "number" && Number.isInteger(servingId)) {
          onForecastStopClickRef.current?.(servingId)
          return
        }
        const id = featureStopId(hit.properties)
        if (id !== null) {
          onStopClickRef.current(id)
          return
        }
      }
    }
    map.on("click", handleClick)

    const popup = new Popup({ closeButton: false, closeOnClick: false, offset: 10, className: "stop-tip" })
    const hoverLayers = props.interactiveNetworkStops === false
      ? ["forecast-points", "forecast-directed"]
      : ["forecast-points", "forecast-directed", "route-stops", "network-stops"]
    const handleMove = (event: MapMouseEvent) => {
      const layers = hoverLayers.filter((id) => map.getLayer(id))
      if (layers.length === 0) return
      const hit = map.queryRenderedFeatures(boxAround(event, 5), { layers })[0]
      map.getCanvas().style.cursor = hit ? "pointer" : ""
      const label = hit && (hit.layer.id === "forecast-points" || hit.layer.id === "forecast-directed")
        ? (typeof hit.properties?.label === "string" ? hit.properties.label : null)
        : hit ? featureStopLabel(hit.properties) : null
      if (!hit || !label || hit.geometry.type !== "Point") {
        popup.remove()
        return
      }
      popup.setLngLat(hit.geometry.coordinates as [number, number]).setText(label).addTo(map)
    }
    map.on("mousemove", handleMove)

    return () => {
      popup.remove()
      map.getCanvas().removeEventListener("webglcontextlost", handleContextLost)
      map.remove()
      mapRef.current = null
      framedRef.current = false
      forecastFrameRef.current = ""
      setReady(false)
    }
  }, [attempt, props.interactiveNetworkStops])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource<GeoJSONSource>("network")
    void source?.setData(network ? asFeatureCollection(network) : EMPTY_COLLECTION)

    if (!network || framedRef.current) return
    const bounds = boundsOfGraph(network)
    if (bounds) {
      graphBoundsRef.current = bounds
      applyLimits(map, bounds)
      map.fitBounds(bounds as LngLatBoundsLike, { padding: FIT_PADDING, duration: 0 })
      framedRef.current = true
    }
  }, [network, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const onResize = () => {
      const bounds = graphBoundsRef.current
      if (bounds) applyLimits(map, bounds)
    }
    map.on("resize", onResize)
    return () => {
      map.off("resize", onResize)
    }
  }, [ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !map.getLayer("network-track")) return
    const dimmed = selectedRoute !== null
    map.setPaintProperty("network-track", "line-color", dimmed ? COLOR.trackDim : COLOR.track)
    map.setPaintProperty("network-track", "line-opacity", dimmed ? 0.5 : 0.85)
    map.setPaintProperty("network-stops", "circle-opacity", dimmed ? 0.3 : 1)
    map.setPaintProperty("network-stops", "circle-stroke-opacity", dimmed ? 0.3 : 1)
  }, [selectedRoute, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource<GeoJSONSource>("route")
    void source?.setData(routeGeoJson ? asFeatureCollection(routeGeoJson) : EMPTY_COLLECTION)

    const bounds = boundsOfGraph(routeGeoJson)
    if (bounds) map.fitBounds(bounds as LngLatBoundsLike, { padding: 56, duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 600 })
  }, [routeGeoJson, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource<GeoJSONSource>("path")
    void source?.setData(pathCollection(path))

    if (!path?.found) return
    const bounds = boundsOfCoordinates(path.geometry)
    if (bounds) map.fitBounds(bounds as LngLatBoundsLike, { padding: 72, duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 600 })
  }, [path, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource<GeoJSONSource>("markers")
    void source?.setData(markerCollection(fromStop, toStop, selectedStop))
  }, [fromStop, toStop, selectedStop, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const markers = props.forecastMarkers ?? []
    if (!network && markers.length > 0 && !graphBoundsRef.current) {
      graphBoundsRef.current = MOSCOW_FORECAST_BOUNDS
      applyLimits(map, MOSCOW_FORECAST_BOUNDS)
    }
    if (props.fitForecastMarkers) {
      const coordinates = markers
        .filter((marker) => Number.isFinite(marker.longitude) && Number.isFinite(marker.latitude))
        .map((marker): [number, number] => [marker.longitude, marker.latitude])
      const frameKey = JSON.stringify(coordinates.map((coordinate) => coordinate.join(",")).sort())
      if (frameKey !== forecastFrameRef.current) {
        const bounds = boundsOfCoordinates(coordinates)
        if (bounds) map.fitBounds(bounds as LngLatBoundsLike, { padding: 56, maxZoom: 14, duration: 0 })
        forecastFrameRef.current = frameKey
      }
    } else forecastFrameRef.current = ""
    const max = Math.max(1, ...markers.map((marker) => marker.value))
    const directedIcon = (marker: ForecastMarker) => {
      const color = quantizeColor(marker.color ?? (marker.synthetic ? "#b86e13" : "#246b88"))
      const id = `forecast-directed-${color}-${marker.selected ? "selected" : "plain"}`
      if (!map.hasImage(id)) map.addImage(id, directedMarkerIcon(color, marker.selected), { pixelRatio: 2 })
      return id
    }
    void map.getSource<GeoJSONSource>("forecast")?.setData({
      type: "FeatureCollection",
      features: markers.map((marker) => ({
        type: "Feature", geometry: { type: "Point", coordinates: [marker.longitude, marker.latitude] },
        properties: { forecast_stop_id: marker.stopId, label: marker.label, value: marker.value,
          color: marker.color ?? null, selected: marker.selected, synthetic: marker.synthetic, radius: 6 + 7 * Math.sqrt(marker.value / max),
          ...(marker.bearing === undefined ? {} : { bearing: marker.bearing, icon: directedIcon(marker) }) },
      })),
    })
  }, [network, props.forecastMarkers, props.fitForecastMarkers, ready])

  return (
    <div className="tram-map-wrap">
      <div className="relative">
      <div className="tram-map" ref={containerRef} data-testid="tram-map"
        data-state={styleError ? "unavailable" : ready ? "ready" : "loading"}
        aria-label="Карта трамвайной сети" />
      {styleError && (
        <div className="tram-map-overlay" role="status">
          <p>Карта недоступна. Используйте список остановок и участков сети ниже.</p>
          <Button onClick={() => {
            setStyleError(false)
            setReady(false)
            setAttempt((value) => value + 1)
          }}>Повторить загрузку карты</Button>
        </div>
      )}
      </div>
      {(network?.metadata?.missing_geometry_edges ?? 0) > 0 && (
        <p role="status">Геометрия отсутствует для {network?.metadata?.missing_geometry_edges} участков. Прямые соединения не показаны как рельсы.</p>
      )}
      {network?.metadata?.synthetic && <p role="status">Синтетическая геометрия — демонстрационные данные.</p>}
      <details>
        <summary>Остановки и участки сети</summary>
        <div className="max-h-80 overflow-auto" tabIndex={0} role="region" aria-label="Прокручиваемый список сети">
          {!network ? <p>Список появится после загрузки графа сети.</p> : (
            <ul aria-label="Объекты локального графа">
              {(routeGeoJson ?? network).features.map((feature, index) => (
                <li key={index}>
                  {"id" in feature.properties ? (
                    <>
                      {props.interactiveNetworkStops === false ? <span>{feature.properties.name} · OSM {feature.properties.id}</span> : <Button variant="ghost" onClick={() => {
                        const id = featureStopId(feature.properties)
                        if (id !== null) onStopClick(id)
                      }}>
                        {feature.properties.name} · OSM {feature.properties.id}
                      </Button>}
                      <span> · {feature.geometry.coordinates.join(", ")}</span>
                    </>
                  ) : (
                    <span>OSM {feature.properties.source} → OSM {feature.properties.target} · {feature.properties.length_m} м · {feature.properties.geometry_quality === "inferred" ? "геометрия отсутствует" : feature.properties.geometry_quality === "synthetic" ? "синтетическая геометрия" : "геометрия источника"}</span>
                  )}
                  <span> · маршруты: {feature.properties.routes.join(", ") || "не указаны"}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </details>
    </div>
  )
}

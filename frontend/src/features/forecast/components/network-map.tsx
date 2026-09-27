import { lazy, Suspense } from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { bucketLabel, bucketStops } from "@/features/forecast/lib/bucket"
import type { ForecastResponse } from "@/features/forecast/types"
import { useNetworkGeoJson } from "@/features/tram-network/hooks/use-tram-network"
import type { ForecastMarker } from "@/features/tram-network/types"
import { formatForecastValue, intervalBounds } from "@/features/forecast/lib/values"
import { forecastUnit } from "@/features/forecast/lib/provenance"

const TramMap = lazy(() => import("@/features/tram-network/components/tram-map").then((module) => ({ default: module.TramMap })))
const ignoreNetworkStop = () => undefined

interface NetworkMapProps {
  snapshot: ForecastResponse
  timestamp: string | null
  selectedStopId: number | null
  onStopSelect: (id: number) => void
}

const reasons: Record<string, string> = {
  mapping_not_configured: "Соответствие остановкам OSM не настроено.",
  mapping_configuration_invalid: "Файл соответствий или версия графа недоступны.",
  forecast_graph_version_unavailable: "Версия графа для этого прогноза не указана.",
  graph_version_mismatch: "Версия графа не совпадает с прогнозом.",
  entity_version_mismatch: "Версия идентификаторов не совпадает с прогнозом.",
  synthetic_mapping_for_real_forecast: "Синтетические соответствия нельзя применять к реальному прогнозу.",
  run_metadata_unavailable: "Метаданные запуска недоступны.",
}

export function NetworkMap({ snapshot, timestamp, selectedStopId, onStopSelect }: NetworkMapProps) {
  const network = useNetworkGeoJson()
  const rows = bucketStops(snapshot, timestamp)
  const envelope = snapshot.run && snapshot.map?.run_id === snapshot.run.run_id
    && snapshot.map.entity_version === snapshot.run.entity_version
    && snapshot.map.graph_version === snapshot.run.graph_version ? snapshot.map : undefined
  const positions = envelope?.positions ?? []
  const stopNames = new Map(snapshot.stops.map((stop) => [stop.id, stop.name]))
  const markers: ForecastMarker[] = rows.flatMap((row) => {
    const position = positions.find((item) => item.stop_id === row.stop_id && item.direction_id === row.direction_id)
    if (!position || position.position_kind === "unavailable" || position.longitude === null || position.latitude === null) return []
    return [{ stopId: row.stop_id, longitude: position.longitude, latitude: position.latitude,
      value: row.predicted_passengers, label: stopNames.get(row.stop_id) ?? `Остановка ${row.stop_id}`,
      selected: row.stop_id === selectedStopId,
      synthetic: position.position_kind === "synthetic_demo" || Boolean(envelope?.synthetic),
    }]
  })
  const mapped = positions.some((position) => position.position_kind !== "unavailable" && position.longitude !== null && position.latitude !== null)
  return (
    <Card className="map-card">
      <CardHeader>
        <CardTitle>Прогноз на карте Москвы</CardTitle>
        <CardDescription>{timestamp ? `${bucketLabel(timestamp)} МСК` : "Интервал не выбран"} · {forecastUnit(snapshot)}</CardDescription>
      </CardHeader>
      <CardContent>
        {mapped ? <>
        <p>Размер точки — прогноз выбранного интервала.</p>
        {(!envelope || envelope.status !== "ready") && <p role="status">{reasons[envelope?.reason ?? "run_metadata_unavailable"] ?? "Не все остановки имеют однозначное соответствие OSM."}</p>}
        {envelope && <p>Сопоставление за выбранное окно — привязаны: {envelope.matched_count}; без соответствия: {envelope.unmatched_count}; неоднозначны: {envelope.ambiguous_count}.</p>}
        {positions.some((position) => position.position_kind === "synthetic_demo") && <p role="status">Демонстрационные координаты; соответствие остановкам OSM не установлено. Жёлтые точки — синтетические данные.</p>}
        {envelope?.synthetic && !positions.some((position) => position.position_kind === "synthetic_demo") && <p>Синтетические прогноз или соответствия; качество на реальных данных не подтверждено.</p>}
        {network.isLoading && <p role="status">Загрузка фоновой сети…</p>}
        {network.isError && <p role="alert">Фоновая сеть недоступна. Значения доступны в таблице. <Button variant="secondary" onClick={() => void network.refetch()}>Повторить фоновую сеть</Button></p>}
        <Suspense fallback={<Skeleton className="h-[420px]" />}>
          <TramMap network={network.data} routeGeoJson={undefined} selectedRoute={null} path={undefined}
            fromStop={null} toStop={null} selectedStop={null} onStopClick={ignoreNetworkStop}
            interactiveNetworkStops={false} forecastMarkers={markers} onForecastStopClick={onStopSelect} />
        </Suspense>
        </> : <p role="status">Координаты остановок для этого прогноза не сопоставлены: значения по остановкам — в таблице ниже. Карта остановок с прогнозом — на вкладке «Прогноз».</p>}
        {rows.length === 0 && <p role="status">Нет прогноза по остановкам для выбранного интервала. Значения других интервалов не подставляются.</p>}
        {rows.length > 0 && <div className="forecast-stop-table" tabIndex={0} role="region" aria-label="Прокручиваемая таблица остановок">
          <table aria-label="Прогноз остановок выбранного интервала">
            <thead><tr><th scope="col">Остановка</th><th scope="col">Направление</th><th scope="col">Прогноз</th><th scope="col">Интервал</th><th scope="col">Карта</th></tr></thead>
            <tbody>{rows.map((row) => {
              const position = positions.find((item) => item.stop_id === row.stop_id && item.direction_id === row.direction_id)
              return <tr key={`${row.stop_id}-${row.direction_id ?? "all"}`}>
                <th scope="row"><Button variant="ghost" onClick={() => onStopSelect(row.stop_id)} aria-pressed={row.stop_id === selectedStopId}>{stopNames.get(row.stop_id) ?? `Остановка ${row.stop_id}`}</Button></th>
                <td>{row.direction_id === "legacy-unspecified" ? "Не указано источником" : row.direction_id ?? "Все доступные (сумма)"}</td>
                <td>{formatForecastValue(row.predicted_passengers)} {snapshot.run?.unit ?? ""}</td>
                <td>{!intervalBounds(row) ? "недоступен" : `${formatForecastValue(row.lower_bound)}–${formatForecastValue(row.upper_bound)}`}</td>
                <td>{position?.position_kind === "synthetic_demo" ? "демо, без привязки OSM" : position?.status === "matched" ? `OSM ${position.osm_stop_id}` : position?.status === "ambiguous" ? "неоднозначно" : "нет соответствия"}</td>
              </tr>
            })}</tbody>
          </table>
        </div>}
        {rows.some((row) => row.bucket_end === null) && <p>Конец отдельных интервалов в исходном демонаборе не указан.</p>}
      </CardContent>
    </Card>
  )
}

import { ArrowDownLeft, ArrowUpRight, Search } from "lucide-react"
import { useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useStopDetail, useStopSearch } from "@/features/tram-network/hooks/use-tram-network"
import { formatMetres, stopLabel } from "@/features/tram-network/lib/network"
import type { StopDetail, StopRef } from "@/features/tram-network/types"

interface StopInspectorProps {
  readonly stopId: number | null
  readonly onSelect: (stopId: number) => void
  readonly onSetEndpoint: (stop: StopRef, role: "from" | "to") => void
}

function NeighbourGroup({ stop, direction }: { stop: StopDetail; direction: "out" | "in" }) {
  const neighbours = stop.neighbours.filter((neighbour) => neighbour.direction === direction)
  if (neighbours.length === 0) return null

  return (
    <div className="neighbour-group">
      <h4>
        {direction === "out" ? <ArrowUpRight aria-hidden="true" /> : <ArrowDownLeft aria-hidden="true" />}
        {direction === "out" ? "Отсюда можно уехать" : "Сюда приезжают с"}
      </h4>
      <ul>
        {neighbours.map((neighbour) => (
          <li key={`${direction}-${neighbour.id}`}>
            <span className="neighbour-name">{stopLabel(neighbour)}</span>
            <span className="neighbour-meta">
              {formatMetres(neighbour.length_m)} · {neighbour.routes.join(", ")}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function StopInspector({ stopId, onSelect, onSetEndpoint }: StopInspectorProps) {
  const [query, setQuery] = useState("")
  const search = useStopSearch(query)
  const detail = useStopDetail(stopId)
  const stop = detail.isError ? undefined : detail.data

  return (
    <Card className="stop-inspector">
      <CardHeader>
        <CardTitle>Остановка</CardTitle>
        <CardDescription>Найдите остановку или кликните точку на карте</CardDescription>
      </CardHeader>
      <CardContent className="stop-inspector-content">
        <div className="stop-search">
          <label htmlFor="tram-stop-search" className="sr-only">
            Поиск остановки
          </label>
          <Search aria-hidden="true" />
          <input
            id="tram-stop-search"
            type="search"
            value={query}
            placeholder="Например, Курский вокзал"
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>

        {query.trim().length >= 2 && search.isPending && <Skeleton className="h-20" />}
        {search.isError && (
          <div role="status"><p className="inline-error">Поиск остановок недоступен.</p>
            <Button disabled={search.isFetching} onClick={() => void search.refetch()}>Повторить: поиск остановок</Button></div>
        )}
        {!search.isError && search.data?.length === 0 && (
          <p className="stop-empty">Ничего не найдено — попробуйте другую часть названия.</p>
        )}
        {!search.isError && search.data && search.data.length > 0 && (
          <ul className="stop-results" aria-label="Результаты поиска">
            {search.data.map((result) => (
              <li key={result.id}>
                <button type="button" onClick={() => onSelect(result.id)}>
                  <span className="stop-name">{stopLabel(result)}</span>
                  <span className="stop-meta">{result.routes.join(", ")}</span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {stopId !== null && detail.isPending && <Skeleton className="h-40" />}
        {detail.isError && (
          <div role="status"><p className="inline-error">Не удалось загрузить остановку.</p>
            <Button disabled={detail.isFetching} onClick={() => void detail.refetch()}>Повторить: остановка</Button></div>
        )}
        {stop && (
          <div className="stop-detail">
            <h3>{stopLabel(stop)}</h3>
            <p className="stop-coordinates">
              OSM {stop.id} · {stop.latitude.toFixed(5)}, {stop.longitude.toFixed(5)}
            </p>
            <div className="route-chips" aria-label="Маршруты через остановку">
              {stop.routes.map((ref) => (
                <Badge key={ref}>{ref}</Badge>
              ))}
            </div>
            <div className="stop-endpoint-actions">
              <Button variant="ghost" onClick={() => onSetEndpoint(stop, "from")}>
                Откуда
              </Button>
              <Button variant="ghost" onClick={() => onSetEndpoint(stop, "to")}>
                Куда
              </Button>
            </div>
            <NeighbourGroup stop={stop} direction="out" />
            <NeighbourGroup stop={stop} direction="in" />
          </div>
        )}
        {stopId === null && !search.data && (
          <p className="stop-empty">
            Остановка не выбрана. Граф направленный: у каждой точки свои входящие и исходящие
            участки.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

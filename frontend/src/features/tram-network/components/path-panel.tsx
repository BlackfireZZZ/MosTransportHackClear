import { ArrowRight, RotateCcw, Shuffle } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { formatCount, formatKm, stopLabel, summarisePath } from "@/features/tram-network/lib/network"
import type { StopRef, TramPath } from "@/features/tram-network/types"

interface PathPanelProps {
  readonly fromStop: StopRef | null
  readonly toStop: StopRef | null
  readonly path: TramPath | undefined
  readonly isPending: boolean
  readonly isError: boolean
  readonly onSwap: () => void
  readonly onClear: () => void
  readonly onRetry: () => void
}

export function PathPanel(props: PathPanelProps) {
  const { fromStop, toStop, path, isPending, isError, onSwap, onClear, onRetry } = props
  const bothPicked = fromStop !== null && toStop !== null
  const samePoint = bothPicked && fromStop.id === toStop.id
  const summary = path ? summarisePath(path) : null

  return (
    <Card className="path-panel">
      <CardHeader>
        <CardTitle>Кратчайший путь</CardTitle>
        <CardDescription>По рельсам, с учётом направления движения</CardDescription>
      </CardHeader>
      <CardContent className="path-panel-content">
        <ol className="path-endpoints">
          <li>
            <span>Откуда</span>
            <strong>{fromStop ? stopLabel(fromStop) : "не выбрано"}</strong>
          </li>
          <li>
            <span>Куда</span>
            <strong>{toStop ? stopLabel(toStop) : "не выбрано"}</strong>
          </li>
        </ol>

        <div className="path-actions">
          <Button variant="ghost" onClick={onSwap} disabled={!bothPicked}>
            <Shuffle aria-hidden="true" />
            Поменять местами
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClear}
            disabled={!fromStop && !toStop}
            aria-label="Сбросить точки маршрута"
          >
            <RotateCcw />
          </Button>
        </div>

        {!bothPicked && (
          <p className="path-hint">
            Выберите две остановки — кнопками «Откуда» и «Куда» в карточке остановки.
          </p>
        )}
        {samePoint && <p className="path-hint">Начало и конец совпадают.</p>}
        {bothPicked && !samePoint && isPending && <Skeleton className="h-24" />}
        {isError && (
          <div className="path-error">
            <p className="inline-error">Не удалось построить путь.</p>
            <Button onClick={onRetry}>Повторить</Button>
          </div>
        )}

        {summary?.status === "unreachable" && (
          <div className="path-unreachable" role="status">
            <strong>Пути нет</strong>
            <p>{summary.explanation}</p>
            <p className="path-reason">{summary.reason}</p>
          </div>
        )}

        {summary?.status === "found" && (
          <div className="path-result" data-testid="tram-path-result" aria-live="polite">
            {path?.geometry_quality === "inferred" && (
              <p role="status">Геометрия отсутствует для {path.missing_geometry_edges} участков пути. Линия пути скрыта; последовательность остановок доступна ниже.</p>
            )}
            {path?.geometry_quality === "synthetic" && <p role="status">Путь использует синтетическую геометрию.</p>}
            <div className="path-numbers">
              <span>
                <strong>{formatKm(summary.lengthM)}</strong>
                <small>по рельсам</small>
              </span>
              <ArrowRight aria-hidden="true" />
              <span>
                <strong>{formatCount(summary.stopCount)}</strong>
                <small>остановок</small>
              </span>
            </div>
            {summary.routes.length > 0 && (
              <div className="route-chips" aria-label="Маршруты на пути">
                {summary.routes.map((ref) => (
                  <Badge key={ref}>{ref}</Badge>
                ))}
              </div>
            )}
            {path?.stops && (
              <ol className="path-stops" aria-label="Остановки на пути">
                {path.stops.map((stop, index) => (
                  <li key={`${stop.id}-${index}`}>{stopLabel(stop)}</li>
                ))}
              </ol>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

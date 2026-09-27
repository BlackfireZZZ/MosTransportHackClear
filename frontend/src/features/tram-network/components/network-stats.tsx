import { componentLabel, formatCount, formatMetres } from "@/features/tram-network/lib/network"
import type { NetworkStats as NetworkStatsData } from "@/features/tram-network/types"

interface NetworkStatsProps {
  readonly stats: NetworkStatsData
}

export function NetworkStats({ stats }: NetworkStatsProps) {
  const termini = stats.degree_histogram["1"] ?? 0
  const junctions = Object.entries(stats.degree_histogram)
    .filter(([degree]) => Number(degree) >= 3)
    .reduce((total, [, count]) => total + count, 0)

  return (
    <section className="kpi-grid" aria-label="Показатели графа сети">
      <article className="kpi">
        <span>Остановки</span>
        <strong>{formatCount(stats.stops)}</strong>
        <small>
          {termini} конечных · {junctions} узлов ветвления
        </small>
      </article>
      <article className="kpi">
        <span>Участки пути</span>
        <strong>{formatCount(stats.edges)}</strong>
        <small>
          направленные · медиана {formatMetres(stats.segment_length_m.median)}
        </small>
      </article>
      <article className="kpi">
        <span>Номера маршрутов</span>
        <strong>{formatCount(stats.routes)}</strong>
        <small>
          {stats.route_relations !== null && stats.route_relations !== undefined
            ? `${formatCount(stats.route_relations)} связей PTv2, по одной на направление`
            : "уникальные refs, включая «А» и «т1»"}
        </small>
      </article>
      <article className="kpi">
        <span>Длина путей</span>
        <strong>
          {new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 }).format(
            stats.total_length_km,
          )}
        </strong>
        <small>км, измерено по рельсам</small>
      </article>
      <article className="kpi kpi-wide">
        <span>Компоненты связности</span>
        <strong className="kpi-text">
          {stats.components.map((component) => formatCount(component.size)).join(" + ")} остановок
        </strong>
        <small>
          {stats.components
            .map((component, index) => `${componentLabel(index)}: ${component.routes.join(", ")}`)
            .join(" · ")}
        </small>
      </article>
    </section>
  )
}

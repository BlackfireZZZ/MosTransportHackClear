import { Button } from "@/components/ui/button"
import { formatTimestamp, OSM_ATTRIBUTION } from "@/features/tram-network/lib/network"
import type { GeoJsonMetadata, OverpassStatus } from "@/features/tram-network/types"

interface NetworkProvenanceProps {
  readonly metadata: GeoJsonMetadata | undefined
  readonly overpass: OverpassStatus | undefined
  readonly isError: boolean
  readonly isFetching: boolean
  readonly onRetry: () => void
}

export function NetworkProvenance({ metadata, overpass, isError, isFetching, onRetry }: NetworkProvenanceProps) {
  return (
    <footer className="provenance">
      <dl>
        <div>
          <dt>Срез OSM</dt>
          <dd>{formatTimestamp(metadata?.osm_data_timestamp)}</dd>
        </div>
        <div>
          <dt>Граф собран</dt>
          <dd>{formatTimestamp(metadata?.generated_at)}</dd>
        </div>
        <div>
          <dt>Источник</dt>
          <dd>{metadata?.source ?? "OpenStreetMap via Overpass API"}</dd>
        </div>
        <div>
          <dt>Overpass</dt>
          <dd>
            <span
              className={overpass?.reachable ? "provenance-dot is-up" : "provenance-dot is-down"}
              aria-hidden="true"
            />
            {isError
              ? "проверка недоступна; работа с локальным графом не требует Overpass"
              : overpass === undefined
              ? "проверяется"
              : overpass.reachable
                ? "доступен"
                : `недоступен — ${overpass.detail ?? "причина не указана"}`}
            {(isError || overpass?.reachable === false) && (
              <Button variant="ghost" disabled={isFetching} onClick={onRetry}>Повторить: Overpass</Button>
            )}
          </dd>
        </div>
      </dl>
      <p className="provenance-license">
        {OSM_ATTRIBUTION}
        {metadata?.license && metadata.license !== "ODbL 1.0" ? ` · ${metadata.license}` : ""}
      </p>
    </footer>
  )
}

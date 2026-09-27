import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { forecastCsv } from "@/features/forecast/lib/export"
import { snapshotFreshness } from "@/features/forecast/lib/provenance"
import type { ForecastFilters } from "@/features/forecast/lib/selection"
import type { ForecastResponse } from "@/features/forecast/types"

interface ForecastExportProps {
  snapshot?: ForecastResponse
  updatedAt: number
  failed: boolean
  pollInterval: number | false
  selectedTimestamp: string | null
  requestedFilters: ForecastFilters
}

export function ForecastExport(props: ForecastExportProps) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 10_000)
    return () => window.clearInterval(timer)
  }, [])
  const stale = snapshotFreshness(props.updatedAt, now, props.failed, props.pollInterval).stale
  const download = () => {
    if (!props.snapshot) return
    const exportedAt = Date.now()
    setNow(exportedAt)
    const csv = forecastCsv(props.snapshot, { ...props, exportedAt })
    if (csv === null) return
    const url = URL.createObjectURL(new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" }))
    const link = document.createElement("a")
    link.href = url
    link.download = `tramflow-route-${props.snapshot.route.id}-${props.snapshot.horizon}-${new Date(exportedAt).toISOString().slice(0, 10)}.csv`
    document.body.append(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <Button variant="secondary" disabled={!props.snapshot?.points.length} onClick={download}>
    {props.snapshot?.points.length && stale ? "Скачать CSV — устаревший снимок" : "Скачать CSV выбранного окна"}
  </Button>
}

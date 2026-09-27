import { Pause, Play } from "lucide-react"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"

export const STEP_MS = 1000

interface TimelineProps {
  points: ReadonlyArray<{ timestamp: string }>
  value: string
  format: (timestamp: string) => string
  onChange: (timestamp: string) => void
}

export function Timeline({ points, value, format, onChange }: TimelineProps) {
  const [playing, setPlaying] = useState(false)
  const index = Math.max(0, points.findIndex((point) => point.timestamp === value))
  const last = points.length - 1

  useEffect(() => {
    if (!playing) return
    const timer = window.setTimeout(() => {
      const next = index + 1
      if (next > last) { setPlaying(false); return }
      onChange(points[next].timestamp)
      if (next === last) setPlaying(false)
    }, STEP_MS)
    return () => window.clearTimeout(timer)
  }, [playing, index, last, points, onChange])

  const toggle = () => {
    if (playing) { setPlaying(false); return }
    if (index >= last) onChange(points[0].timestamp)
    setPlaying(true)
  }

  return <div className="planning-timeline" role="group" aria-label="Проигрывание прогноза по времени">
    <Button variant="secondary" disabled={last < 1} onClick={toggle}>{playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}{playing ? "Пауза" : "Воспроизвести"}</Button>
    <label>Интервал на карте · МСК
      <input type="range" min={0} max={Math.max(0, last)} step={1} value={index} aria-valuetext={format(value)} disabled={last < 1}
        onChange={(event) => { setPlaying(false); onChange(points[Number(event.target.value)].timestamp) }} />
    </label>
    <output>{format(value)}</output>
  </div>
}

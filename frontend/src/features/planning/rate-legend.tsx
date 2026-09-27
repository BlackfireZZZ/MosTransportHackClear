import { RATE_STOPS, ZERO_COLOR } from "./flow-color"

const top = RATE_STOPS.at(-1)!.rate

export function RateLegend() {
  const gradient = `linear-gradient(to right, ${RATE_STOPS.map((stop) => `${stop.color} ${(stop.rate / top) * 100}%`).join(", ")})`
  return <div className="rate-legend">
    <p>Цвет точки — посадок в час на остановке · размер — посадок за интервал</p>
    <div className="rate-legend-scale" role="img" aria-label={`Шкала интенсивности посадок в час: серый — 0; от жёлтого через оранжевый и красный к бордовому — от 1 до ${top} и больше посадок в час`}>
      <span className="rate-legend-zero"><i style={{ backgroundColor: ZERO_COLOR }} />0</span>
      <div className="rate-legend-track">
        <div className="rate-legend-bar" style={{ background: gradient }} />
        <div className="rate-legend-ticks">{RATE_STOPS.map((stop) => <span key={stop.rate} style={{ left: `${(stop.rate / top) * 100}%` }}>{stop.rate === 0 ? ">0" : stop.rate === top ? `${top}+` : stop.rate}</span>)}</div>
      </div>
      <span className="rate-legend-unit">посадок/ч</span>
    </div>
  </div>
}

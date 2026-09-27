const labels: Record<string, string> = { base: "Без внешних источников", calendar: "Календарь", weather: "Погода", traffic: "Дорожный трафик", news: "Новости", all: "Все источники", incumbent_robust28: "Статистический ориентир" }
interface Score { origin: string; variant: string; score: number; horizon_days: number }
function evaluationScores(raw: string): Score[] {
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object" || !("scores" in parsed) || !Array.isArray(parsed.scores)) return []
    return parsed.scores.filter((row: unknown): row is Score => Boolean(row) && typeof row === "object" && row !== null && "origin" in row && typeof row.origin === "string" && "variant" in row && typeof row.variant === "string" && "score" in row && typeof row.score === "number" && Number.isFinite(row.score) && "horizon_days" in row && row.horizon_days === 61)
  } catch { return [] }
}
export function PlanningEvaluation({ raw }: { raw: string }) {
  const scores = evaluationScores(raw)
  const origins = [...new Set(scores.map((row) => row.origin))].sort()
  const variants = [...new Set(scores.map((row) => row.variant))]
  return <details><summary>Проверка экспериментальной модели</summary>
    <p>Оценка 1 − WAPE на трёх исторических окнах по 61 дню; выше — лучше. Эти результаты не являются баллом скрытого теста и не подтверждают годовую точность.</p>
    {scores.length > 0 ? <div className="forecast-stop-table" tabIndex={0} role="region" aria-label="Сравнение качества внешних источников"><table><thead><tr><th>Вариант</th>{origins.map((origin) => <th key={origin}>{origin}</th>)}</tr></thead><tbody>{variants.map((variant) => <tr key={variant}><th scope="row">{labels[variant] ?? variant}</th>{origins.map((origin) => <td key={origin}>{scores.find((row) => row.variant === variant && row.origin === origin)?.score.toFixed(5) ?? "—"}</td>)}</tr>)}</tbody></table></div> : <p>Сводная таблица оценки недоступна; исходные метаданные сохранены в CSV.</p>}
  </details>
}

import { useEffect, useState } from "react"
import { ExternalLink, RotateCcw } from "lucide-react"

import { getBenchmarkResult, type BenchmarkResult } from "@/api/benchmark"
import { kindLabels, levels, repoLink, type Level } from "./architecture-data"
import "./architecture.css"

const levelOrder: Level[] = ["overview", "request", "deployment"]
const graphTitles: Record<Level, string> = {
  overview: "КОНКУРСНЫЙ МАРШРУТНО-ЧАСОВОЙ ПОТОК",
  request: "ПУТЬ ВЫБРАННОГО HTTP-ЗАПРОСА",
  deployment: "ПАКЕТНАЯ ПУБЛИКАЦИЯ И ONLINE-СЕРВИС",
}
const benchmarkGroups = [
  { title: "Опубликованный ряд", detail: "Чтение из PostgreSQL", matches: (id: string) => id.startsWith("get_") },
  { title: "Утверждённый сценарий", detail: "POST approved", matches: (id: string) => id.startsWith("post_approved_") },
  { title: "Оценочные остановки", detail: "POST stop_model", matches: (id: string) => id.startsWith("post_stop_") },
  { title: "Ошибки и границы", detail: "Проверка отказа", matches: (id: string) => !id.startsWith("get_") && !id.startsWith("post_approved_") && !id.startsWith("post_stop_") },
]
const edgeLabels: Record<Level, string[]> = {
  overview: ["сверенные часы", "признаки", "CSV + SHA", "активный ряд", "JSON"],
  request: ["фильтр", "HTTP", "проверка", "точки", "ответ"],
  deployment: ["архив", "bundle", "строки", "выборка", "JSON"],
}
const desktopPaths = [
  "M215 238 C254 229 230 122 270 110",
  "M460 110 H585",
  "M775 110 C823 119 805 236 855 250",
  "M855 250 C805 264 823 381 775 390",
  "M585 390 H460",
]
const mobilePaths = [
  "M175 105 H215",
  "M300 165 V285",
  "M300 405 V525",
  "M215 585 H175",
  "M90 525 V405",
]

function GraphLines({ paths, mobile, selectedIndex }: { paths: string[]; mobile?: boolean; selectedIndex: number }) {
  return <svg className={`architecture-graph-lines ${mobile ? "mobile" : "desktop"}`} viewBox={mobile ? "0 0 390 750" : "0 0 1100 500"} preserveAspectRatio="none" aria-hidden="true">
    <defs><marker id={mobile ? "architecture-arrow-mobile" : "architecture-arrow-desktop"} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="context-stroke" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></marker></defs>
    {paths.map((path, index) => <path key={path} d={path} className={selectedIndex === index || selectedIndex === index + 1 ? "is-related" : ""} markerEnd={`url(#architecture-arrow-${mobile ? "mobile" : "desktop"})`} />)}
  </svg>
}

function SpeedPanel() {
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    let active = true
    getBenchmarkResult().then((value) => {
      if (active) { setResult(value); setError(null); setLoading(false) }
    }).catch((cause: unknown) => {
      if (active) { setError(cause instanceof Error ? cause.message : "Ошибка загрузки"); setLoading(false) }
    })
    return () => { active = false }
  }, [revision])

  return <section className="architecture-speed" aria-labelledby="speed-title">
    <div className="architecture-section-heading"><div><p className="architecture-kicker">04 / ПРОВЕРКА ПОСЛЕ ДОРАБОТКИ</p><h2 id="speed-title">Скорость и узкие места</h2></div><a href={repoLink("docs/benchmark/README.md")} target="_blank" rel="noreferrer">Протокол замера <ExternalLink aria-hidden="true" /></a></div>
    <p className="architecture-speed-lead">Изолированный HTTP-замер опубликованного прогноза и сценариев планирования. Полный протокол, условия и сырой JSON доступны по ссылке выше.</p>
    {loading && <div role="status" className="architecture-metric-state">Загружаем файл результатов…</div>}
    {error && <div role="alert" className="architecture-metric-state architecture-metric-error"><strong>Результат недоступен</strong><span>{error}</span><button type="button" onClick={() => { setLoading(true); setRevision((value) => value + 1) }}><RotateCcw aria-hidden="true" /> Повторить</button></div>}
    {!loading && !error && result && <>
      {result.runs.length > 0 && <div className="architecture-measurement-summary">
        <strong>{result.runs.length} сценариев · {result.runs.reduce((sum, run) => sum + run.requests, 0).toLocaleString("ru-RU")} запросов · {result.runs.reduce((sum, run) => sum + run.errors, 0)} ошибок</strong>
        <span>{result.runs[0].conditions.host_cpu} · {result.runs[0].conditions.host_arch} · backend {result.runs[0].conditions.backend_cpus} vCPU / {(result.runs[0].conditions.backend_memory_bytes / 2 ** 30).toFixed(0)} GiB / swap {result.runs[0].conditions.backend_swap_bytes} · {result.runs[0].conditions.workers} worker · {result.runs[0].conditions.postgres_version} отдельно.</span>
        <span>По 5 прогревочных и 200 измеряемых запросов на сценарий, concurrency 4. Опубликованный GET достигает сотен RPS; остановочный и годовой POST превышают ориентир p95 200–300 мс.</span>
      </div>}
      {result.runs.length === 0 && <div role="status" className="architecture-metric-state"><strong>Замер ожидается</strong><span>Профиль и проверки готовы; чисел текущего прогона пока нет.</span></div>}
      {result.required_cases.length === 0 && <div role="status" className="architecture-metric-state">Перечень режимов пуст. Проверьте версионированный файл.</div>}
      <div className="architecture-cases" aria-label="Сценарии замера">
        {benchmarkGroups.map((group) => {
          const items = result.required_cases.filter((item) => group.matches(item.id))
          return items.length > 0 && <section className="architecture-case-group" key={group.title}><div className="architecture-case-group-heading"><h3>{group.title}</h3><span>{group.detail}</span></div><ul>{items.map((item) => {
            const run = result.runs.find((candidate) => candidate.case_id === item.id)
            return <li className="architecture-case" key={item.id}><span>{item.label}</span><span className="architecture-case-status">{run ? "Измерено" : "Ожидает замера"}</span>{run && <small><strong>{run.rps.toFixed(1)}</strong> RPS · p50 / p95 / p99 <strong>{run.latency_ms.p50.toFixed(1)} / {run.latency_ms.p95.toFixed(1)} / {run.latency_ms.p99.toFixed(1)}</strong> мс · HTTP {Object.keys(run.http_status_counts).join("/")} · CPU до {run.backend_cpu_percent_sample_max.toFixed(1)}% · RAM до {(run.backend_memory_bytes_sample_max / 2 ** 20).toFixed(1)} MiB · ошибок {run.errors}{run.latency_ms.p95 > 300 ? " · выше ориентира p95" : ""}</small>}</li>
          })}</ul></section>
        })}
      </div>
      {result.runs.length > 0 && <p className="architecture-source">Источник: version {result.schema_version}, {result.updated_at ?? "дата отсутствует"}, commit {result.runs[0].conditions.git_sha.slice(0, 7)}, SHA-256 raw {result.source_sha256 ?? "не указан"}. CPU/RAM — дискретные выборки, не непрерывный пик; PostgreSQL вне лимита backend. Замер не проверяет длительную нагрузку и масштабирование репликами.</p>}
    </>}
    <p className="architecture-history"><strong>Исторический замер · 27.09.2026:</strong> 200 запросов, concurrency 4, backend 2 CPU / 4 GiB без swap, PostgreSQL отдельно. Остановочный режим измерен отдельно без CPU-лимита. Источник — <a href={repoLink("docs/analysis/2026-09-27-forecast-criteria/README.md")} target="_blank" rel="noreferrer">отчёт с исходными условиями</a>. Эти результаты не сравниваются с текущим изолированным стендом как равные.</p>
    <p className="architecture-hypothesis"><strong>Гипотеза:</strong> расчёт остановочного и годового сценариев может быть дороже чтения опубликованного ряда. Узкое место можно назвать установленным только после замеров по этапам.</p>
  </section>
}

export function ArchitectureView() {
  const [level, setLevel] = useState<Level>("overview")
  const [selectedId, setSelectedId] = useState<string | null>("raw")
  const current = levels[level]
  const selectedIndex = current.nodes.findIndex((node) => node.id === selectedId)
  const selected = selectedIndex >= 0 ? current.nodes[selectedIndex] : current.branches?.find((node) => node.id === selectedId) ?? null
  const selectedSide = selectedIndex < 0 && selected !== null

  function chooseLevel(next: Level) {
    setLevel(next)
    setSelectedId(levels[next].nodes[0]?.id ?? null)
  }

  return <div className="architecture-page">
    <header className="architecture-intro"><div><p className="architecture-kicker">СИСТЕМА ПРИНЯТИЯ РЕШЕНИЙ · МОСКВА</p><h2>От валидации до решения диспетчера</h2><p>Три вида одной реализованной системы: происхождение прогноза, путь HTTP-запроса и контейнерное развёртывание.</p></div><div className="architecture-intro-mark" aria-hidden="true">TF<span>01—03</span></div></header>
    <section className="architecture-stage" aria-label="Интерактивная схема TramFlow">
      <div className="architecture-levels" role="group" aria-label="Уровень схемы">{levelOrder.map((item, index) => <button key={item} type="button" className={item === level ? "is-active" : ""} aria-pressed={item === level} onClick={() => chooseLevel(item)}><span>0{index + 1}</span>{levels[item].label}</button>)}</div>
      <div className="architecture-stage-heading"><div><p className="architecture-kicker">{current.eyebrow}</p><h3>{current.label}</h3><p>{current.description}</p></div><span className="architecture-stage-count">НАЖМИТЕ НА УЗЕЛ → ДЕТАЛИ</span></div>
      <div className="architecture-legend" aria-label="Происхождение данных">{(["observed", "inferred", "experimental", "qualitative"] as const).map((kind) => <span className={`architecture-kind kind-${kind}`} key={kind}>{kindLabels[kind]}</span>)}</div>
      <div className="architecture-graph" role="group" aria-label={`${current.label}: схема потока`} data-graph-title={graphTitles[level]}>
        <GraphLines paths={desktopPaths} selectedIndex={selectedIndex} />
        <GraphLines paths={mobilePaths} mobile selectedIndex={selectedIndex} />
        {edgeLabels[level].map((label, index) => <span key={label} className={`architecture-edge-label architecture-edge-${index}${selectedIndex === index || selectedIndex === index + 1 ? " is-related" : ""}`}>{label}</span>)}
        {current.nodes.map((node, index) => {
          const related = selectedIndex >= 0 && Math.abs(selectedIndex - index) <= 1
          return <button key={node.id} type="button" className={`architecture-graph-node architecture-position-${index} kind-border-${node.kind}${node.id === selectedId ? " is-selected" : ""}${related ? " is-related" : ""}`} aria-pressed={node.id === selectedId} aria-controls="architecture-details" onClick={() => setSelectedId(node.id)}><span className="architecture-node-index">{String(index + 1).padStart(2, "0")}</span><strong className="architecture-node-title">{node.title}</strong><span className="architecture-node-subtitle">{node.subtitle}</span><span className="architecture-node-kind">{kindLabels[node.kind]}</span></button>
        })}
      </div>
      {current.branches && <details className="architecture-branches"><summary><span><strong>Отдельные сценарии и инструменты</strong><small>Остановки, внешние факторы, год и планирование сети — вне конкурсного CSV</small></span><span className="architecture-branch-count">{current.branches.length} ВЕТКИ</span></summary><div className="architecture-branch-grid">{current.branches.map((node) => <button key={node.id} type="button" className={`architecture-branch${selectedId === node.id ? " is-selected" : ""}`} aria-pressed={selectedId === node.id} aria-controls="architecture-details" onClick={() => setSelectedId(node.id)}><span className={`architecture-kind kind-${node.kind}`}>{kindLabels[node.kind]}</span><strong>{node.title}</strong><small>{node.subtitle}</small></button>)}</div></details>}
      <div className="architecture-details" id="architecture-details" aria-live="polite">{selected ? <><div className="architecture-details-title"><div><p className="architecture-kicker">{selectedSide ? "ОТДЕЛЬНАЯ ВЕТКА" : `ВЫБРАННЫЙ УЗЕЛ · ${String(selectedIndex + 1).padStart(2, "0")}`}</p><h4>{selected.title}</h4></div><span className={`architecture-kind kind-${selected.kind}`}>{kindLabels[selected.kind]}</span></div><p className="architecture-purpose">{selected.purpose}</p><dl><div><dt>Вход</dt><dd>{selected.input}</dd></div><div><dt>Выход</dt><dd>{selected.output}</dd></div><div><dt>Происхождение</dt><dd>{selected.provenance}</dd></div><div><dt>Ограничение</dt><dd>{selected.limit}</dd></div></dl><a className="architecture-module" href={repoLink(selected.path)} target="_blank" rel="noreferrer"><span>Модуль / документ</span><strong>{selected.path}</strong><ExternalLink aria-hidden="true" /></a></> : <p>Для этого уровня нет узлов.</p>}</div>
    </section>
    <aside className="architecture-boundary"><strong>Граница конкурсного прогноза</strong><p>14 640 маршрутно-часовых значений формируются без оценочных остановок, направлений, экспериментальных внешних факторов и графа OSM. GTFS описывает сеть и расписание; он не фиксирует место фактической посадки.</p></aside>
    <SpeedPanel />
  </div>
}

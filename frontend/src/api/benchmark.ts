export type BenchmarkResult = {
  schema_version: number
  status: "pending" | "measured"
  updated_at: string | null
  source_sha256?: string
  required_cases: { id: string; label: string }[]
  runs: Array<{
    case_id: string
    requests: number
    warmup: number
    concurrency: number
    rps: number
    errors: number
    http_status_counts: Record<string, number>
    backend_cpu_percent_sample_max: number
    backend_memory_bytes_sample_max: number
    latency_ms: { p50: number; p95: number; p99: number }
    conditions: { backend_cpus: number; backend_memory_bytes: number; backend_swap_bytes: number; workers: number; postgres_scope: string; postgres_version: string; host_cpu: string; host_arch: string; git_sha: string; compose_sha256: string; artifact_sha256: string }
  }>
}

export async function getBenchmarkResult(): Promise<BenchmarkResult> {
  const response = await fetch("/benchmark-results.json", { cache: "no-store" })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  const data: unknown = await response.json()
  if (!data || typeof data !== "object" || !("schema_version" in data) || data.schema_version !== 1 || !("required_cases" in data) || !Array.isArray(data.required_cases) || !("runs" in data) || !Array.isArray(data.runs)) {
    throw new Error("Несовместимый формат результата замера")
  }
  return data as BenchmarkResult
}

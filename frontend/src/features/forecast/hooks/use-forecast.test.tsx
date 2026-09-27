import { onlineManager, QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { api } from "@/api/client"
import { useForecast } from "./use-forecast"

vi.mock("@/api/client", () => ({ api: { forecast: vi.fn() } }))
afterEach(() => { cleanup(); vi.resetAllMocks() })

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  return ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe("forecast selection and refresh", () => {
  it.each([null, 0, -1, 1.5, NaN])("never requests an invalid selection %s", (routeId) => {
    const { result } = renderHook(() => useForecast(routeId, "day"), { wrapper: wrapper() })
    expect(result.current.fetchStatus).toBe("idle")
    expect(api.forecast).not.toHaveBeenCalled()
  })

  it("retains the last successful snapshot after a refresh error and recovers", async () => {
    const data = { model_version: "demo" } as Awaited<ReturnType<typeof api.forecast>>
    vi.mocked(api.forecast).mockResolvedValueOnce(data).mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(data)
    const { result } = renderHook(() => useForecast(2, "day"), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    await act(async () => { await result.current.refetch() })
    await waitFor(() => expect(result.current.isRefetchError).toBe(true))
    expect(result.current.data).toEqual(data)
    await act(async () => { await result.current.refetch() })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
  })

  it("isolates a late response from the previous route and horizon", async () => {
    let resolveOld!: (value: Awaited<ReturnType<typeof api.forecast>>) => void
    vi.mocked(api.forecast).mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve }))
    const current = { model_version: "current" } as Awaited<ReturnType<typeof api.forecast>>
    vi.mocked(api.forecast).mockResolvedValueOnce(current)
    const { result, rerender } = renderHook(({ id, horizon }: { id: number; horizon: "day" | "month" }) => useForecast(id, horizon), {
      initialProps: { id: 1, horizon: "day" }, wrapper: wrapper(),
    })
    rerender({ id: 2, horizon: "month" })
    await waitFor(() => expect(result.current.data).toEqual(current))
    await act(async () => { resolveOld({ model_version: "old" } as typeof current); await Promise.resolve() })
    expect(result.current.data).toEqual(current)
  })
})

it("does not request an incomplete window", () => {
  const { result } = renderHook(() => useForecast(1, "day", {}, false), { wrapper: wrapper() })
  expect(result.current.fetchStatus).toBe("idle")
  expect(api.forecast).not.toHaveBeenCalled()
})

it("isolates a late response from the previous stop", async () => {
  type Response = Awaited<ReturnType<typeof api.forecast>>
  let finish!: (value: Response) => void
  vi.mocked(api.forecast).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve }))
  const current = { model_version: "selected-stop" } as Response
  vi.mocked(api.forecast).mockResolvedValueOnce(current)
  const { result, rerender } = renderHook(({ stop }) => useForecast(1, "day", { stop_id: stop, start: "2026-10-01T00:00:00Z", end: "2026-10-01T01:00:00Z" }), { initialProps: { stop: 11 }, wrapper: wrapper() })
  rerender({ stop: 12 })
  await waitFor(() => expect(result.current.data).toEqual(current))
  await act(async () => { finish({ model_version: "old-stop" } as Response); await Promise.resolve() })
  expect(result.current.data).toEqual(current)
})


it("manual mode does not refetch on reconnect", async () => {
  const data = { model_version: "manual" } as Awaited<ReturnType<typeof api.forecast>>
  vi.mocked(api.forecast).mockResolvedValue(data)
  const { result } = renderHook(() => useForecast(1, "day", {}, true, false), { wrapper: wrapper() })
  await waitFor(() => expect(result.current.isSuccess).toBe(true))
  await act(async () => {
    onlineManager.setOnline(false)
    onlineManager.setOnline(true)
    await new Promise((resolve) => setTimeout(resolve, 20))
  })
  expect(api.forecast).toHaveBeenCalledTimes(1)
})

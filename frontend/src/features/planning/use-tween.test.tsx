import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useTweenedValues } from "./use-tween"

let now = 0
let queued: FrameRequestCallback[] = []
const flush = (ms: number) => act(() => { now += ms; const callbacks = queued; queued = []; callbacks.forEach((callback) => callback(now)) })

beforeEach(() => {
  now = 0; queued = []
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => { queued.push(callback); return queued.length })
  vi.stubGlobal("cancelAnimationFrame", () => { queued = [] })
  vi.spyOn(performance, "now").mockImplementation(() => now)
})
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe("tweened values", () => {
  it("passes through intermediate values and lands exactly on the target", () => {
    const initial = new Map([["a", 0]])
    const { result, rerender } = renderHook(({ target }) => useTweenedValues(target, 1000), { initialProps: { target: initial } })
    expect(result.current.get("a")).toBe(0)
    rerender({ target: new Map([["a", 100]]) })
    flush(0)
    flush(500)
    expect(result.current.get("a")).toBeCloseTo(50)
    flush(500)
    expect(result.current.get("a")).toBe(100)
  })

  it("shows new keys at their target immediately", () => {
    const { result, rerender } = renderHook(({ target }) => useTweenedValues(target, 1000), { initialProps: { target: new Map([["a", 1]]) } })
    rerender({ target: new Map([["b", 7]]) })
    flush(0)
    expect(result.current.get("b")).toBe(7)
  })
})

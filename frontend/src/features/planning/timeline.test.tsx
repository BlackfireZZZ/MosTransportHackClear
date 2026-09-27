import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { STEP_MS, Timeline } from "./timeline"

const points = ["2025-11-01T00:00:00+03:00", "2025-11-01T01:00:00+03:00", "2025-11-01T02:00:00+03:00"].map((timestamp) => ({ timestamp }))
const format = (value: string) => value.slice(11, 16)

function Harness({ initial = points[0].timestamp, onChange }: { initial?: string; onChange?: (value: string) => void }) {
  const [value, setValue] = useState(initial)
  return <Timeline points={points} value={value} format={format} onChange={(next) => { onChange?.(next); setValue(next) }} />
}

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { cleanup(); vi.useRealTimers() })

describe("timeline playback", () => {
  it("advances one interval per step and stops on the last one", () => {
    render(<Harness />)
    const slider = screen.getByRole("slider", { name: /Интервал на карте/ })
    fireEvent.click(screen.getByRole("button", { name: "Воспроизвести" }))
    expect(screen.getByRole("button", { name: "Пауза" })).toBeVisible()
    act(() => { vi.advanceTimersByTime(STEP_MS) })
    expect(slider).toHaveValue("1")
    expect(slider).toHaveAttribute("aria-valuetext", "01:00")
    act(() => { vi.advanceTimersByTime(STEP_MS) })
    expect(slider).toHaveValue("2")
    expect(screen.getByRole("button", { name: "Воспроизвести" })).toBeVisible()
    act(() => { vi.advanceTimersByTime(STEP_MS * 3) })
    expect(slider).toHaveValue("2")
  })

  it("restarts from the first interval when played at the end", () => {
    const onChange = vi.fn()
    render(<Harness initial={points[2].timestamp} onChange={onChange} />)
    fireEvent.click(screen.getByRole("button", { name: "Воспроизвести" }))
    expect(onChange).toHaveBeenLastCalledWith(points[0].timestamp)
    act(() => { vi.advanceTimersByTime(STEP_MS) })
    expect(onChange).toHaveBeenLastCalledWith(points[1].timestamp)
  })

  it("pauses on manual selection and supports the native slider", () => {
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)
    fireEvent.click(screen.getByRole("button", { name: "Воспроизвести" }))
    fireEvent.change(screen.getByRole("slider", { name: /Интервал на карте/ }), { target: { value: "2" } })
    expect(onChange).toHaveBeenLastCalledWith(points[2].timestamp)
    expect(screen.getByRole("button", { name: "Воспроизвести" })).toBeVisible()
    act(() => { vi.advanceTimersByTime(STEP_MS * 2) })
    expect(onChange).toHaveBeenCalledTimes(1)
  })

  it("disables playback for a single interval", () => {
    render(<Timeline points={points.slice(0, 1)} value={points[0].timestamp} format={format} onChange={vi.fn()} />)
    expect(screen.getByRole("button", { name: "Воспроизвести" })).toBeDisabled()
  })
})

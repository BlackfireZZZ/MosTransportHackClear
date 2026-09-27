import { useEffect, useRef, useState } from "react"

const ease = (t: number) => -(Math.cos(Math.PI * t) - 1) / 2

/** Animates each keyed value from what is currently shown to the new target; reduced motion or no rAF jumps straight to it. */
export function useTweenedValues(target: ReadonlyMap<string, number>, duration: number): ReadonlyMap<string, number> {
  const [shown, setShown] = useState(target)
  const shownRef = useRef(target)
  useEffect(() => {
    const apply = (values: ReadonlyMap<string, number>) => { shownRef.current = values; setShown(values) }
    const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches
    if (reduced || typeof requestAnimationFrame !== "function" || duration <= 0) {
      const timer = window.setTimeout(() => apply(target), 0)
      return () => window.clearTimeout(timer)
    }
    const from = shownRef.current
    const start = performance.now()
    let frame = 0
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / duration)
      const eased = ease(progress)
      const next = new Map<string, number>()
      for (const [key, value] of target) {
        const origin = from.get(key) ?? value
        next.set(key, origin + (value - origin) * eased)
      }
      apply(next)
      if (progress < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [target, duration])
  return shown
}

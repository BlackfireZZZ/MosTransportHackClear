import AxeBuilder from "@axe-core/playwright"
import { expect, test } from "@playwright/test"
import { readFile } from "node:fs/promises"
import { response } from "../src/features/planning/fixtures.test-support"
import type { PlanningRequest } from "../src/api/planning"
import { forecastResponse, routes } from "./forecast-fixtures"
import { graph } from "./network-fixtures"
import { readCsv } from "./csv-reader"

const localMapStyle = { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#eeede8" } }] }

for (const width of [390, 768, 1440]) {
  test(`map-first home, controls, export and accessibility at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 })
    await page.route("https://**/*", (route) => route.abort())
    await page.route("https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", (route) => route.fulfill({ json: localMapStyle }))
    await page.route("**/api/v1/**", async (route) => {
      if (route.request().url().endsWith("/planning/forecast")) {
        const body = route.request().postDataJSON() as PlanningRequest
        const weight = body.factors.weather.enabled ? body.factors.weather.multiplier : 0
        const ratio = (1 + weight * 1.2) / (1 + weight)
        await route.fulfill({ json: { ...response, ...body, points: response.points.map((point) => ({ ...point, scenario: point.baseline * ratio, spatial: point.spatial.map((stop) => ({ ...stop, scenario: stop.baseline * ratio })) })) } })
      } else await route.fulfill({ json: route.request().url().includes("/tram-graph/geojson") ? graph() : [] })
    })
    await page.goto("/")
    await expect(page.getByRole("heading", { name: "Посадки на трамвайной сети" })).toBeVisible()
    await expect(page.getByRole("navigation", { name: "Основная навигация" })).not.toContainText("Опубликованный прогноз")
    await expect(page.getByRole("region", { name: "Прогноз на карте" })).toBeVisible()
    await expect(page.getByTestId("tram-map").filter({ visible: true })).toHaveAttribute("data-state", "ready")
    await expect(page.getByRole("slider", { name: /Интервал на карте/ })).toBeVisible()
    await expect(page.getByText("Модель и веса", { exact: true })).toBeVisible()
    await page.getByText("Модель и веса", { exact: true }).click()
    await expect(page.getByLabel("Основа расчёта")).toHaveValue("approved")
    await expect(page.getByRole("option", { name: "Эксперимент с источниками" })).toHaveCount(0)
    const weather = page.getByRole("checkbox", { name: "Погода", exact: true })
    await expect(page.getByLabel("Вес: Погода")).toHaveCount(0)
    await expect(page.getByRole("slider", { name: "Ползунок веса: Погода" })).toHaveCount(0)
    await weather.focus()
    await page.keyboard.press("Space")
    await expect(page.getByLabel("Вес: Погода")).toBeVisible()
    await page.getByLabel("Вес: Погода").fill("1.5")
    await expect(page.locator(".planning-selection")).toContainText("112")
    await page.getByLabel("Основа расчёта").selectOption("stop_model")
    await expect(page.getByRole("checkbox", { name: "Трафик", exact: true })).toHaveCount(1)
    await page.getByText("Скачать CSV", { exact: true }).click()
    await expect(page.getByText(/маршрут 1 · все часы дня/)).toBeVisible()
    const download = page.waitForEvent("download")
    await page.getByRole("button", { name: "Скачать файл" }).click()
    const file = await download
    expect(file.suggestedFilename()).toBe("planning-1.csv")
    const [total] = readCsv(await readFile(await file.path(), "utf8")).filter((row) => row.row_kind === "total")
    expect(Number(total.scenario)).toBeCloseTo(112)
    expect(JSON.parse(total.scenario_weights).weather).toEqual({ enabled: true, multiplier: 1.5 })
    expect(JSON.parse(total.source_enabled).weather).toBe(false)
    await page.getByLabel("Основа расчёта").selectOption("approved")
    await page.getByText("Опубликованный ряд и происхождение", { exact: true }).click()
    await expect(page.getByRole("link", { name: "Открыть подробный опубликованный ряд" })).toHaveAttribute("href", "/?view=published")
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    const result = await new AxeBuilder({ page }).include(".planning").analyze()
    expect(result.violations).toEqual([])
  })
}

test("clicking a forecast marker opens its GTFS estimate even after a row without coordinates", async ({ page }) => {
  await page.route("https://**/*", (route) => route.abort())
  await page.route("https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", (route) => route.fulfill({ json: localMapStyle }))
  await page.route("**/api/v1/**", async (route) => {
    if (route.request().url().endsWith("/planning/forecast")) {
      const body = route.request().postDataJSON() as PlanningRequest
      const stop = response.points[0].spatial[0]
      await route.fulfill({ json: { ...response, ...body, points: [{ ...response.points[0], spatial: [
        { ...stop, stop_id: "gtfs:missing", name: "Без координат", latitude: null, longitude: null },
        { ...stop, stop_id: "gtfs:target", name: "Целевая", latitude: 55.60, longitude: 37.59, scenario: 57 },
      ] }] } })
    } else await route.fulfill({ json: route.request().url().includes("/tram-graph/geojson") ? graph() : [] })
  })
  await page.goto("/")
  const map = page.getByTestId("tram-map").filter({ visible: true })
  await expect(map).toHaveAttribute("data-state", "ready")
  await expect(async () => {
    await map.click()
    await expect(page.locator(".planning-selection")).toContainText("Целевая")
  }).toPass()
  await expect(page.locator(".planning-selection")).toContainText("57")
})

test("whole-network mode requests ten routes and waits for them before exporting", async ({ page }) => {
  const requested: string[] = []
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/**", async (route) => {
    if (route.request().url().endsWith("/planning/forecast")) {
      const body = route.request().postDataJSON() as PlanningRequest
      requested.push(body.route)
      await route.fulfill({ json: { ...response, ...body } })
    } else await route.fulfill({ json: route.request().url().includes("/tram-graph/geojson") ? graph() : [] })
  })
  await page.goto("/")
  await page.getByRole("button", { name: "Вся сеть" }).click()
  await expect(page.getByRole("button", { name: "Вся сеть" })).toHaveAttribute("aria-pressed", "true")
  await expect(page.locator(".planning-selection")).toContainText("Загружено 10 из 10 маршрутов")
  expect(new Set(requested)).toEqual(new Set(["1", "5", "7", "11", "12", "17", "25", "26", "28", "50"]))
  await page.getByText("Скачать CSV", { exact: true }).click()
  await expect(page.getByText(/10 маршрутов · 2025-11-01 · все часы/)).toBeVisible()
  const download = page.waitForEvent("download")
  await page.getByRole("button", { name: "Скачать файл" }).click()
  const rows = readCsv(await readFile(await (await download).path(), "utf8"))
  expect(new Set(rows.filter((row) => row.row_kind === "total").map((row) => row.route)).size).toBe(10)
})

test("published year opens planning with one selected month on the timeline and in CSV", async ({ page }) => {
  const publishedHorizons: string[] = []
  const points = Array.from({ length: 12 }, (_, index) => {
    const month = (10 + index) % 12 + 1
    const year = 2025 + Math.floor((10 + index) / 12)
    const nextMonth = month === 12 ? 1 : month + 1
    const nextYear = month === 12 ? year + 1 : year
    const baseline = 100 + index * 100
    return { ...response.points[0], timestamp: `${year}-${String(month).padStart(2, "0")}-01T00:00:00+03:00`, bucket_end: `${nextYear}-${String(nextMonth).padStart(2, "0")}-01T00:00:00+03:00`, basis: index < 2 ? "competition_period" as const : "scenario_projection" as const, baseline, scenario: baseline }
  })
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith("/planning/forecast")) {
      const body = route.request().postDataJSON() as PlanningRequest
      await route.fulfill({ json: { ...response, ...body, qualitative: true, points } })
    } else if (url.pathname.endsWith("/routes")) await route.fulfill({ json: routes.map((item) => ({ ...item, number: String(item.id) })) })
    else if (url.pathname.endsWith("/forecasts")) {
      publishedHorizons.push(url.searchParams.get("horizon") ?? "")
      await route.fulfill({ json: forecastResponse(1, "day") })
    } else await route.fulfill({ json: url.pathname.endsWith("/tram-graph/geojson") ? graph() : [] })
  })
  await page.goto("/?view=published")
  await page.getByRole("button", { name: "1 год · качественный сценарий" }).click()
  await expect(page.locator(".planning-primary-controls select").nth(1)).toHaveValue("year")
  await expect(page.getByRole("note")).toContainText("Точность на годовом горизонте не проверена")
  const slider = page.getByRole("slider", { name: /Интервал на карте/ })
  await expect(slider).toHaveAttribute("max", "11")
  await slider.fill("2")
  await expect(slider).toHaveAttribute("aria-valuetext", "январь 2026 г.")
  await page.getByText("Скачать CSV", { exact: true }).click()
  await expect(page.getByText(/Только выбранный на таймлайне месяц/)).toBeVisible()
  const download = page.waitForEvent("download")
  await page.getByRole("button", { name: "Скачать файл" }).click()
  const rows = readCsv(await readFile(await (await download).path(), "utf8"))
  expect(rows.filter((row) => row.row_kind === "total")).toHaveLength(1)
  expect(rows[0].timestamp).toBe(points[2].timestamp)
  expect(rows[0].basis).toBe("scenario_projection")
  expect(publishedHorizons).not.toContain("year")
})

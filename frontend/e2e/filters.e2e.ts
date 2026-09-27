import { expect, test } from "@playwright/test"
import { forecastResponse, routes } from "./forecast-fixtures"
import { toMoscowInput } from "../src/features/forecast/lib/selection"
import type { ForecastHorizon } from "../src/features/forecast/types"

for (const horizon of ["day", "month"] as const) {
  test(`stop and Moscow window are coherent for ${horizon}`, async ({ page }) => {
    const requests: URL[] = []
    await page.route("**/api/v1/routes", (route) => route.fulfill({ json: routes }))
    await page.route("**/api/v1/routes/*/stops", (route) => {
      const id = Number(new URL(route.request().url()).pathname.split("/").at(-2))
      return route.fulfill({ json: forecastResponse(id, "day").stops })
    })
    await page.route("**/api/v1/forecasts?*", (route) => {
      const url = new URL(route.request().url())
      requests.push(url)
      const selectedHorizon = url.searchParams.get("horizon") as ForecastHorizon
      const body = forecastResponse(Number(url.searchParams.get("route_id")), selectedHorizon)
      const start = url.searchParams.get("start")
      const end = url.searchParams.get("end")
      const selectedStop = url.searchParams.get("stop_id")
      const points = body.points.filter((point) => !start || !end || (Date.parse(point.timestamp) >= Date.parse(start) && Date.parse(point.timestamp) < Date.parse(end)))
        .map((point) => selectedStop ? { ...point, predicted_passengers: 5, lower_bound: 2, upper_bound: 8, capacity: null } : point)
      return route.fulfill({ json: { ...body, points, peak_passengers: Math.max(0, ...points.map((p) => p.predicted_passengers)), peak_load_percent: selectedStop ? null : body.peak_load_percent } })
    })
    await page.goto("/?view=published")
    await page.getByRole("tab", { name: horizon === "day" ? "1 день" : "1 месяц", exact: true }).click()
    const stop = page.getByRole("combobox", { name: "Остановка", exact: true })
    await expect(stop).toBeEnabled()
    await stop.focus()
    await page.keyboard.press("Enter")
    await page.getByRole("option", { name: "Тестовая остановка" }).press("Enter")
    await expect(page.locator("article.kpi", { hasText: "Пиковый поток" })).toContainText("5")
    const body = forecastResponse(1, horizon)
    const first = body.points[0].timestamp
    const third = body.points[2].timestamp
    await page.getByLabel("Начало · МСК").fill(toMoscowInput(first))
    await expect(page.getByRole("alert").filter({ hasText: "исправьте интервал" })).toBeVisible()
    await expect(page.getByRole("region", { name: "Ключевые показатели" })).toHaveCount(0)
    const count = requests.length
    await page.getByLabel("Конец (не включительно) · МСК").fill(toMoscowInput(third))
    await expect(page.locator("article.kpi", { hasText: "Пиковый поток" })).toContainText("5")
    expect(requests.length).toBeGreaterThan(count)
    expect(requests.at(-1)?.searchParams.get("stop_id")).toBe("11")
    expect(requests.at(-1)?.searchParams.get("start")).toBe(new Date(first).toISOString())
    expect(requests.at(-1)?.searchParams.get("end")).toBe(new Date(third).toISOString())
    await page.getByRole("button", { name: "Сбросить остановку и интервал" }).click()
    await expect(stop).toContainText("Весь маршрут")
    await expect(page.getByLabel("Начало · МСК")).toHaveValue("")
    await stop.click()
    await page.getByRole("option", { name: "Тестовая остановка" }).click()
    await page.getByRole("combobox", { name: "Маршрут", exact: true }).click()
    await page.getByRole("option", { name: /Т2/ }).click()
    await expect(stop).toContainText("Весь маршрут")
    await expect(page.locator("article.kpi", { hasText: "Пиковый поток" })).toContainText("820")
    expect(requests.at(-1)?.searchParams.has("stop_id")).toBe(false)
  })
}


test("stop catalog retry and empty filtered window retain recovery controls", async ({ page }) => {
  let failed = true
  await page.route("**/api/v1/routes", (route) => route.fulfill({ json: routes }))
  await page.route("**/api/v1/routes/*/stops", (route) => route.fulfill({ status: failed ? 503 : 200, json: failed ? {} : forecastResponse(1, "day").stops }))
  await page.route("**/api/v1/forecasts?*", (route) => {
    const body = forecastResponse(1, "day")
    const selected = new URL(route.request().url()).searchParams.has("stop_id")
    return route.fulfill({ json: selected ? { ...body, points: [], stops: [], peak_passengers: 0, peak_load_percent: null } : body })
  })
  await page.goto("/?view=published")
  await expect(page.getByRole("button", { name: "Повторить остановки" })).toBeVisible()
  await expect(page.getByRole("heading", { name: "Почасовой прогноз" })).toBeVisible()
  failed = false
  await page.getByRole("button", { name: "Повторить остановки" }).click()
  const stop = page.getByRole("combobox", { name: "Остановка", exact: true })
  await expect(stop).toBeEnabled()
  await stop.click()
  await page.getByRole("option", { name: "Тестовая остановка" }).click()
  await expect(page.getByRole("heading", { name: "Нет точек прогноза" })).toBeVisible()
  await expect(page.getByRole("region", { name: "Ключевые показатели" })).toHaveCount(0)
  await page.getByRole("button", { name: "Сбросить остановку и интервал" }).click()
  await expect(page.getByRole("heading", { name: "Почасовой прогноз" })).toBeVisible()
})

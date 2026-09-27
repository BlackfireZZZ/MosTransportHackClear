import AxeBuilder from "@axe-core/playwright"
import { expect, test, type Page, type Route } from "@playwright/test"

import type { ForecastHorizon } from "../src/features/forecast/types"

import { graph } from "./network-fixtures"
import { forecastResponse, routes } from "./forecast-fixtures"

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    body: JSON.stringify(body),
    contentType: "application/json",
    status,
  })
}

async function mockDashboardApi(
  page: Page,
  options: {
    forecastHandler?: (route: Route) => Promise<void>
  } = {},
) {
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/tram-graph/geojson", (route) => fulfillJson(route, graph()))
  await page.route("**/api/v1/routes/*/stops", (route) => {
    const routeId = Number(new URL(route.request().url()).pathname.split("/").at(-2))
    return fulfillJson(route, forecastResponse(routeId, "day").stops)
  })
  await page.route("**/api/v1/routes", (route) => fulfillJson(route, routes))
  await page.route("**/api/v1/forecasts?*", async (route) => {
    if (options.forecastHandler) {
      await options.forecastHandler(route)
      return
    }
    const url = new URL(route.request().url())
    const routeId = Number(url.searchParams.get("route_id"))
    const horizon = (url.searchParams.get("horizon") ?? "day") as ForecastHorizon
    await fulfillJson(route, forecastResponse(routeId, horizon))
  })
}

test("loads the dashboard and refreshes forecast by route and horizon", async ({ page }) => {
  await mockDashboardApi(page)
  await page.goto("/?view=published")

  await expect(page.getByRole("heading", { name: "Пассажиропоток трамвайной сети" })).toBeVisible()
  await expect(page.getByRole("heading", { name: "Почасовой прогноз" })).toBeVisible()
  await expect(page.locator("article.kpi", { hasText: "Пиковый поток" })).toContainText("600")

  const routeSelect = page.getByRole("combobox", { name: "Маршрут" })
  await routeSelect.focus()
  await page.keyboard.press("Enter")
  const secondRoute = page.getByRole("option", { name: /Т2/ })
  await expect(secondRoute).toBeVisible()
  await secondRoute.press("Enter")
  await expect(routeSelect).toContainText("Т2")
  await expect(page.locator("article.kpi", { hasText: "Пиковый поток" })).toContainText("820")

  await page.getByRole("tab", { name: "1 день" }).focus()
  await page.keyboard.press("ArrowRight")
  await expect(page.getByRole("tab", { name: "1 месяц" })).toHaveAttribute("aria-selected", "true")
  await expect(page.getByRole("heading", { name: "Прогноз по дням" })).toBeVisible()

  const accessibility = await new AxeBuilder({ page }).analyze()
  expect(accessibility.violations).toEqual([])
})

test("shows an API error and recovers through retry", async ({ page }) => {
  let attempts = 0
  await mockDashboardApi(page, {
    forecastHandler: async (route) => {
      attempts += 1
      if (attempts <= 2) {
        await fulfillJson(route, { detail: "temporary outage" }, 503)
        return
      }
      await fulfillJson(route, forecastResponse(1, "day"))
    },
  })
  await page.goto("/?view=published")

  await expect(page.getByRole("heading", { name: "Прогноз временно недоступен" })).toBeVisible()
  await page.getByRole("button", { name: "Повторить прогноз" }).click()
  await expect(page.getByRole("heading", { name: "Почасовой прогноз" })).toBeVisible()
  expect(attempts).toBeGreaterThanOrEqual(3)
})

test("retains the last forecast after a failed refresh and recovers", async ({ page }) => {
  let unavailable = false
  await mockDashboardApi(page, { forecastHandler: (route) => unavailable
    ? fulfillJson(route, { detail: "refresh outage" }, 503)
    : fulfillJson(route, forecastResponse(1, "day")) })
  await page.goto("/?view=published")
  await expect(page.getByRole("heading", { name: "Почасовой прогноз" })).toBeVisible()
  unavailable = true
  await page.getByRole("button", { name: "Обновить прогноз" }).click()
  await expect(page.getByRole("heading", { name: "Показан сохранённый прогноз" })).toBeVisible()
  await expect(page.locator("article.kpi", { hasText: "Пиковый поток" })).toContainText("600")
  await expect(page.getByRole("heading", { name: "Почасовой прогноз" })).toBeVisible()
  unavailable = false
  await page.getByRole("button", { name: "Повторить прогноз" }).click()
  await expect(page.getByRole("heading", { name: "Показан сохранённый прогноз" })).toHaveCount(0)
})

test("does not request forecasts while routes are loading or empty", async ({ page }) => {
  const requests: string[] = []
  let release!: () => void
  const pending = new Promise<void>((resolve) => { release = resolve })
  await page.route("**/api/v1/routes", async (route) => {
    await pending
    await fulfillJson(route, [])
  })
  page.on("request", (request) => {
    if (/\/forecasts\?/.test(request.url())) requests.push(request.url())
  })
  await page.goto("/?view=published")
  await expect(page.getByRole("heading", { name: "Пассажиропоток трамвайной сети" })).toBeVisible()
  expect(requests).toEqual([])
  release()
  await expect(page.getByRole("heading", { name: "Маршруты не найдены" })).toBeVisible()
  await expect(page.getByRole("button", { name: "Обновить прогноз" })).toBeDisabled()
  expect(requests).toEqual([])
})

for (const width of [390, 768, 1440]) {
  test(`saved forecast is accessible at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 })
    let offline = false
    await mockDashboardApi(page, { forecastHandler: (route) => fulfillJson(route,
      offline ? { detail: "offline" } : forecastResponse(1, "day"), offline ? 503 : 200) })
    await page.goto("/?view=published")
    await expect(page.getByRole("heading", { name: "Почасовой прогноз" })).toBeVisible()
    offline = true
    await page.getByRole("button", { name: "Обновить прогноз" }).click()
    await expect(page.getByRole("heading", { name: "Показан сохранённый прогноз" })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
    await page.screenshot({ path: testInfo.outputPath(`saved-forecast-${width}.png`), fullPage: true })
  })
}

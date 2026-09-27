import AxeBuilder from "@axe-core/playwright"
import { expect, test, type Page } from "@playwright/test"
import { mappedForecastResponse, routes } from "./forecast-fixtures"
import { graph, json } from "./network-fixtures"
import type { ForecastHorizon, ForecastResponse } from "../src/features/forecast/types"

async function setup(page: Page, horizon: ForecastHorizon, change?: (data: ForecastResponse) => ForecastResponse) {
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/routes", (route) => json(route, routes))
  await page.route("**/api/v1/routes/*/stops", (route) => json(route, mappedForecastResponse(1, horizon).stops))
  await page.route("**/api/v1/tram-graph/geojson", (route) => json(route, graph()))
  await page.route("**/api/v1/forecasts?*", (route) => {
    const url = new URL(route.request().url())
    let data = mappedForecastResponse(Number(url.searchParams.get("route_id")), (url.searchParams.get("horizon") ?? "day") as ForecastHorizon)
    const stop = Number(url.searchParams.get("stop_id"))
    if (stop) data = { ...data, selection: { ...data.selection!, stop_id: stop }, points: data.stop_points!.filter((point) => point.stop_id === stop).map((point) => ({
      timestamp: point.timestamp, predicted_passengers: point.predicted_passengers,
      lower_bound: point.lower_bound, upper_bound: point.upper_bound, capacity: point.capacity,
    })), stop_points: data.stop_points!.filter((point) => point.stop_id === stop) }
    return json(route, change ? change(data) : data)
  })
  await page.goto("/?view=published")
  if (horizon === "month") await page.getByRole("tab", { name: "1 месяц" }).click()
}

for (const horizon of ["day", "month"] as const) {
  test(`${horizon}: timestamp updates map values and current KPI, keyboard stop selection stays coherent`, async ({ page }) => {
    await setup(page, horizon)
    const data = mappedForecastResponse(1, horizon)
    const selector = page.getByRole("combobox", { name: "Интервал на графике и карте · МСК" })
    await expect(selector).toHaveValue(data.points[0].timestamp)
    await page.getByRole("button", { name: "Следующий интервал" }).press("Enter")
    await expect(selector).toHaveValue(data.points[1].timestamp)
    await expect(page.getByTestId("current-forecast-value")).toContainText(String(data.points[1].predicted_passengers))
    const table = page.getByRole("table", { name: "Прогноз остановок выбранного интервала" })
    await expect(table).toContainText(String(data.stop_points![2].predicted_passengers))
    await expect(page.getByText(/соответствие остановкам OSM не установлено/)).toBeVisible()
    await table.getByRole("button", { name: "Альфа прогноза" }).press("Enter")
    await expect(page.getByRole("combobox", { name: "Остановка", exact: true })).toContainText("Альфа прогноза")
    await expect(selector).toHaveValue(data.points[0].timestamp)
    await expect(page.getByTestId("current-forecast-value")).toContainText(String(data.stop_points![0].predicted_passengers))
    await expect(table.getByRole("button", { name: "Бета прогноза" })).toHaveCount(0)
  })
}

test("later seed-like buckets stay empty instead of reusing first stop predictions", async ({ page }) => {
  await setup(page, "day", (data) => ({ ...data, stop_points: data.stop_points?.slice(0, 2) }))
  await page.getByRole("combobox", { name: "Интервал на графике и карте · МСК" }).selectOption({ index: 1 })
  await expect(page.getByText(/Нет прогноза по остановкам для выбранного интервала/)).toBeVisible()
  await expect(page.getByRole("table", { name: "Прогноз остановок выбранного интервала" }).getByRole("button")).toHaveCount(0)
})

for (const width of [390, 1440]) {
  test(`map values are accessible with unavailable basemap at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 1000 })
    await setup(page, "day")
    await expect(page.getByRole("table", { name: "Прогноз остановок выбранного интервала" })).toBeVisible()
    await expect(page.getByText(/Карта недоступна. Используйте список/)).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
    await page.screenshot({ path: info.outputPath(`forecast-map-${width}.png`), fullPage: true })
  })
}

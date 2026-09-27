import AxeBuilder from "@axe-core/playwright"
import { expect, test } from "@playwright/test"
import { mappedForecastResponse, routes } from "./forecast-fixtures"
import { graph } from "./network-fixtures"
import type { ForecastHorizon } from "../src/features/forecast/types"

for (const horizon of ["day", "month"] as const) {
  test(`${horizon}: every plotted value is inspectable and selectable with keyboard`, async ({ page }) => {
    await page.route("https://**/*", (route) => route.abort())
    await page.route("**/api/v1/routes", (route) => route.fulfill({ json: routes }))
    await page.route("**/api/v1/routes/*/stops", (route) => route.fulfill({ json: mappedForecastResponse(1, horizon).stops }))
    await page.route("**/api/v1/tram-graph/geojson", (route) => route.fulfill({ json: graph() }))
    await page.route("**/api/v1/forecasts?*", (route) => {
      const selected = new URL(route.request().url()).searchParams.get("horizon") as ForecastHorizon
      return route.fulfill({ json: mappedForecastResponse(1, selected) })
    })
    await page.goto("/?view=published")
    await page.getByRole("tab", { name: horizon === "day" ? "1 день" : "1 месяц", exact: true }).click()
    const summary = page.locator("summary", { hasText: "Все значения прогноза" })
    await summary.focus()
    await page.keyboard.press("Enter")
    const table = page.getByRole("table", { name: "Все значения прогноза", exact: true })
    const snapshot = mappedForecastResponse(1, horizon)
    await expect(table.getByRole("row")).toHaveCount(snapshot.points.length + 1)
    for (const [index, point] of snapshot.points.entries()) {
      await expect(table.getByRole("row").nth(index + 1).getByRole("cell").first()).toHaveText(String(point.predicted_passengers))
    }
    await table.getByRole("button").nth(1).press("Enter")
    await expect(page.getByRole("combobox", { name: "Интервал на графике и карте · МСК" })).toHaveValue(snapshot.points[1].timestamp)
    await expect(table.getByRole("button").nth(1)).toHaveAttribute("aria-pressed", "true")
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
  })
}

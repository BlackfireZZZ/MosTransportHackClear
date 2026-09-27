import { expect, test, type Page, type Route } from "@playwright/test"
import { forecastResponse, mappedForecastResponse, routes } from "./forecast-fixtures"
import { graph } from "./network-fixtures"
import type { ForecastHorizon, ForecastResponse } from "../src/features/forecast/types"

function snapshot(version: string, routeId = 1, horizon: ForecastHorizon = "day"): ForecastResponse {
  const base = mappedForecastResponse(routeId, horizon)
  const delta = version === "run-B" ? 1000 : 0
  return { ...base, points: base.points.map((point) => ({ ...point, predicted_passengers: point.predicted_passengers + delta, lower_bound: point.lower_bound === null ? null : point.lower_bound + delta, upper_bound: point.upper_bound === null ? null : point.upper_bound + delta })),
    stop_points: base.stop_points?.map((point) => ({ ...point, predicted_passengers: point.predicted_passengers + delta, lower_bound: point.lower_bound === null ? null : point.lower_bound + delta, upper_bound: point.upper_bound === null ? null : point.upper_bound + delta })),
    peak_passengers: base.peak_passengers + delta, peak_load_percent: null,
    map: { ...base.map!, run_id: version }, generated_at: "2026-09-30T21:05:00Z", model_version: `model-${version}`,
    run: { run_id: version, dataset_id: "fixture-dataset", source_version: "fixture-source", feature_version: "fixture-features",
      entity_version: "fixture-entities", calendar_version: "fixture-calendar", graph_version: null,
      target: "synthetic_boardings", unit: "event_count", synthetic: true, forecast_origin: "2026-09-30T21:00:00Z",
      data_cutoff: "2026-09-29T21:00:00Z", interval_level: null, interval_method: null, identity_namespace: "serving-surrogate-integer" },
  }
}

async function setup(page: Page, handler: (route: Route) => Promise<void>) {
  await page.clock.install({ time: new Date("2026-10-01T00:00:00Z") })
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/routes", (route) => route.fulfill({ json: routes }))
  await page.route("**/api/v1/routes/*/stops", (route) => route.fulfill({ json: forecastResponse(1, "day").stops }))
  await page.route("**/api/v1/tram-graph/geojson", (route) => route.fulfill({ json: graph() }))
  await page.route("**/api/v1/forecasts?*", handler)
  await page.goto("/?view=published")
}

const panel = (page: Page) => page.getByRole("region", { name: "Происхождение и актуальность прогноза" })
const value = (page: Page, label: string) => panel(page).locator("div").filter({ has: page.locator("dt", { hasText: label }) }).last().locator("dd")

test("polling outage retains a snapshot and recovery replaces its provenance", async ({ page }) => {
  let mode: "first" | "failed" | "recovered" = "first"
  await setup(page, (route) => mode === "failed" ? route.fulfill({ status: 503, body: "unavailable" })
    : route.fulfill({ json: snapshot(mode === "first" ? "run-A" : "run-B") }))
  await expect(value(page, "Запуск прогноза")).toHaveText("run-A")
  await expect(page.getByTestId("current-forecast-value")).toContainText("480")
  await expect(page.getByRole("table", { name: "Прогноз остановок выбранного интервала" })).toContainText("192")
  await expect(value(page, "Время формирования")).toContainText("01.10.2026")
  await expect(value(page, "Граница исходных данных")).toContainText("30.09.2026")
  await expect(value(page, "Последняя успешная проверка")).toContainText("01.10.2026")
  await expect(panel(page)).toContainText("Поток оперативных наблюдений не подключён")
  mode = "failed"
  await page.clock.runFor(65_000)
  await expect(page.getByTestId("forecast-freshness")).toContainText("Обновление не удалось")
  await expect(value(page, "Запуск прогноза")).toHaveText("run-A")
  mode = "recovered"
  await page.clock.runFor(65_000)
  await expect(value(page, "Запуск прогноза")).toHaveText("run-B")
  await expect(page.getByTestId("current-forecast-value")).toContainText("1480")
  await expect(page.getByRole("table", { name: "Прогноз остановок выбранного интервала" })).toContainText("1192")
  await expect(page.getByTestId("forecast-freshness")).not.toContainText("Устаревший")
  await expect(value(page, "Граница исходных данных")).toContainText("00:00:00")
})

test("manual mode makes no polling requests and explains overdue checks", async ({ page }) => {
  let requests = 0
  await setup(page, (route) => { requests++; return route.fulfill({ json: snapshot("manual-run") }) })
  await expect(value(page, "Запуск прогноза")).toHaveText("manual-run")
  await page.getByLabel("Автопроверка").selectOption("off")
  const before = requests
  await page.clock.runFor(130_000)
  expect(requests).toBe(before)
  await expect(page.getByTestId("forecast-freshness")).toContainText("Устаревший снимок")
  await page.getByRole("button", { name: "Обновить прогноз", exact: true }).click()
  await expect(page.getByTestId("forecast-freshness")).not.toContainText("Устаревший")
  expect(requests).toBe(before + 1)
})

test("late response cannot replace the selected horizon provenance", async ({ page }) => {
  let pending: Route | undefined
  await setup(page, (route) => {
    const horizon = new URL(route.request().url()).searchParams.get("horizon") as ForecastHorizon
    if (horizon === "day") { pending = route; return Promise.resolve() }
    return route.fulfill({ json: snapshot("month-run", 1, horizon) })
  })
  await page.getByRole("tab", { name: "1 месяц", exact: true }).click()
  await expect(value(page, "Запуск прогноза")).toHaveText("month-run")
  await pending?.fulfill({ json: snapshot("late-day-run") }).catch(() => undefined)
  await expect(value(page, "Запуск прогноза")).toHaveText("month-run")
  await expect(page.getByRole("heading", { name: "Прогноз по дням" })).toBeVisible()
})

test("a delayed refresh expires the UI check without inventing a source update", async ({ page }) => {
  let calls = 0
  let delayed: Route | undefined
  await setup(page, (route) => {
    calls++
    if (calls > 1) { delayed = route; return Promise.resolve() }
    return route.fulfill({ json: snapshot("held-run") })
  })
  await expect(value(page, "Запуск прогноза")).toHaveText("held-run")
  const source = await value(page, "Граница исходных данных").innerText()
  await page.clock.runFor(140_000)
  await expect(page.getByTestId("forecast-freshness")).toContainText("Устаревший снимок")
  await expect(page.getByTestId("current-forecast-value")).toContainText("480")
  await expect(value(page, "Граница исходных данных")).toHaveText(source)
  await delayed?.fulfill({ json: snapshot("held-run") })
  await expect(page.getByTestId("forecast-freshness")).not.toContainText("Устаревший")
})

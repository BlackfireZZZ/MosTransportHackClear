import { readFile } from "node:fs/promises"
import { expect, test, type Page } from "@playwright/test"
import { mappedForecastResponse, routes } from "./forecast-fixtures"
import { graph } from "./network-fixtures"
import { readCsv } from "./csv-reader"

async function setup(page: Page, mode: () => "success" | "error" | "empty") {
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/routes", (route) => route.fulfill({ json: routes }))
  await page.route("**/api/v1/routes/*/stops", (route) => route.fulfill({ json: mappedForecastResponse(1, "day").stops }))
  await page.route("**/api/v1/tram-graph/geojson", (route) => route.fulfill({ json: graph() }))
  let requests = 0
  await page.route("**/api/v1/forecasts?*", (route) => {
    requests++
    if (mode() === "error") return route.fulfill({ status: 503, body: "unavailable" })
    let data = mappedForecastResponse(1, "day")
    const stop = Number(new URL(route.request().url()).searchParams.get("stop_id"))
    if (stop) data = { ...data, selection: { ...data.selection!, stop_id: stop, aggregation_key: "stop_bucket" },
      points: data.stop_points!.filter((row) => row.stop_id === stop).map((row) => ({ timestamp: row.timestamp, predicted_passengers: row.predicted_passengers, lower_bound: row.lower_bound, upper_bound: row.upper_bound, capacity: row.capacity })) }
    return route.fulfill({ json: mode() === "empty" ? { ...data, points: [] } : data })
  })
  await page.goto("/?view=published")
  return () => requests
}

test("download equals the selected stop table and retains stale provenance without refetch", async ({ page }) => {
  let mode: "success" | "error" = "success"
  const requests = await setup(page, () => mode)
  await page.getByRole("combobox", { name: "Остановка", exact: true }).click()
  await page.getByRole("option", { name: "Альфа прогноза" }).click()
  await page.locator("summary", { hasText: "Все значения прогноза" }).click()
  const table = page.getByRole("table", { name: "Все значения прогноза", exact: true })
  await expect(table.getByRole("row")).toHaveCount(25)
  const before = requests()
  const downloaded = page.waitForEvent("download")
  await page.getByRole("button", { name: "Скачать CSV выбранного окна" }).click()
  const file = await downloaded
  const records = readCsv(await readFile((await file.path())!, "utf8"))
  expect(requests()).toBe(before)
  expect(records).toHaveLength(24)
  expect(records[0]).toMatchObject({ stop_id: "11", aggregation_key: "stop_bucket", run_id: "fixture-run-day", stale: "false", synthetic: "true", timezone: "Europe/Moscow" })
  for (let i = 0; i < records.length; i++) await expect(table.getByRole("row").nth(i + 1).getByRole("cell").first()).toHaveText(records[i].predicted_value.replace(".", ","))
  mode = "error"
  await page.getByRole("button", { name: "Обновить прогноз", exact: true }).click()
  await expect(page.getByTestId("forecast-freshness")).toContainText("Обновление не удалось")
  const staleDownload = page.waitForEvent("download")
  await page.getByRole("button", { name: "Скачать CSV — устаревший снимок" }).click()
  const staleFile = await staleDownload
  const staleRows = readCsv(await readFile((await staleFile.path())!, "utf8"))
  expect(staleRows[0]).toMatchObject({ run_id: records[0].run_id, stale: "true", predicted_value: records[0].predicted_value })
})

for (const state of ["error", "empty"] as const) {
  test(`does not export an ${state} result`, async ({ page }) => {
    await setup(page, () => state)
    await expect(page.getByRole("heading", { name: state === "error" ? "Прогноз временно недоступен" : "Нет точек прогноза" })).toBeVisible()
    await expect(page.getByRole("button", { name: /Скачать CSV/ })).toBeDisabled()
  })
}

test("invalid partial window disables export of previously visible data", async ({ page }) => {
  await setup(page, () => "success")
  await expect(page.getByRole("button", { name: /Скачать CSV/ })).toBeEnabled()
  await page.getByLabel("Начало · МСК").fill("2026-10-01T00:00")
  await expect(page.getByRole("button", { name: /Скачать CSV/ })).toBeDisabled()
})

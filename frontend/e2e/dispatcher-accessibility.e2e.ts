import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import AxeBuilder from "@axe-core/playwright"
import { expect, test, type Locator, type Page, type TestInfo } from "@playwright/test"
import { mappedForecastResponse, routes } from "./forecast-fixtures"
import { graph } from "./network-fixtures"
import type { ForecastHorizon } from "../src/features/forecast/types"

async function setup(page: Page) {
  const state = { mode: "success" as "success" | "error" | "empty" }
  await page.clock.install({ time: new Date("2026-10-01T00:00:00Z") })
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/routes", (route) => route.fulfill({ json: routes }))
  await page.route("**/api/v1/routes/*/stops", (route) => route.fulfill({ json: mappedForecastResponse(1, "day").stops }))
  await page.route("**/api/v1/tram-graph/geojson", (route) => route.fulfill({ json: graph() }))
  await page.route("**/api/v1/forecasts?*", (route) => {
    if (state.mode === "error") return route.fulfill({ status: 503, body: "unavailable" })
    const params = new URL(route.request().url()).searchParams
    const data = mappedForecastResponse(Number(params.get("route_id")), params.get("horizon") as ForecastHorizon)
    return route.fulfill({ json: state.mode === "empty" ? { ...data, points: [], stop_points: [], stops: [], peak_passengers: 0, peak_load_percent: null } : data })
  })
  return state
}
const values = (page: Page) => page.getByRole("table", { name: "Все значения прогноза", exact: true })
const summary = (page: Page) => page.locator("summary", { hasText: "Все значения прогноза" })
async function inspect(page: Page, info: TestInfo, name: string, screenshot = true) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }))
  if (screenshot) await page.screenshot({ path: info.outputPath(`${name}.png`), fullPage: true })
}
async function tabTo(page: Page, target: Locator) {
  for (let step = 0; step < 100; step++) {
    await page.keyboard.press("Tab")
    if (await target.evaluate((element) => element === document.activeElement)) {
      await expect(target).toBeInViewport()
      expect(await target.evaluate((element) => {
        const style = getComputedStyle(element)
        return style.outlineStyle !== "none" && parseFloat(style.outlineWidth) > 0 || style.boxShadow !== "none"
      })).toBe(true)
      return
    }
  }
  throw new Error(`Tab could not reach ${await target.evaluate((element) => element.outerHTML)}`)
}
for (const width of [390, 768, 1440]) {
  test(`integrated open values and stale map at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 1000 })
    await setup(page)
    await page.goto("/?view=published")
    await summary(page).click()
    await expect(values(page)).toBeVisible()
    await expect(page.getByRole("region", { name: "Происхождение и актуальность прогноза" })).toContainText("fixture-run-day")
    await expect(page.getByText(/Карта недоступна. Используйте список/)).toBeVisible()
    await expect(page.getByText(/соответствие остановкам OSM не установлено/)).toBeVisible()
    await expect(page.locator(".chart-legend")).toContainText("Прогноз")
    await inspect(page, info, `dispatcher-${width}-success`)
    await page.getByLabel("Автопроверка").selectOption("off")
    await page.clock.runFor(130_000)
    await expect(page.getByTestId("forecast-freshness")).toContainText("Устаревший снимок")
    await expect(values(page)).toBeVisible()
    await inspect(page, info, `dispatcher-${width}-stale`)
  })
}
test("error retry, invalid filters, and empty selection remain accessible", async ({ page }, info) => {
  await page.setViewportSize({ width: 390, height: 1000 })
  const state = await setup(page)
  state.mode = "error"
  await page.goto("/?view=published")
  await expect(page.getByRole("heading", { name: "Прогноз временно недоступен" })).toBeVisible()
  await inspect(page, info, "dispatcher-error")
  state.mode = "success"
  await page.getByRole("button", { name: "Повторить прогноз", exact: true }).click()
  await expect(summary(page)).toBeVisible()
  await page.getByLabel("Начало · МСК").fill("2026-10-01T00:00")
  await expect(page.getByRole("alert").filter({ hasText: "исправьте интервал" })).toBeVisible()
  await inspect(page, info, "dispatcher-invalid-filter")
  state.mode = "empty"
  await page.getByLabel("Конец (не включительно) · МСК").fill("2026-10-01T02:00")
  await expect(page.getByRole("heading", { name: "Нет точек прогноза" })).toBeVisible()
  await inspect(page, info, "dispatcher-empty")
})
test("Tab reaches filters, table selection, and map stop without a focus shortcut", async ({ page }, info) => {
  await page.setViewportSize({ width: 768, height: 1000 })
  await setup(page)
  await page.goto("/?view=published")
  await expect(summary(page)).toBeVisible()
  await tabTo(page, page.getByRole("combobox", { name: "Маршрут", exact: true }))
  await page.keyboard.press("Enter")
  await expect(page.getByRole("listbox")).toBeVisible()
  await page.keyboard.press("Escape")
  await tabTo(page, page.getByRole("tab", { name: "1 день", exact: true }))
  await page.keyboard.press("ArrowRight")
  await expect(page.getByRole("heading", { name: "Прогноз по дням" })).toBeVisible()
  await tabTo(page, page.getByRole("combobox", { name: "Остановка", exact: true }))
  await page.keyboard.press("Enter")
  await expect(page.getByRole("listbox")).toBeVisible()
  await page.keyboard.press("Escape")
  await tabTo(page, page.getByLabel("Начало · МСК"))
  await tabTo(page, page.getByLabel("Конец (не включительно) · МСК"))
  await tabTo(page, page.getByLabel("Автопроверка"))
  await tabTo(page, page.getByRole("button", { name: "Обновить прогноз", exact: true }))
  await page.keyboard.press("Enter")
  await tabTo(page, summary(page))
  await page.keyboard.press("Enter")
  await tabTo(page, page.getByRole("region", { name: "Прокручиваемые значения прогноза" }))
  await tabTo(page, values(page).getByRole("button").nth(1))
  await page.keyboard.press("Enter")
  await expect(values(page).getByRole("button").nth(1)).toHaveAttribute("aria-pressed", "true")
  await tabTo(page, page.getByRole("table", { name: "Прогноз остановок выбранного интервала" }).getByRole("button", { name: "Альфа прогноза" }))
  await page.keyboard.press("Enter")
  await expect(page.getByRole("combobox", { name: "Остановка", exact: true })).toContainText("Альфа прогноза")
  await inspect(page, info, "dispatcher-keyboard")
})
test("native browser 200 percent zoom preserves controls and reflows", async ({ playwright, baseURL }, info) => {
  const profile = await mkdtemp(join(tmpdir(), "tramflow-zoom-"))
  const context = await playwright.chromium.launchPersistentContext(profile, {
    channel: "chromium", ...info.project.use.launchOptions, baseURL, viewport: { width: 1440, height: 1000 },
  })
  try {
    const settings = await context.newPage()
    await settings.goto("chrome://settings/appearance")
    await settings.locator("#zoomLevel").selectOption("2")
    await expect(settings.locator("#zoomLevel")).toHaveValue("2")
    const page = await context.newPage()
    await setup(page)
    await page.goto("/?view=published")
    await expect.poll(() => page.evaluate(() => window.innerWidth)).toBe(720)
    expect(await page.evaluate(() => getComputedStyle(document.documentElement).zoom)).toBe("1")
    await summary(page).click()
    await expect(page.getByRole("combobox", { name: "Маршрут", exact: true })).toBeVisible()
    await values(page).getByRole("button").nth(1).click()
    await expect(values(page).getByRole("button").nth(1)).toHaveAttribute("aria-pressed", "true")
    await expect(page.getByRole("table", { name: "Прогноз остановок выбранного интервала" })).toBeVisible()
    await inspect(page, info, "dispatcher-native-zoom-200", false)
    const session = await context.newCDPSession(page)
    // Native zoom makes Playwright clip at unscaled CSS bounds; Chrome needs the doubled capture rectangle.
    const size = await page.evaluate(() => ({ width: document.documentElement.scrollWidth * 2, height: document.documentElement.scrollHeight * 2 }))
    const capture = await session.send("Page.captureScreenshot", { captureBeyondViewport: true, fromSurface: true, clip: { x: 0, y: 0, ...size, scale: 1 } })
    await writeFile(info.outputPath("dispatcher-native-zoom-200-full.png"), Buffer.from(capture.data, "base64"))
    await session.detach()
  } finally {
    await context.close()
    await rm(profile, { recursive: true, force: true })
  }
})
test("reduced motion preserves open values and map selection", async ({ page }, info) => {
  await page.emulateMedia({ reducedMotion: "reduce" })
  await setup(page)
  await page.goto("/?view=published")
  await summary(page).click()
  expect(await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)).toBe(true)
  expect(await page.evaluate(() => [...document.querySelectorAll("body *")].every((element) => {
    const style = getComputedStyle(element)
    return style.animationName === "none" && style.transitionDuration.split(",").every((value) => parseFloat(value) === 0)
  }))).toBe(true)
  await page.getByRole("button", { name: "Следующий интервал" }).click()
  await expect(values(page).getByRole("button").nth(1)).toHaveAttribute("aria-pressed", "true")
  await inspect(page, info, "dispatcher-reduced-motion")
})

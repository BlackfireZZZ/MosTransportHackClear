import { expect, test, type Page } from "@playwright/test"

const pending = {
  schema_version: 1,
  status: "pending",
  updated_at: null,
  runs: [],
  required_cases: [
    { id: "get_day", label: "Опубликованный GET · день" },
    { id: "get_month", label: "Опубликованный GET · месяц" },
    { id: "get_day_interval", label: "Опубликованный GET · интервал дня" },
    { id: "post_approved_day", label: "POST approved · день" },
    { id: "post_approved_month", label: "POST approved · месяц" },
    { id: "post_approved_year", label: "POST approved · год, качественный сценарий" },
    { id: "post_stop_day", label: "POST stop_model · день" },
    { id: "post_stop_month", label: "POST stop_model · месяц" },
    { id: "post_stop_year", label: "POST stop_model · год, качественный сценарий" },
    { id: "post_stop_filter", label: "POST stop_model · оценочная остановка/направление" },
    { id: "post_invalid_date", label: "POST · ошибка даты вне периода" },
  ],
}

test.use({ viewport: { width: 1440, height: 900 }, timezoneId: "Europe/Moscow", locale: "ru-RU", reducedMotion: "reduce", colorScheme: "light", deviceScaleFactor: 1 })

async function openArchitecture(page: Page, metrics: unknown = pending, status = 200) {
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/routes", (route) => route.fulfill({ contentType: "application/json", body: "[]" }))
  await page.route("**/benchmark-results.json", (route) => route.fulfill({ contentType: "application/json", status, body: JSON.stringify(metrics) }))
  await page.goto("/?view=architecture")
  await expect(page.getByRole("heading", { name: "От валидации до решения диспетчера" })).toBeVisible()
  await expect(page.getByRole("status", { name: "" }).filter({ hasText: "Замер ожидается" })).toBeVisible()
  await page.evaluate(() => document.fonts.ready)
}

test("overview is deterministic and selection shows provenance", async ({ page }) => {
  await openArchitecture(page)
  await expect(page.locator(".architecture-graph-node")).toHaveCount(6)
  await expect(page.locator(".architecture-graph-lines.desktop path.is-related")).toHaveCount(1)
  await expect(page.getByRole("button", { name: /Учёт посадок/ })).toHaveAttribute("aria-pressed", "true")
  await page.getByRole("button", { name: /Маршрут × час/ }).click()
  await expect(page.locator("#architecture-details")).toContainText("Только история до даты прогноза")
  await expect(page.getByRole("button", { name: /Учёт посадок/ })).toHaveClass(/is-related/)
  await expect(page.locator(".architecture-graph-lines.desktop path.is-related")).toHaveCount(2)
  const branches = page.locator("details.architecture-branches summary")
  await branches.focus()
  await page.keyboard.press("Enter")
  await expect(page.getByRole("button", { name: /Остановки и направления/ })).toBeVisible()
  await page.getByRole("button", { name: /Остановки и направления/ }).click()
  await expect(page.locator("#architecture-details")).toContainText("не наблюдаемые метки посадок")
  await page.getByRole("button", { name: /Годовой сценарий/ }).click()
  await expect(page.locator("#architecture-details")).toContainText("Годовая точность не подтверждена")
  await page.getByRole("button", { name: /Планирование сети.*эвристика выпуска/ }).click()
  await expect(page.locator("#architecture-details")).toContainText("фактический выпуск, наполненность и эффект для затрат не измерены")
  await page.getByRole("button", { name: /Учёт посадок/ }).click()
  await branches.click()
  await page.evaluate(() => window.scrollTo(0, 0))
  await expect(page).toHaveScreenshot("architecture-overview-desktop.png", { fullPage: true, animations: "disabled" })
})

test("request level supports keyboard selection", async ({ page }) => {
  await openArchitecture(page)
  await page.getByRole("button", { name: /Путь запроса/ }).click()
  await expect(page.getByRole("heading", { name: "Путь запроса" })).toBeVisible()
  const client = page.getByRole("button", { name: /API-клиент/ })
  await client.focus()
  await page.keyboard.press("Enter")
  await expect(client).toHaveAttribute("aria-pressed", "true")
  await expect(page.locator("#architecture-details")).toContainText("src/api")
  await page.evaluate(() => window.scrollTo(0, 0))
  await expect(page).toHaveScreenshot("architecture-request-desktop.png", { fullPage: true, animations: "disabled" })
})

test("deployment and speed panel expose pending cases", async ({ page }) => {
  await openArchitecture(page)
  await page.getByRole("button", { name: /Развёртывание/ }).click()
  await expect(page.getByRole("button", { name: /PostgreSQL/ })).toBeVisible()
  await expect(page.getByText("Измерено", { exact: true })).toHaveCount(0)
  await expect(page.getByText("Ожидает замера")).toHaveCount(pending.required_cases.length)
  await page.evaluate(() => window.scrollTo(0, 0))
  await expect(page).toHaveScreenshot("architecture-speed-desktop.png", { fullPage: true, animations: "disabled" })
})

test("published benchmark shows all measured cases", async ({ page }) => {
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/routes", (route) => route.fulfill({ contentType: "application/json", body: "[]" }))
  await page.goto("/?view=architecture")
  await page.getByRole("button", { name: /Развёртывание/ }).click()
  await expect(page.getByText("Измерено", { exact: true })).toHaveCount(11)
  await expect(page.getByText("Ожидает замера")).toHaveCount(0)
  await expect(page.locator(".architecture-measurement-summary")).toContainText(/2\s?200 запросов · 0 ошибок/)
  await expect(page.locator(".architecture-cases")).toContainText("2195.4")
  await expect(page.locator(".architecture-cases")).toContainText("выше ориентира p95")
  await expect(page.locator(".architecture-source")).toContainText("2026-09-27")
  await page.evaluate(() => window.scrollTo(0, 0))
  await expect(page).toHaveScreenshot("architecture-measured-desktop.png", { fullPage: true, animations: "disabled" })
})

test("empty and error benchmark states are explicit", async ({ page }) => {
  await openArchitecture(page, { ...pending, required_cases: [] })
  await expect(page.getByText("Перечень режимов пуст")).toBeVisible()
  await page.unroute("**/benchmark-results.json")
  await page.route("**/benchmark-results.json", (route) => route.fulfill({ status: 503, body: "{}" }))
  await page.reload()
  await expect(page.getByRole("alert", { name: "" }).filter({ hasText: "Результат недоступен" })).toBeVisible()
  await page.unroute("**/benchmark-results.json")
  await page.route("**/benchmark-results.json", (route) => route.fulfill({ contentType: "application/json", body: JSON.stringify(pending) }))
  await page.getByRole("button", { name: "Повторить" }).click()
  await expect(page.locator(".architecture-metric-state").getByText("Замер ожидается", { exact: true })).toBeVisible()
})

test("tablet schematic stays within the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 900 })
  await openArchitecture(page)
  await page.getByRole("button", { name: /Экран диспетчера/ }).click()
  await expect(page.locator("#architecture-details")).toContainText("Карта остановок основана на расчётной привязке")
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true })
  test("touch selection and narrow screenshot", async ({ page }) => {
    await openArchitecture(page)
    await page.getByRole("button", { name: /Развёртывание/ }).tap()
    await page.getByRole("button", { name: /PostgreSQL/ }).tap()
    await expect(page.locator("#architecture-details")).toContainText("Хранит версии")
    await page.getByRole("button", { name: /Обзор/ }).tap()
    await expect(page.getByRole("button", { name: /Учёт посадок/ })).toBeVisible()
    await page.evaluate(() => window.scrollTo(0, 0))
    await expect(page).toHaveScreenshot("architecture-overview-mobile.png", { fullPage: true, animations: "disabled" })
    const noOverflow = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)
    expect(noOverflow).toBe(true)
  })
})

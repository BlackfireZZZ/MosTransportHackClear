import AxeBuilder from "@axe-core/playwright"
import { expect, test } from "@playwright/test"
import type { PlanningRequest } from "../src/api/planning"
import { response } from "../src/features/planning/fixtures.test-support"

const routes = ["1", "5", "7", "11", "12", "17", "25", "26", "28", "50"]

function hourlyResponse(body: PlanningRequest) {
  const route = Number(body.route)
  const points = Array.from({ length: 24 }, (_, hour) => {
    const rush = hour >= 7 && hour <= 9 ? 2.35 : hour >= 16 && hour <= 18 ? 1.75 : hour < 5 ? 0.22 : 1
    const boarding = route === 5 ? 0 : Math.round((route === 1 ? 420 : route === 7 ? 240 : 95 + route * 4) * rush)
    return { ...response.points[0], timestamp: `${body.start_date}T${String(hour).padStart(2, "0")}:00:00+03:00`,
      bucket_end: hour === 23 ? "2025-11-02T00:00:00+03:00" : `${body.start_date}T${String(hour + 1).padStart(2, "0")}:00:00+03:00`,
      baseline: boarding, scenario: boarding, route_baseline: boarding, route_scenario: boarding,
      route_unallocated_baseline: Math.round(boarding * .2), route_unallocated_scenario: Math.round(boarding * .2),
      unallocated_baseline: Math.round(boarding * .2), unallocated_scenario: Math.round(boarding * .2), spatial: [] }
  })
  return { ...response, ...body, routes, model_version: "approved-fixture.v1", points }
}

for (const width of [390, 768, 1440]) {
  test(`network decision lab at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 })
    await page.route("https://**/*", (route) => route.abort())
    await page.route("**/api/v1/**", async (route) => {
      if (route.request().url().endsWith("/planning/forecast")) {
        const body = route.request().postDataJSON() as PlanningRequest
        await route.fulfill({ json: hourlyResponse(body) })
      } else await route.fulfill({ json: [] })
    })
    await page.goto("/?view=lab")
    await expect(page.getByRole("heading", { name: "Где проверить усиление выпуска?" })).toBeVisible()
    await expect(page.getByRole("region", { name: "Рейтинг маршрутов для проверки" })).toContainText("Маршрут 1")
    await expect(page.getByRole("region", { name: "Сценарий дополнительного вагона" })).toContainText("Посадок на отправление")
    await page.getByRole("button", { name: /Маршрут 1\b/ }).click()
    await expect(page.getByRole("region", { name: "Профиль выбранного маршрута" })).toContainText("Маршрут 1: спрос по часам")
    const scenario = page.getByRole("region", { name: "Сценарий дополнительного вагона" })
    await expect(page.getByLabel("Полный оборот вагона, мин")).toHaveValue("76")
    await expect(scenario).toContainText("Время полного оборота маршрута по расписанию оценено")
    const baseline = await scenario.locator(".lab-result-strip").textContent()
    await page.getByRole("button", { name: "Добавить один вагон" }).click()
    await expect(scenario.locator(".lab-result-strip")).not.toHaveText(baseline ?? "")
    await page.getByRole("button", { name: "Убрать один вагон" }).click()
    const endHandle = page.getByRole("slider", { name: "Конец окна" })
    await endHandle.scrollIntoViewIfNeeded()
    const end = Number(await endHandle.getAttribute("aria-valuenow"))
    const target = end <= 22 ? end + 2 : end - 2
    const rail = await page.locator(".lab-time-rail").boundingBox()
    const handle = await endHandle.boundingBox()
    if (!rail || !handle) throw new Error("window timeline is not visible")
    await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2)
    await page.mouse.down()
    await page.mouse.move(rail.x + rail.width * target / 24, handle.y + handle.height / 2, { steps: 8 })
    await page.mouse.up()
    await expect(endHandle).toHaveAttribute("aria-valuenow", String(target))
    await endHandle.press("ArrowRight")
    await expect(endHandle).toHaveAttribute("aria-valuenow", String(Math.min(24, target + 1)))
    await page.getByRole("button", { name: /Маршрут 7\b/ }).click()
    await expect(page.getByRole("region", { name: "Профиль выбранного маршрута" })).toContainText("Маршрут 7: спрос по часам")
    await expect(page.getByLabel("Полный оборот вагона, мин")).toHaveValue("163")
    await page.getByRole("button", { name: /Маршрут 17\b/ }).click()
    await expect(page.getByRole("button", { name: /Маршрут 17\b/ })).toHaveClass(/is-regular is-selected/)
    await page.getByRole("button", { name: /Маршрут 1\b/ }).click()
    await expect(page.getByRole("button", { name: /Маршрут 1\b/ })).toHaveClass(/is-high is-selected/)
    await expect(page.getByRole("region", { name: "Профиль выбранного маршрута" })).toContainText("Маршрут 1: спрос по часам")
    await page.getByLabel("Схема движения").selectOption("1")
    await expect(scenario).toContainText("кольца")
    await page.getByLabel("Схема движения").selectOption("2")
    await expect(page.locator(".lab-rank-button.is-unknown")).toContainText("Маршрут 5")
    await expect(page.locator(".lab-decision")).toHaveCSS("background-color", "rgb(240, 248, 242)")
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    const axe = await new AxeBuilder({ page }).include(".decision-lab").analyze()
    expect(axe.violations).toEqual([])
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.screenshot({ path: `/tmp/network-decision-lab-${width}.png`, fullPage: true })
  })
}

test("network timeline reaches one hour and the full day with keyboard controls", async ({ page }) => {
  await page.route("**/api/v1/**", async (route) => {
    if (route.request().url().endsWith("/planning/forecast")) await route.fulfill({ json: hourlyResponse(route.request().postDataJSON() as PlanningRequest) })
    else await route.fulfill({ json: [] })
  })
  await page.goto("/?view=lab")
  const start = page.getByRole("slider", { name: "Начало окна" })
  const end = page.getByRole("slider", { name: "Конец окна" })
  await end.press("Home")
  await expect(page.locator(".lab-timeline-heading")).toContainText("1 ч")
  await start.press("Home")
  await end.press("End")
  await expect(page.locator(".lab-timeline-heading")).toContainText("00:00–24:00 · 24 ч")
  await expect(page.getByRole("region", { name: "Профиль выбранного маршрута" })).toContainText("Выбранное окно")
})

test("network decision lab explains partial API failure and invalid operational assumptions", async ({ page }) => {
  await page.route("https://**/*", (route) => route.abort())
  await page.route("**/api/v1/**", async (route) => {
    if (route.request().url().endsWith("/planning/forecast")) {
      const body = route.request().postDataJSON() as PlanningRequest
      if (body.route === "25") await route.fulfill({ status: 503, json: { detail: "unavailable" } })
      else await route.fulfill({ json: hourlyResponse(body) })
    } else await route.fulfill({ json: [] })
  })
  await page.goto("/?view=lab")
  await expect(page.getByRole("alert")).toContainText("Рейтинг неполный")
  await page.getByLabel("Интервал в направлении, мин").fill("0")
  await expect(page.getByRole("status").last()).toContainText("интервал 2–240 мин")
  await page.getByLabel("Интервал в направлении, мин").fill("12")
  await expect(page.getByRole("region", { name: "Сценарий дополнительного вагона" })).toContainText("Посадок на отправление")
  await page.getByRole("button", { name: /Маршрут 5\b/ }).click()
  await expect(page.getByRole("status")).toContainText("нет основания рассчитывать выпуск")
  await expect(page.getByRole("region", { name: "Вывод для диспетчера" })).toContainText("Не меняйте выпуск на основании нулевого прогноза")
})

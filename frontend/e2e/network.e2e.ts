import AxeBuilder from "@axe-core/playwright"
import { expect, test, type Page } from "@playwright/test"
import { graph, json, mockNetwork, stops } from "./network-fixtures"

async function openNetwork(page: Page) {
  await page.goto("/")
  await page.getByRole("button", { name: "Граф сети" }).click()
  await expect(page.getByRole("searchbox", { name: "Поиск остановки" })).toBeVisible()
}
async function selectStop(page: Page, name: string) {
  await page.getByRole("searchbox", { name: "Поиск остановки" }).fill(name)
  const result = page.getByRole("list", { name: "Результаты поиска" }).getByRole("button", { name: new RegExp(name) })
  await result.focus()
  await page.keyboard.press("Enter")
  await expect(page.getByRole("heading", { name, exact: true })).toBeVisible()
}

test("keyboard navigation preserves directed paths and component separation", async ({ page }) => {
  const requests = await mockNetwork(page)
  await openNetwork(page)
  const route = page.getByRole("combobox", { name: "Маршрут" })
  await route.press("Enter")
  await expect(page.getByText("Основная сеть", { exact: true })).toBeVisible()
  await expect(page.getByText("Северная сеть", { exact: true })).toBeVisible()
  await page.getByRole("option", { name: /^1 ·/ }).press("Enter")
  await expect.poll(() => requests.includes("tram-graph/geojson?route=1")).toBe(true)
  await page.getByRole("button", { name: "Показать всю сеть" }).click()
  await selectStop(page, "Альфа")
  await page.getByRole("button", { name: "Откуда", exact: true }).click()
  await selectStop(page, "Бета")
  await page.getByRole("button", { name: "Куда", exact: true }).click()
  await expect(page.getByTestId("tram-path-result")).toContainText("1,7 км")
  await expect(page.getByRole("list", { name: "Остановки на пути" })).toContainText("АльфаБета")
  await page.getByRole("button", { name: "Поменять местами" }).click()
  await expect(page.getByText("Пути нет", { exact: true })).toBeVisible()
  await expect(page.getByText(/встречном направлении/)).toBeVisible()
  await selectStop(page, "Северная")
  await page.getByRole("button", { name: "Куда", exact: true }).click()
  await expect(page.getByText(/Тимирязевской/)).toBeVisible()
  await page.getByRole("button", { name: "Сбросить точки маршрута" }).click()
  await expect(page.getByText("не выбрано", { exact: true })).toHaveCount(2)
  await expect(page.getByText("Пути нет", { exact: true })).toHaveCount(0)
  await page.getByRole("searchbox", { name: "Поиск остановки" }).fill("Нет такой")
  await expect(page.getByText(/Ничего не найдено/)).toBeVisible()
})

for (const width of [390, 768, 1440]) {
  test(`local network survives an offline basemap at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 })
    await mockNetwork(page)
    await openNetwork(page)
    await expect(page.getByText(/Карта недоступна/)).toBeVisible()
    await expect(page.getByTestId("tram-map")).toHaveAttribute("data-state", "unavailable")
    await selectStop(page, "Альфа")
    await expect(page.getByRole("button", { name: "Откуда", exact: true })).toBeEnabled()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
    await page.screenshot({ path: testInfo.outputPath(`network-${width}.png`), fullPage: true })
  })
}

test("retries failed statistics without refetching healthy graph and routes", async ({ page }) => {
  const failures = new Set(["tram-graph/stats"])
  const requests = await mockNetwork(page, failures)
  await openNetwork(page)
  await expect(page.getByRole("heading", { name: "Статистика сети: ошибка загрузки" })).toBeVisible()
  await selectStop(page, "Альфа")
  const healthy = requests.filter((key) => key === "tram-graph/geojson" || key === "tram-graph/routes")
  failures.clear()
  await page.getByRole("button", { name: "Повторить: статистика сети" }).click()
  await expect(page.getByRole("region", { name: "Показатели графа сети" })).toBeVisible()
  expect(requests.filter((key) => key === "tram-graph/geojson" || key === "tram-graph/routes")).toEqual(healthy)
})

test("route geometry, search, stop and path failures have independent retries", async ({ page }) => {
  const failures = new Set(["tram-graph/geojson?route=1", "tram-graph/stops?q=Альфа&limit=20", "tram-graph/stops/101", "tram-graph/path?from=101&to=102"])
  await mockNetwork(page, failures)
  await openNetwork(page)
  await page.getByRole("combobox", { name: "Маршрут" }).press("Enter")
  await page.getByRole("option", { name: /^1 ·/ }).press("Enter")
  await expect(page.getByRole("heading", { name: "Геометрия маршрута: ошибка загрузки" })).toBeVisible()
  failures.delete("tram-graph/geojson?route=1")
  await page.getByRole("button", { name: "Повторить: геометрия маршрута" }).click()
  await expect(page.getByRole("heading", { name: "Геометрия маршрута: ошибка загрузки" })).toHaveCount(0)
  await page.getByRole("searchbox", { name: "Поиск остановки" }).fill("Альфа")
  await expect(page.getByText("Поиск остановок недоступен.")).toBeVisible()
  failures.delete("tram-graph/stops?q=Альфа&limit=20")
  await page.getByRole("button", { name: "Повторить: поиск остановок" }).click()
  await page.getByRole("list", { name: "Результаты поиска" }).getByRole("button").click()
  await expect(page.getByText("Не удалось загрузить остановку.")).toBeVisible()
  failures.delete("tram-graph/stops/101")
  await page.getByRole("button", { name: "Повторить: остановка" }).click()
  await expect(page.getByRole("heading", { name: "Альфа", exact: true })).toBeVisible()
  await page.getByRole("button", { name: "Откуда", exact: true }).click()
  await selectStop(page, "Бета")
  await page.getByRole("button", { name: "Куда", exact: true }).click()
  await expect(page.getByText("Не удалось построить путь.")).toBeVisible()
  failures.clear()
  await page.locator(".path-panel").getByRole("button", { name: "Повторить", exact: true }).click()
  await expect(page.getByTestId("tram-path-result")).toContainText("1,7 км")
})

test("basemap retries recover with a local style and reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" })
  await mockNetwork(page)
  let recover = false
  await page.route("https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", async (route) => {
    if (!recover) await route.abort()
    else await route.fulfill({ json: { version: 8, sources: {}, layers: [] } })
  })
  await openNetwork(page)
  await expect(page.getByTestId("tram-map")).toHaveAttribute("data-state", "unavailable")
  await page.getByText("Остановки и участки сети", { exact: true }).click()
  await page.getByRole("button", { name: "Альфа · OSM 101" }).click()
  await expect(page.getByRole("heading", { name: "Альфа", exact: true })).toBeVisible()
  recover = true
  await page.getByRole("button", { name: "Повторить загрузку карты" }).click()
  await expect(page.getByTestId("tram-map")).toHaveAttribute("data-state", "ready")
  await expect(page.getByText(/Карта недоступна/)).toHaveCount(0)
})

test("WebGL failure leaves the local graph accessible", async ({ page }) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, ...args: Parameters<typeof original>) {
      if (String(args[0]).includes("webgl")) return null
      return original.apply(this, args)
    } as typeof original
  })
  await mockNetwork(page)
  await openNetwork(page)
  await expect(page.getByText(/Карта недоступна/)).toBeVisible()
  await page.getByText("Остановки и участки сети", { exact: true }).click()
  await expect(page.getByRole("list", { name: "Объекты локального графа" })).toContainText("OSM 101 → OSM 102")
  await page.getByRole("button", { name: "Бета · OSM 102" }).press("Enter")
  await expect(page.getByRole("heading", { name: "Бета", exact: true })).toBeVisible()
})

test("Overpass status errors do not block topology and retry independently", async ({ page }) => {
  const failures = new Set(["overpass/status"])
  const requests = await mockNetwork(page, failures)
  await openNetwork(page)
  await expect(page.getByText(/проверка недоступна/)).toBeVisible()
  await selectStop(page, "Альфа")
  const graphRequests = requests.filter((key) => key.startsWith("tram-graph/"))
  failures.clear()
  await page.getByRole("button", { name: "Повторить: Overpass" }).click()
  await expect(page.getByText(/недоступен — Offline fixture/)).toBeVisible()
  expect(requests.filter((key) => key.startsWith("tram-graph/"))).toEqual(graphRequests)
})

test("tile failures are visible even when the style itself loads", async ({ page }) => {
  await mockNetwork(page)
  await page.route("https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", (route) => route.fulfill({ json: {
    version: 8,
    sources: { base: { type: "raster", tiles: ["https://tiles.fixture.invalid/{z}/{x}/{y}.png"], tileSize: 256 } },
    layers: [{ id: "base", source: "base", type: "raster" }],
  } }))
  await openNetwork(page)
  await expect(page.getByTestId("tram-map")).toHaveAttribute("data-state", "unavailable")
  await selectStop(page, "Альфа")
  await expect(page.getByRole("button", { name: "Откуда", exact: true })).toBeEnabled()
})


for (const width of [390, 1440]) {
  test(`missing geometry is explicit and keyboard usable at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 })
    await mockNetwork(page)
    await page.route("**/api/v1/tram-graph/geojson", async (route) => {
      const data = graph()
      await json(route, { ...data, metadata: { ...data.metadata, missing_geometry_edges: 1 },
        features: data.features.map((feature) => "source" in feature.properties
          ? { ...feature, properties: { ...feature.properties, geometry_quality: "inferred" } }
          : feature) })
    })
    await page.route("**/api/v1/tram-graph/path?*", (route) => json(route, {
      found: true, stops: stops.slice(0, 2), routes: ["1"], total_length_m: 1700,
      geometry: [[37.6, 55.75], [37.62, 55.76]], geometry_quality: "inferred", missing_geometry_edges: 1,
    }))
    await openNetwork(page)
    await expect(page.getByText(/Прямые соединения не показаны как рельсы/)).toBeVisible()
    await expect(page.getByText(/Синтетическая геометрия/)).toBeVisible()
    await selectStop(page, "Альфа")
    await page.getByRole("button", { name: "Откуда", exact: true }).press("Enter")
    await selectStop(page, "Бета")
    await page.getByRole("button", { name: "Куда", exact: true }).press("Enter")
    await expect(page.getByText(/Линия пути скрыта/)).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
  })
}

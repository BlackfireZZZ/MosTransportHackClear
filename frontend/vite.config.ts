import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { dirname, join } from "node:path"

import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import type { Plugin } from "vite"
import { defineConfig } from "vitest/config"

/**
 * maplibre-gl resolves its worker with `new URL('./maplibre-gl-worker.mjs', import.meta.url)`.
 * Neither Vite path preserves that base: dep pre-bundling rewrites it under
 * node_modules/.vite/deps/, and the production build inlines the library into the main
 * chunk. The worker then 404s and maplibre fails *silently* — canvas, controls, scale bar
 * and attribution all render, nothing is ever painted, and the console stays empty.
 *
 * The two files must stay adjacent: the worker imports './maplibre-gl-shared.mjs'
 * relatively. `setWorkerUrl()` in tram-map.tsx points maplibre at this path.
 */
const MAPLIBRE_WORKER_DIR = "maplibre"
const MAPLIBRE_WORKER_FILES = ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]

function maplibreWorkerAssets(): Plugin {
  const require = createRequire(import.meta.url)
  const distDir = dirname(require.resolve("maplibre-gl/dist/maplibre-gl.mjs"))
  const read = (name: string) => readFileSync(join(distDir, name))

  return {
    name: "maplibre-worker-assets",

    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const name = MAPLIBRE_WORKER_FILES.find((file) =>
          request.url?.startsWith(`/${MAPLIBRE_WORKER_DIR}/${file}`),
        )
        if (!name) return next()
        response.setHeader("Content-Type", "text/javascript; charset=utf-8")
        response.end(read(name))
      })
    },

    // `fileName`, not `name`: `name` would hash the paths apart.
    generateBundle() {
      for (const name of MAPLIBRE_WORKER_FILES) {
        this.emitFile({
          type: "asset",
          fileName: `${MAPLIBRE_WORKER_DIR}/${name}`,
          source: read(name),
        })
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), maplibreWorkerAssets()],
  resolve: {
    alias: { "@": new URL("./src", import.meta.url).pathname },
  },
  build: {
    // maplibre-gl is one indivisible WebGL renderer; splitting it buys nothing.
    chunkSizeWarningLimit: 1600,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
})

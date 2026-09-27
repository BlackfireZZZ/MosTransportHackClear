# ADR-0002: uv workspace, one-shot migrations and Caddy edge

- Статус: принято
- Дата: 2026-09-19

## Контекст

Backend serving и ML требуют разных зависимостей, но совместимой версии Python и одного воспроизводимого lock-файла. В production нужны предсказуемый порядок старта БД/миграций/API, раздача SPA и единый `/api` origin. В dev разработчик должен поднимать только используемый контур.

## Решение

- Backend и ML — отдельные пакеты одного uv workspace с общим `uv.lock`.
- Alembic запускается one-shot Compose service после readiness PostgreSQL; API ждёт успешного завершения миграции.
- Caddy раздаёт Vite build и проксирует `/api/*` без изменения пути.
- В dev frontend работает отдельно через Vite proxy; `make dev-backend` поднимает только БД и локальный FastAPI; `make dev-ml` не запускает web stack.

## Обоснование опытом upstream

- uv описывает workspace как набор пакетов с единым lock-файлом и позволяет `sync/run --package`: https://docs.astral.sh/uv/concepts/projects/workspaces/
- uv Docker guide рекомендует отдельный dependency layer, `--no-install-workspace` и финальный `--locked`: https://docs.astral.sh/uv/guides/integration/docker/
- Alembic использует async engine через `connection.run_sync`, а не отдельный async API: https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic
- Docker Compose рекомендует healthcheck + `service_healthy`/`service_completed_successfully` для порядка старта: https://docs.docker.com/compose/how-tos/startup-order/
- Caddy документирует взаимоисключающие `handle` для `/api/*` и SPA fallback через `try_files`: https://caddyserver.com/docs/caddyfile/patterns#single-page-apps-spas
- shadcn Vite setup использует Tailwind Vite plugin, alias и локальные component files: https://ui.shadcn.com/docs/installation/vite

## Последствия

- Docker build требует root context, потому lock принадлежит workspace.
- ML-библиотеки не попадают в backend image при `--package tramflow-backend`.
- Локальный frontend не требует Caddy; production-like проверка выполняется только `make up`/CI.
- Публичный TLS завершается внешним ingress либо Caddy-конфигурацией конкретного окружения; локальный Compose сознательно слушает HTTP.

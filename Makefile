SHELL := /bin/sh

.PHONY: bootstrap setup doctor frontend-install sync-all sync-backend sync-ml \
	up down logs dev dev-db dev-backend dev-backend-no-db dev-frontend dev-ml \
	check verify-fast verify verify-full backend-check frontend-check ml-check \
	ml-eval architecture-check contract-generate contract-check reference-contract-check e2e e2e-install \
	compose-check migration migration-check migration-verify stack-verify backend-sql-test scored-upgrade-verify smoke

BENCH_ENV := COMPOSE_PROJECT_NAME=tramflow-benchmark POSTGRES_PORT=15432 BACKEND_PORT=18000 FRONTEND_PORT=18080
BENCH_COMPOSE := docker compose -f compose.yaml -f compose.benchmark.yaml

.PHONY: benchmark-config benchmark-up benchmark-preflight benchmark-run benchmark-publish benchmark-down

benchmark-config:
	$(BENCH_ENV) $(BENCH_COMPOSE) config --quiet
	.venv/bin/python scripts/benchmark_forecast.py dry-run

benchmark-up:
	$(BENCH_ENV) $(BENCH_COMPOSE) up --build -d --wait backend

benchmark-preflight:
	$(BENCH_ENV) .venv/bin/python scripts/benchmark_forecast.py preflight

benchmark-run:
	@test -n "$(OUTPUT)" || { echo "OUTPUT is required (example: docs/benchmark/results/run.json)" >&2; exit 2; }
	$(BENCH_ENV) .venv/bin/python scripts/benchmark_forecast.py run --output "$(OUTPUT)"

benchmark-publish:
	@test -n "$(INPUT)" || { echo "INPUT is required (example: docs/benchmark/results/run.json)" >&2; exit 2; }
	.venv/bin/python scripts/benchmark_forecast.py publish --input "$(INPUT)"

benchmark-down:
	$(BENCH_ENV) $(BENCH_COMPOSE) down -v --remove-orphans

PYTHON_VERSION := $(shell tr -d '[:space:]' < .python-version)
NODE_VERSION := $(shell tr -d '[:space:]' < .node-version)
POSTGRES_PORT ?= 5432
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 8080
LOCAL_DATABASE_URL ?= postgresql+asyncpg://tramflow:tramflow_local@localhost:$(POSTGRES_PORT)/tramflow

# Reproduce the dependency state from the committed lock files. Both workspace
# packages share one .venv, and a per-package `uv sync` is exact: running
# sync-backend and then sync-ml would uninstall the backend dev extras (pytest-asyncio),
# so the local checks need a single sync that keeps every member and extra.
bootstrap: sync-all frontend-install

# Backward-compatible alias used by the README and existing workflows.
setup: bootstrap

doctor:
	@PYTHON_VERSION=$(PYTHON_VERSION) NODE_VERSION=$(NODE_VERSION) ./scripts/doctor.sh

sync-all:
	uv sync --all-packages --all-extras --locked

sync-backend:
	uv sync --package tramflow-backend --extra dev --locked

sync-ml:
	uv sync --package tramflow-ml --extra dev --locked

frontend-install:
	cd frontend && npm ci

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

# Fast, deterministic checks that do not require a running database.
verify-fast: architecture-check backend-check ml-check ml-eval frontend-check contract-check reference-contract-check compose-check

# Default proof gate: fast checks plus migrations on a disposable clean database.
verify: verify-fast migration-verify backend-sql-test

check: verify-fast

# Full proof gate additionally builds and exercises the production-like stack.
verify-full: verify stack-verify e2e

architecture-check:
	uv run --no-project python scripts/architecture_check.py

backend-check:
	uv run --package tramflow-backend ruff check backend
	uv run --package tramflow-backend mypy backend/app
	uv run --package tramflow-backend pytest backend/tests

frontend-check:
	cd frontend && npm run lint && npm run test -- --run && npm run build

ml-check:
	uv run --package tramflow-ml ruff check ml
	uv run --package tramflow-ml mypy --config-file ml/mypy-ci.ini ml/src
	uv run --package tramflow-ml pytest ml/tests

ml-eval:
	uv run --package tramflow-ml tramflow-ml evaluate

contract-generate:
	uv run --package tramflow-backend python scripts/export_openapi.py
	cd frontend && npm run api:generate

reference-contract-check:
	uv run --package tramflow-backend ruff check contracts --config backend/pyproject.toml
	uv run --package tramflow-backend mypy contracts/forecast_v1.py contracts/calendar_v1.py contracts/data_v1.py --config-file backend/pyproject.toml
	uv run --package tramflow-backend pytest contracts/tests

contract-check:
	cd frontend && npm run contract:check

e2e-install:
	cd frontend && npm run test:e2e:install

e2e:
	cd frontend && npm run test:e2e

compose-check:
	docker compose config --quiet

dev: dev-db
	$(MAKE) -j2 dev-backend-no-db dev-frontend

dev-db:
	POSTGRES_PORT=$(POSTGRES_PORT) docker compose up -d db

dev-backend: dev-db
	$(MAKE) dev-backend-no-db

dev-backend-no-db:
	DATABASE_URL=$(LOCAL_DATABASE_URL) uv run --package tramflow-backend alembic -c backend/alembic.ini upgrade head
	DATABASE_URL=$(LOCAL_DATABASE_URL) uv run --package tramflow-backend uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $(BACKEND_PORT) --reload

dev-frontend:
	cd frontend && npm run dev

dev-ml:
	uv run --package tramflow-ml tramflow-ml

smoke:
	BASE_URL=$${BASE_URL:-http://localhost:$(BACKEND_PORT)} \
	FRONTEND_URL=$${FRONTEND_URL:-http://localhost:$(FRONTEND_PORT)} \
	./scripts/smoke.sh

migration:
	DATABASE_URL=$(LOCAL_DATABASE_URL) uv run --package tramflow-backend alembic -c backend/alembic.ini revision --autogenerate -m "$(NAME)"

migration-check:
	DATABASE_URL=$(LOCAL_DATABASE_URL) uv run --package tramflow-backend alembic -c backend/alembic.ini check

backend-sql-test:
	uv run --no-project python scripts/backend-test-env.py sql

scored-upgrade-verify:
	uv run --no-project python scripts/backend-test-env.py scored-upgrade

# Runs upgrade and autogenerate drift detection against a newly-created volume.
migration-verify:
	uv run --no-project python scripts/backend-test-env.py migration

# Uses an isolated Compose project so it cannot overwrite the developer's stack.
stack-verify:
	uv run --no-project python scripts/backend-test-env.py stack

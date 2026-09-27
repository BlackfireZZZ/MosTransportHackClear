# Проверка публикации — 26 сентября 2026

База: `47b8298323ffeb4bb821ff43b823ae543c94b4de`.
Рабочее дерево: `/Users/cute/MosTransport2026Hack-worktrees/boarding-research-publish`,
ветка `agent/boarding-research-publish`. Интеграция — fast-forward в
`codex/current-state-2026-09-26`, публикация обычным push без force.
Рабочее дерево оставлено для повторения проверок; автоматическая очистка не выполнялась.

## Наблюдаемые результаты

- `python3 docs/analysis/2026-09-26-boarding-matching/artifacts/build-anchor-matching.py`: exit 0; 15 конфигураций, слабые пары при ±30 с: 3/4/5, при ±60 с: 7/7/9 для маршрутов 17/12/11.
- `python3 docs/analysis/2026-09-26-boarding-matching/artifacts/verify-small-stops.py`: exit 0; 108 вариантов, проверены исходные агрегаты, ошибки, порядок, непересечение поддержек, статистики устройств, инвариантность сдвига и хеш.
- Побайтовое сравнение всех 27 архивных файлов с исходными артефактами: различий нет.
- `make check`: exit 0; backend 350 passed / 48 skipped, ML 625 passed, frontend 92 passed, contracts 119 passed. Всего 1186 passed.
- Architecture, Ruff, Mypy, ML golden evaluation, frontend build, OpenAPI и TypeScript API snapshot, `docker compose config --quiet`: пройдены.

Для проверки использован существующий runtime без установки зависимостей:

```sh
PATH=/tmp/tramflow-tools/bin:/tmp/tramflow-node/node_modules/.bin:/Applications/Docker.app/Contents/Resources/bin:$PATH \
UV_PROJECT_ENVIRONMENT=/Users/cute/MosTransport2026Hack-worktrees/boarding-real-alignment/.venv \
make check
```

48 SQL-тестов пропущены без тестовой БД. Это ограничение проверки, а не успешный SQL smoke test. Данные и схема БД этой публикацией не меняются.

## Границы результата

Повтор сопоставления проверяет воспроизводимость и внутренние ограничения, но не точность определения реальных остановок: независимой разметки положения вагона нет. Полные исходные прогоны требуют локальных приватных входов. HTML — сохранённые снимки; самостоятельно открываемые версии используют D3 7.9.0 через CDN. Новые зависимости не добавлены.

В коммит включены код аудита, агрегаты, графики и документация. Сырой `dataset (1).zip`, `.DS_Store`, локальные окружения и `node_modules` исключены.

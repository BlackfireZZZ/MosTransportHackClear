# Изолированный замер архитектуры

Статус: **прогон выполнен 27.09.2026** на историческом снимке `937cca5b6d0fd2564f047cae5bce7e0c27e903d9`. История этого коммита не перенесена в новый репозиторий; идентификатор сохранён только как метаданные замера. [Сырой результат](results/2026-09-27-main-937cca5.json) имеет SHA-256 `9cf3ff520172599e2d53d7841ef138e983a4377399a3854eccf1df18c7de500b`; тот же SHA указан в [публичном JSON страницы архитектуры](../../frontend/public/benchmark-results.json). [Список контрольных сумм](../evidence/repository-artifacts.sha256) проверяет включённые в эту копию файлы, а команды ниже позволяют получить новый замер. Исторические несопоставимые числа остаются в [анализе критериев](../analysis/2026-09-27-forecast-criteria/README.md#проверки-и-производительность).

## Стенд и область вывода

`compose.benchmark.yaml` с `compose.yaml` создаёт проект `tramflow-benchmark`, отдельный том PostgreSQL и порты 15432/18000/18080. `benchmark-up` запускает backend и его зависимости; frontend-порт зарезервирован для отдельной UI-проверки, но frontend не входит в HTTP-замер. Backend ограничен **2 vCPU, 4 GiB RAM, swap=0**. Это лимит backend, а не PostgreSQL, frontend, Docker VM или всего хоста. Uvicorn в текущем Dockerfile запускает один worker; процедура записывает фактическое число процессов. Горизонтальное масштабирование допускает несколько независимых backend-реплик при общей PostgreSQL и одинаковой опубликованной версии; этот стенд не измеряет реплики, балансировщик, холодный старт и устойчивость в течение часов.

`docs/benchmark/cases.json` фиксирует 11 сценариев: опубликованный GET day/month, GET с интервалом, POST approved и stop_model для day/month/year, фильтр остановки/направления и ошибку даты. `preflight` делает по одному функциональному запросу каждого вида и проверяет статус, горизонт, версию, непустой ряд, режим, пометку качественного года, 12 месяцев года и текст ошибки. Эти проверки не являются нагрузкой и не публикуют скоростных чисел. GET year намеренно отсутствует: опубликованный ряд покрывает день/месяц, год — качественный POST-сценарий.

## Команды

Запускать из корня этого worktree при свободных 15432/18000/18080. Python-зависимости устанавливаются существующим `make bootstrap`; дополнительные пакеты не нужны.

```bash
make benchmark-config
make benchmark-up
make benchmark-preflight
make benchmark-run OUTPUT=docs/benchmark/results/YYYY-MM-DD-run.json
make benchmark-publish INPUT=docs/benchmark/results/YYYY-MM-DD-run.json
make benchmark-down
```

`benchmark-config` проверяет Compose и fixture без HTTP. `benchmark-up` создаёт изолированный стек. `benchmark-preflight` сверяет реальные Docker HostConfig и cgroup `cpu.max`, `memory.max`, `memory.swap.max` с профилем и отказывает при несовпадении. Он фиксирует SHA образов, Git HEAD, SHA нормализованной Compose-конфигурации, fixture, manifest прогноза и approved-конфигурации, архитектуру/CPU хоста, версию PostgreSQL, worker count и отдельные лимиты PostgreSQL. `benchmark-run` повторяет preflight перед нагрузкой и требует чистого закоммиченного worktree; при несовпадении лимитов запись замера не создаётся. Процедура не пишет payload, секреты и пассажирские идентификаторы в результат.

Метрики: успешные RPS, p50/p95/p99 полного HTTP и JSON-проверки (очередь клиентского semaphore не входит), 5 прогревочных и 200 измеряемых запросов на сценарий при concurrency 4. CPU/RAM — максимум дискретных выборок `docker stats`, не непрерывный пик; для коротких GET-сценариев получена всего одна выборка, так что она не характеризует фактический пик. Не сравнивать RPS между стендами с разными CPU-лимитами или PostgreSQL-условиями. Узкое место считать установленным только после отдельного измерения по этапам.

Страница «Как работает Тормоза» читает версионированный `frontend/public/benchmark-results.json` со статусом `measured`. `benchmark-publish` принимает только успешный сырой результат всех 11 фиксированных случаев, проверяет SHA fixture и переносит `runs[]`, условия и SHA сырого файла в публичный JSON. `runs[]` содержит `case_id`, `requests`, `warmup`, `concurrency`, `rps`, `errors`, `latency_ms.p50/p95/p99`, `backend_cpu_percent_sample_max`, `backend_memory_bytes_sample_max`, `conditions`.

## Результат 27.09.2026

Замер шёл с 19:32:18 до 19:34:56 UTC. `make benchmark-config`, `make benchmark-up`, `make benchmark-preflight`, `make benchmark-run OUTPUT=docs/benchmark/results/2026-09-27-main-937cca5.json` и `make benchmark-publish INPUT=docs/benchmark/results/2026-09-27-main-937cca5.json` завершились с кодом 0. Preflight проверил Docker HostConfig, cgroup и семантику всех 11 ответов. В raw JSON статус `measured`: 2 200 измеряемых запросов, 0 ошибок, 2 000 ответов HTTP 200 и 200 ожидаемых HTTP 422.

| Условие | Значение |
|---|---|
| Хост | AMD Ryzen 7 5800X, Linux 7.0.0-34-generic x86_64 |
| Backend | 1 Uvicorn worker; cgroup `cpu.max=200000 100000`, `memory.max=4294967296`, `memory.swap.max=0` |
| PostgreSQL | 17.11, отдельный контейнер без заданных CPU/RAM-лимитов |
| Образы backend / PostgreSQL | `sha256:52a477bec546a7b74533d196c68c60c4f7bbfdba0c55ee29e2782ff3da01426b` / `sha256:b0f9560a2de083e2cc7382e75f808c7381a32852a7ec49117deedb300e552b24` |
| SHA-256 Compose / fixture | `7e81d33fa8d8649f1140a61f15187aa369ada0edfc1bfa46e62ce16d9d65d63f` / `b44a90ef386c8b3a13ac2c6bd3ea82db774318b0bff9d109e1a40f7d868a8c66` |
| SHA-256 manifest / approved config | `a0a7144aba27918d5d413e5d4eb0c81410b565508a5401af06e1e68d9640d430` / `fdc248598b715860663e2ec708ad9224c2c8a91f4d9e013bafc02cbc9771bba8` |
| Нагрузка | Каждый сценарий: 5 прогревочных, 200 измеряемых запросов, concurrency 4 |

| Сценарий | RPS | p50 / p95 / p99, мс | CPU % выборка max | RAM MiB выборка max | Статус × 200 |
|---|---:|---:|---:|---:|---|
| GET day | 251,89 | 14,67 / 17,73 / 75,05 | 78,51 | 144,4 | 200 |
| GET month | 271,01 | 14,45 / 17,56 / 19,38 | 0,06 | 143,6 | 200 |
| GET day interval | 308,22 | 12,25 / 15,16 / 30,68 | 0,07 | 143,1 | 200 |
| POST approved day | 150,46 | 24,46 / 39,44 / 67,19 | 43,83 | 147,6 | 200 |
| POST approved month | 44,64 | 87,42 / 128,72 / 155,01 | 119,62 | 149,9 | 200 |
| POST approved year | 5,47 | 713,42 / 1 167,85 / 1 414,50 | 121,14 | 162,2 | 200 |
| POST stop_model day | 18,69 | 211,42 / 304,26 / 364,51 | 120,25 | 150,2 | 200 |
| POST stop_model month | 13,08 | 299,62 / 440,57 / 509,96 | 119,74 | 164,4 | 200 |
| POST stop_model year | 2,99 | 1 295,74 / 2 195,39 / 2 465,87 | 120,64 | 163,6 | 200 |
| POST stop_model filter | 20,22 | 194,25 / 293,72 / 315,41 | 119,55 | 152,0 | 200 |
| POST invalid date | 571,89 | 5,65 / 8,74 / 47,43 | 0,06 | 150,5 | 422 |

Основной путь опубликованного маршрутного прогноза (GET day/month/interval) на этом стенде достиг числовых ориентиров RPS и p95 из раздела «Нефункциональные требования» PDF: сотни успешных RPS при p95 существенно ниже 200–300 мс, backend 2 vCPU / 4 GiB без swap. Короткие выборки CPU не подтверждают устойчивый запас 60–80%. `approved` POST day/month также ниже ориентира p95, но даёт 150/45 RPS. Для критерия 3 «Архитектура и производительность» PDF предусматривает оценку по заявленным командой latency/RPS в README и подтверждённому честному замеру, без отдельного нагрузочного теста жюри; числовой профиль не указан как порог для каждого API-метода.

Отдельные `stop_model` day/month/year и `approved` year превышают ориентир по p95. Высокая длительность сопровождается выборками CPU около 100–121%, однако этап расчёта, сериализации или передачи, который ограничивает скорость, отдельно не измерялся. PostgreSQL вне backend-лимитов, поэтому 2 vCPU/4 GiB нельзя называть лимитом всей системы. Длительная нагрузка, несколько реплик, балансировщик и холодный старт не проверялись. Изменение кода, образов, fixture или опубликованного прогноза требует нового замера.

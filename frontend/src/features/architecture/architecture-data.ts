export type Level = "overview" | "request" | "deployment"
export type Kind = "observed" | "inferred" | "experimental" | "qualitative" | "service"
export type Node = {
  id: string
  title: string
  subtitle: string
  kind: Kind
  purpose: string
  input: string
  output: string
  provenance: string
  limit: string
  path: string
}

export const kindLabels: Record<Kind, string> = {
  observed: "Наблюдаемые данные",
  inferred: "Оценка",
  experimental: "Эксперимент",
  qualitative: "Качественный сценарий",
  service: "Сервис",
}

export const levels: Record<Level, { label: string; eyebrow: string; description: string; nodes: Node[]; branches?: Node[] }> = {
  overview: {
    label: "Обзор",
    eyebrow: "01 / ОТ ИСТОЧНИКА ДО ЭКРАНА",
    description: "Основная ветка хранит маршрутно-часовые посадки. Оценка остановок и внешние факторы идут отдельными путями и не меняют конкурсный CSV.",
    nodes: [
      { id: "raw", title: "Учёт посадок", subtitle: "валидации организатора", kind: "observed", purpose: "Сверка успешных валидаций с целевой маршрутно-часовой сеткой.", input: "Исходные записи и справочники организатора", output: "Проверенные маршрут, дата и час", provenance: "Организаторский dataset.zip; сырые записи не публикуются в Git", limit: "Валидация не содержит наблюдаемой остановки посадки и направления", path: "ml/src/tramflow_ml/competition_reconcile.py" },
      { id: "features", title: "Маршрут × час", subtitle: "сверка и признаки", kind: "observed", purpose: "Нормализация времени и построение истории без утечки из будущего.", input: "Сверенные маршрутно-часовые значения", output: "Признаки для 10 маршрутов и 61 конкурсного дня", provenance: "Только история до даты прогноза; производственный календарь", limit: "Остановки GTFS не являются метками посадок", path: "ml/src/tramflow_ml/competition_profile.py" },
      { id: "model", title: "Прогноз ML", subtitle: "обучение и batch", kind: "service", purpose: "Отдельное обучение и пакетный вывод 14 640 маршрутно-часовых прогнозов.", input: "Хронологические признаки", output: "Конкурсный CSV и манифест SHA-256", provenance: "Воспроизводимый код и проверенные артефакты", limit: "Годовая точность не подтверждена", path: "ml/src/tramflow_ml/competition_hybrid.py" },
      { id: "publish", title: "Версия в БД", subtitle: "проверка и публикация", kind: "service", purpose: "Проверка целостности bundle и атомарное переключение активной версии.", input: "CSV, манифест и контрольные суммы", output: "Активный ряд в PostgreSQL", provenance: "ml/competition_submissions/current.json и approved.json", limit: "Публикуется только утверждённый маршрутный ряд", path: "backend/app/infrastructure/forecast_publication.py" },
      { id: "api", title: "API прогноза", subtitle: "GET ряд · POST сценарий", kind: "service", purpose: "Ограниченная HTTP-выдача, агрегация и явные ошибки.", input: "Опубликованный ряд или отдельный артефакт планирования", output: "Версионированный JSON", provenance: "PostgreSQL для GET; файл артефакта для POST", limit: "GET опубликованного прогноза поддерживает день и месяц", path: "backend/app/api/routes/forecast.py" },
      { id: "ui", title: "Экран диспетчера", subtitle: "главная · карта · CSV", kind: "service", purpose: "Главная показывает конкурсный снимок и сценарий; подробный опубликованный ряд доступен из раскрываемого блока.", input: "JSON через src/api и feature hooks", output: "График, карта оценочных остановок, CSV и происхождение", provenance: "Версия и происхождение данных приходят из API", limit: "Карта остановок основана на расчётной привязке; GET может иметь иную активную версию БД", path: "frontend/src/App.tsx" },
    ],
    branches: [
      { id: "stop-estimate", title: "Остановки и направления", subtitle: "отдельный POST · GTFS + расписание", kind: "inferred", purpose: "Расчётное распределение посадок по каталоговым остановкам и направлениям.", input: "Маршрутный прогноз и GTFS-расписание", output: "Оценки по остановкам с нераспределённой частью", provenance: "GTFS плюс алгоритм привязки; не наблюдаемые метки посадок", limit: "Не участвует в построении конкурсного маршрутно-часового CSV", path: "backend/app/application/services/stop_planning.py" },
      { id: "external", title: "Внешние факторы", subtitle: "обученные ветки · веса", kind: "experimental", purpose: "Смешивание отдельно обученных веток с конкурсной основой по нормированным весам.", input: "Архив выпущенной погоды, календарь, сводки Дептранса и выбранные веса", output: "Опциональный approved POST; прежний external_experiment отдельно", provenance: "Версионированные артефакты с SHA-256 и доли в ответе", limit: "Конкурсный CSV и GET прежние; сценарный вариант не отправлен на лидерборд", path: "backend/app/application/services/planning.py" },
      { id: "network-map", title: "Вся сеть на карте", subtitle: "10 POST day · OSM схема", kind: "inferred", purpose: "Показать оценки посадок маршрутов датасета на фоне всей трамвайной сети.", input: "Десять отдельных ответов планирования и OSM GeoJSON", output: "GTFS-оценки по маршрутам и отдельные линии/остановки OSM", provenance: "Planning артефакт и независимый OSM снимок", limit: "Идентификаторы GTFS и OSM не соединяются; неполная выгрузка блокируется", path: "frontend/src/features/planning/use-network-planning.ts" },
      { id: "year", title: "Годовой сценарий", subtitle: "12 месяцев · qualitative=true", kind: "qualitative", purpose: "Качественное продление профиля для планирования по месяцам.", input: "Маршрут, начальная дата и planning-артефакт", output: "12 месячных точек POST", provenance: "Сценарный расчёт, не опубликованный ряд PostgreSQL", limit: "Годовая точность не подтверждена хронологической оценкой", path: "backend/app/application/services/planning.py" },
      { id: "decision-lab", title: "Планирование сети", subtitle: "approved POST · GTFS · эвристика выпуска", kind: "inferred", purpose: "Ранжирование окон 1–24 часа по посадкам на плановое отправление и условный расчёт добавленного вагона.", input: "Маршрутно-часовой approved POST, локальный снимок GTFS и ручные корректировки", output: "Рейтинг, сценарный график и CSV исходного прогноза", provenance: "Утверждённый прогноз + GTFS, захваченный 30.06.2026; оборот по предполагаемому коду выхода", limit: "block_id пуст; фактический выпуск, наполненность и эффект для затрат не измерены", path: "frontend/src/features/decision-lab/service-estimates.ts" },
    ],
  },
  request: {
    label: "Путь запроса",
    eyebrow: "02 / ОДИН ВЫБОР, ДВА КОНТРАКТА",
    description: "Опубликованный день/месяц читается из PostgreSQL. Остановки и год рассчитываются через отдельный POST и явно обозначены как оценка или сценарий.",
    nodes: [
      { id: "select", title: "Выбор параметров", subtitle: "маршрут · горизонт · веса", kind: "service", purpose: "Пользователь задаёт фильтр и веса источников на главной.", input: "Маршрут, дата, горизонт, остановка и источники", output: "Параметры GET или тело POST", provenance: "Выбор пользователя", limit: "Год ведёт к качественному сценарию, не к опубликованному GET", path: "frontend/src/App.tsx" },
      { id: "client", title: "API-клиент", subtitle: "src/api → hooks", kind: "service", purpose: "Передаёт запрос по контракту и различает loading, error и пустой результат.", input: "Проверенный выбор", output: "HTTP GET /forecasts или POST /planning/forecast", provenance: "Типы OpenAPI и planning-контракт", limit: "Прямых fetch в feature-компонентах нет", path: "frontend/src/api/client.ts" },
      { id: "endpoint", title: "HTTP endpoint", subtitle: "FastAPI /api/v1", kind: "service", purpose: "Валидирует окно, режим и маршрут; вызывает application-сервис.", input: "HTTP-параметры", output: "Ответ или объяснимая ошибка", provenance: "Контракты схем backend/app/schemas", limit: "GET year не является опубликованным прогнозом", path: "backend/app/api/routes/planning.py" },
      { id: "read", title: "Чтение / расчёт", subtitle: "БД или planning artifact", kind: "service", purpose: "GET читает активную версию; POST строит маршрутный или остановочный сценарий и смешивает включённые ветки по весам.", input: "Период, режим, веса; при выборе источника — артефакт веток", output: "Временные точки, provenance и ограничения", provenance: "Активная версия БД либо checksum-bound артефакт", limit: "Остановки/направления — оценки; сценарные ветки не опубликованы на лидерборде", path: "backend/app/application/services/planning.py" },
      { id: "response", title: "Ответ JSON", subtitle: "версия · время · точки", kind: "service", purpose: "Возвращает числовые точки и метаданные для выбранного режима.", input: "Результат сервиса", output: "JSON с model_version и generated_at", provenance: "Сервис прогнозирования", limit: "Годовой сценарий помечен qualitative", path: "backend/app/schemas/planning.py" },
      { id: "render", title: "Показ и CSV", subtitle: "график · карта · экспорт", kind: "service", purpose: "Показывает динамику и расчётное распределение; режим всей сети совмещает десять ответов с OSM схемой.", input: "Версионированный JSON", output: "График, карта, таблица и скачиваемый CSV", provenance: "Тот же ответ API и выбранные веса, без повторного запроса при экспорте", limit: "GTFS-точки не являются наблюдаемыми местами посадки и не соединены с OSM ID", path: "frontend/src/features/planning/planning-view.tsx" },
    ],
  },
  deployment: {
    label: "Развёртывание",
    eyebrow: "03 / КОНТЕЙНЕРЫ И ВНЕШНИЕ ДАННЫЕ",
    description: "Обучение и публикация выполняются отдельно от веб-запроса. Внешние источники не входят в обязательный маршрутно-часовой путь.",
    nodes: [
      { id: "sources", title: "Архив организатора", subtitle: "dataset.zip · валидации", kind: "observed", purpose: "Организаторские валидации и справочники поступают в пакетную подготовку.", input: "Архив организатора", output: "Файлы для пакетной подготовки", provenance: "Организатор; GTFS и OSM используются в отдельных сценариях", limit: "GTFS/OSM не наблюдают факт остановки посадки", path: "docs/product/DATASET.md" },
      { id: "batch", title: "Пакетный ML", subtitle: "вне web worker", kind: "service", purpose: "Обучает, выводит и проверяет прогноз до публикации.", input: "Исторический архив", output: "CSV, манифест и planning artifact", provenance: "ML pipeline и контрольные суммы", limit: "Не выполняется в HTTP-процессе", path: "ml/competition_submissions/2026-09-27/README.md" },
      { id: "publisher", title: "Публикация", subtitle: "forecast-publish", kind: "service", purpose: "Проверяет утверждённый bundle и загружает активную версию транзакционно.", input: "Batch-артефакт", output: "Строки прогноза в PostgreSQL", provenance: "Манифест + SHA-256", limit: "Повторная публикация требует явной версии", path: "backend/app/infrastructure/forecast_publication.py" },
      { id: "db", title: "PostgreSQL", subtitle: "активный маршрутный ряд", kind: "service", purpose: "Хранит версии опубликованных прогнозов и точки для ограниченных GET-запросов.", input: "Проверенная публикация", output: "Выборка по маршруту и времени", provenance: "forecast-publish", limit: "Состояние БД общее для реплик backend", path: "backend/app/infrastructure/repositories/forecast.py" },
      { id: "backend", title: "FastAPI", subtitle: "backend container", kind: "service", purpose: "Обслуживает GET, POST и граф сети, читает БД и неизменяемые артефакты.", input: "HTTP, PostgreSQL, planning files", output: "JSON API", provenance: "Версия образа и коммита", limit: "Количество workers и лимиты должны фиксироваться в замере", path: "backend/app/main.py" },
      { id: "frontend", title: "React + Caddy", subtitle: "frontend container", kind: "service", purpose: "Раздаёт интерфейс и проксирует /api/v1 к backend.", input: "JSON API", output: "Диспетчерская, карта, график и CSV", provenance: "Собранный frontend image", limit: "Внешние тайлы карты могут быть недоступны", path: "frontend/Caddyfile" },
    ],
  },
}

export const repoLink = (path: string) => `https://github.com/BlackfireZZZ/MosTransportHackClear/blob/main/${path}`

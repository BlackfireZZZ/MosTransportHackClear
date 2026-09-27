# Концепция решения: прогноз потоков на транспортном графе

Authoritative product requirements: [TASK.md](TASK.md). The scored outcome is
[14,640 route-date-hour boarding predictions](DATASET.md#scored-submission) for
November–December 2025. Day/hour and month are the hackathon focus; year is
optional. The graph, OD, multimodal modeling and what-if below are team hypotheses.

## Direction and observability

The organizer answered in Telegram `message2081` (2026-09-26) that the supplied
data cannot reveal which direction a tram travelled or where it came from. The
raw validation CSV has route and vehicle/run fields, but no observed stop or
direction; `place_id` means depot/site, not stop. The reference workbook contains
trip patterns and `direction_id`, yet joining a pattern to an individual 2025
validation would be an **inference** requiring compatible identity, date and
schedule evidence. Some inspected duty rows are dated 2026. Reference geometry
alone cannot turn a route-hour boarding label into a stop-hour or direction-hour
ground truth.

The repository's canonical `(route_id, direction_id, stop_id)` contracts and
[ADR-0006](../decisions/0006-stop-identity-and-direction.md) are useful for
**synthetic fixtures, OSM geometry and optional allocation estimates**. They do
not mean the organizer provided direction labels. Heuristics that align a vehicle
run, schedule or ordered OSM stops must report source versions, temporal coverage,
ambiguous/unmatched share and sensitivity to alternative assignments. Show their
output as estimated and keep it out of the scored target and WAPE evaluation.
If those checks cannot distinguish directions, present only route-level forecasts
on the map and mark stop/direction overlays unavailable.

The delivery path is route-hour boarding forecast → evaluated month/day views →
graph-conditioned, labelled allocation. The competition metric and the graph
concept are **joint objectives**: improve route-hour WAPE while making the map
and estimated directions internally coherent. One must not be sacrificed to
claim the other. OD, onboard load and capacity cannot be claimed as observed
from validation counts alone.

For each `(route, date, hour)`, publish one calibrated route-level prediction.
An allocation layer can split that prediction over plausible route patterns,
directions and stops using schedule/vehicle/geometry evidence. Its nonnegative
components must sum back to the published route-hour total, or explicitly report
an unallocated remainder. Version the allocation weights, source dates and
confidence/ambiguity, and compare plausible alternative assignments. The scored
CSV is produced from the route-level total; it must not depend on an unvalidated
split. This makes the graph useful for dispatcher explanation and scenarios while
keeping WAPE evaluation honest.

After a reproducible baseline, test graph-derived **route-level features** such
as route topology, nearby event exposure or connectivity on the same chronological
holdout and cutoff. Accept them only when they improve route-hour WAPE across
relevant slices without leakage; a visually attractive direction reconstruction
does not establish forecast accuracy. The rest of this document describes the
longer-term modeling idea, conditional on new data and measured gains.

## Гипотеза

Мы прогнозируем не отдельный трамвай и не изолированный маршрут, а пассажирский спрос на перемещения внутри городской транспортной сети. Это делает прогноз переносимым на изменения маршрутной сети и события, которых не было в обучающей выборке.

## Представление сети

Городская сеть — ориентированный временной граф:

- узлы: трамвайные и автобусные остановки, станции метро, транспортно-пересадочные узлы и значимые зоны притяжения;
- рёбра: возможность перемещения или пересадки, время в пути, пропускная способность, регулярность и стоимость пересадки;
- динамические признаки: история валидаций, телематика, календарь, погода, городские события, ремонты и ограничения;
- целевая величина: поток пассажиров между частями сети по временным интервалам с интервалом неопределённости.

Прогнозы соседних узлов и рёбер зависимы: модель должна учитывать пространственное распространение спроса и временной лаг.

## Два этапа прогноза

1. **Demand forecasting.** Оценить origin-destination спрос и поток по рёбрам мультимодального графа.
2. **Assignment and capacity.** Распределить поток между маршрутами и подвижным составом с учётом расписания, интервалов, вместимости и пересадок.

Так можно переоценить загрузку после изменения сети без полного переобучения route-specific модели. Например, городской концерт создаёт внешний импульс рядом с площадкой; графовый слой распространяет спрос, а assignment layer показывает, какие участки примут нагрузку.

## Горизонты

- оперативный: ближайшие часы и сутки, высокий вес live-телематики;
- среднесрочный: месяц, календарь, события и сезонность;
- долгосрочный: год, изменения сети, расписаний и инфраструктуры.

Один горизонт не является простой агрегацией другого: набор признаков, частота переобучения и допустимая неопределённость различаются.

## What-if сценарии

Диспетчер меняет параметры сети и сравнивает сценарий с baseline:

- добавить или убрать трамваи;
- изменить интервалы движения;
- временно закрыть маршрут, остановку или ребро;
- добавить событие или локальный всплеск спроса;
- изменить доступность пересадки.

Результат содержит delta пассажиропотока, перегруженные участки, перераспределение на соседние маршруты и доверительный диапазон. Сценарий не перезаписывает опубликованный прогноз и хранит входные параметры для воспроизводимости.

## ML-кандидаты и проверка

Кандидаты (ST-GNN, temporal graph networks, graph transformers) выбираются только после baseline и backtesting. Минимальные сравнения: seasonal naive, gradient boosting с lag/rolling features и независимая временная модель. Split — по времени и по нетипичным событиям; отдельно оценивается перенос на изменённую сеть.

Метрики: MAE/WAPE потока, ошибка пиков, calibration интервалов, precision обнаружения перегрузки и качество assignment по участкам. Общая средняя метрика не может скрывать провал на пересадочных узлах.

## Границы текущего шаблона

Seed-данные и сценарный расчёт `network-flow-baseline-v1` демонстрационные. Они доказывают сквозной контракт UI → API → PostgreSQL, но не являются математической моделью для пилота. Production-модель подключается через `ForecastModel`, а сценарный solver — через отдельный application port.

# Сданные маршрутно-часовые прогнозы, 2026-09-27

Пользователь загрузил эти пять CSV в публичный лидерборд организатора в разделе Data Science. Файлы содержат только прогнозы: исходных валидаций и идентификаторов пассажиров в них нет. Значения баллов сообщил пользователь; [снимок экрана платформы](../../../docs/evidence/leaderboard-2026-09-27.png) подтверждает успешную отправку 27.09.2026 в 08:57 МСК и показанный балл 0.89195, но не показывает имя загруженного CSV. Выгрузка платформы в репозитории не хранится. Каждый CSV соответствует точной сетке организатора `route;date;hour;prediction`: 14 640 неотрицательных целочисленных прогнозов на ноябрь–декабрь 2025 года.

| CSV | Модель | Публичный WAPE-score, со слов пользователя | SHA-256 |
|---|---|---:|---|
| `submission-calendar_robust_routeshape.csv` | Равная смесь короткого и 28-дневного устойчивых профилей с коррекцией формы по маршруту и часу | 0.88670 | `89dee02fc045982c42bfef61e8614e8af4db15fba54ca58aa43e6236633f4a54` |
| `submission-calendar_robust_28_routeshape.csv` | 28-дневный устойчивый профиль с коррекцией формы по маршруту и часу | 0.88947 | `a045e21ef743b72bc8f643e7003af4bc5b6896207aaf7e90fe4eb6428071a311` |
| `submission-robust-hourshape.csv` | Равная смесь устойчивых профилей с коррекцией формы по маршруту, типу дня и часу | 0.88673 | `516f2fb4c7836126a2200c39c1a05a4af50d671249d8346711ce44726422cc5c` |
| `submission-catboost50-daily-bounded05-routeshape.csv` | 50% профиля и 50% ограниченной посуточной поправки CatBoost | 0.89146 | `1b7158f61fed50156b6f74cd53355f843a6c0c66362c4b7011cb5167381b4e86` |
| `submission-catboost50-uniform-recency60-routeshape.csv` | 50% профиля, 25% CatBoost с равными весами и 25% CatBoost с весами по давности | **0.89195** | `edc07cca26ff21dd5228f744dea79940eb81a65c3ce5368aa743e2fc2fd0f11f` |

Все пять файлов побайтово воспроизводятся включённой в репозиторий командой `tramflow-competition` из [`competition_cli.py`](../../src/tramflow_ml/competition_cli.py), с использованием [`competition_profile.py`](../../src/tramflow_ml/competition_profile.py) и [`competition_hybrid.py`](../../src/tramflow_ml/competition_hybrid.py). Для двух результатов CatBoost здесь также сохранены три исходных скрипта. `overnight_catboost_daily50.py` обучает первую посуточную поправку. `overnight_catboost_recency.py` обучает модель с той же целевой переменной и экспоненциальными весами обучающих моментов прогноза с полупериодом 60 дней. `overnight_catboost_recency_ensemble_submit.py` смешивает их с тем же 28-дневным профилем. Поправка каждого компонента CatBoost ограничена ±5% до смешивания. При имитации исторических моментов прогноза и в итоговом прогнозе от 1 ноября используются только метки до соответствующего момента; фактическая погода будущего не используется. Имена двух ML-моделей в CLI: `catboost_daily_bounded_50` и `catboost_daily_recency_ensemble_50`.

`producer_commit` в `current.json` и `approved.json` — исторический идентификатор
сборки; история Git в этом репозитории начинается с новой копии и такой коммит
здесь не открывается. Проверяемое соответствие файлов задают SHA-256: поле
`config_sha256` (`cf5240b9…`) равно хешу включённого
`overnight_catboost_recency_ensemble_submit.py`, а CSV и оценка сверяются с
`csv_sha256` и `evaluation_sha256`. [Список контрольных сумм](../../../docs/evidence/repository-artifacts.sha256)
позволяет проверить эти файлы локально командой
`sha256sum -c docs/evidence/repository-artifacts.sha256` из корня репозитория.

## Воспроизведение из архива организатора

Храните `dataset.zip` организатора вне Git. Укажите в `TRAMFLOW_DATA_DIR` закрытый каталог с этим архивом. Его SHA-256 должен быть `7e34e56b93f7379cb5abe9aca8e87967cb986111b9939caead5a3fea6c296e5a`. До заполнения отсутствующих агрегированных ключей нулями требуется доказательство сверки `reconciliation.json`.

```bash
uv sync --all-packages --all-extras --locked
export TRAMFLOW_DATA_DIR=/absolute/path/to/private-data
uv run --package tramflow-ml tramflow-competition \
  --archive "$TRAMFLOW_DATA_DIR/dataset.zip" reconcile \
  --output "$TRAMFLOW_DATA_DIR/reconciliation.json"

for model in calendar_robust_routeshape calendar_robust_28_routeshape calendar_robust_hourshape; do
  uv run --package tramflow-ml tramflow-competition \
    --archive "$TRAMFLOW_DATA_DIR/dataset.zip" \
    --reconciliation "$TRAMFLOW_DATA_DIR/reconciliation.json" \
    submit --model "$model" \
    --output "$TRAMFLOW_DATA_DIR/submission-$model.csv"
done

uv run --package tramflow-ml python \
  ml/competition_submissions/2026-09-27/overnight_catboost_daily50.py
uv run --package tramflow-ml python \
  ml/competition_submissions/2026-09-27/overnight_catboost_recency_ensemble_submit.py
```

В цикле CLI создаёт файлы с именами моделей; у загруженных CSV выше более короткие исторические имена. При сверке ориентируйтесь на SHA-256, а не на имя файла. Скрипты CatBoost записывают CSV и отчёты JSON в `TRAMFLOW_DATA_DIR`. Версия дополнительной зависимости зафиксирована: CatBoost 1.2.10. [Журнал экспериментов](../../../docs/experiments/route-hour-2025.md) содержит хронологические оценки, допущения об источниках и известную слабость на августовских окнах.

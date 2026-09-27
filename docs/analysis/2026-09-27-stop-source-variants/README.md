# Independent stop-source model variants

The 16 source masks are independently fitted CatBoost models. They are experimental
comparators, not a promotion over the existing stop model. The existing stop-model
artifact and final scored route model remain unchanged. Every mask is trained from
inferred stop pseudo-counts; no observed boarding-stop ground truth exists.

| Model | May 1, 61 days | July 1, 61 days | September 1, 61 days |
|---|---:|---:|---:|
| Existing calendar stop model | 0.863650 | 0.821191 | 0.885933 |
| All sources off | 0.667333 | 0.764937 | 0.710040 |
| All four sources on | 0.726095 | 0.749810 | 0.714443 |

The displayed score is `max(0, 1 - sum(abs(actual - round(prediction))) / sum(actual))`
after independently predicted stop values are summed by route/date/hour. It uses
actual route validation totals, separately from the inferred-stop WAPE metrics.
The complete results, all route slices, month slices and lead-week slices are in
`scores.json`. None of the 16 models beats the existing model averaged across the
three slices. September is an explicitly non-blind diagnostic. No quality gain,
causal effect, or public leaderboard improvement is claimed.

## Contracts and evidence

`frozen-plan.json` fixes the seed, CatBoost configuration, sample budget, source
checksums, chronological origins, target, baseline and acceptance criterion before
training. `dataset-audit.json` records the checksum-bound mapped target export and
reconciled organizer route labels. No raw passenger validation records are published.
The evidence level for stop labels is repository inferred mapping contract, not an
organizer-observed stop label. Train origins advance in 30-day steps, each using at
most 12,000 finite target cells and targets strictly before the outer origin.

Ordinary weekday/hour, stop identity, geometry and lagged count profiles remain
when all sources are off. Calendar-off removes month, holiday and special-day
grouping, including from historical stop and route profiles. Calendar-on adds
month, holidays and special-day profiles. Disabled sources contribute no columns;
there is no mixing of per-source predictions at serving time.

Weather uses the prior 28-day route/hour mean from historical weather, never actual
future weather. Traffic uses prior 56-day weekday/hour means of unedited official
citywide numeric snapshots with explicit support counts. Neither is a current
weather/traffic forecast or a route-specific traffic sensor.

Events are genuine KudaGo session starts collected on 2026-09-27, explicitly
retrospective. A bounded capture contains 100 cards from an API listing reporting
4,738 cards. Page 2 failed after three retries; coverage is not complete or random.
There are 631 resolved starts across January–October, 151 within 1 km of at least
one declared stop and 2,979 nonzero stop-hour tensor cells. A source feature is the
prior 84-day weekday/hour mean of these local start counts. Unresolved times,
unknown coordinates and attendees are not invented. Zero means no captured event,
not no actual event. The source was unavailable at the 2025 forecast origins and
must not enter the scored route forecast. Acquisition metadata and raw page hashes
are embedded in `frozen-plan.json` and the serving manifest; numerical source facts
are preserved in `event_starts.csv` with their matching manifest. Its original
CRLF bytes are preserved through a file-specific Git attribute so cross-platform
checkout cannot invalidate the recorded source checksum.

## Storage and reproduction

The experimental model registry is `data/planning/stop-source-models/model-{mask}.cbm`; the
serving manifest binds each weight file and each per-mask prediction gzip by hash.
Source bit order is calendar, weather, traffic, events. Prediction gzip files use
the established `independent-stop-forecast.v1` schema at full float precision; no
route-level normalization is applied. `stop_predictions_sha256` hashes the original
float64 C-order day/identity/hour tensor, with encoding named in provenance.
Serving loads exact frozen forecasts; changing masks does not retrain any model.

Reproduce from the dedicated worktree, with the existing locked boosting extra:

```sh
PYTHONPATH=ml/src .venv/bin/python scripts/build_stop_source_variants.py \
  --source ../duty-mapping-full-20260927/ml/artifacts/full-duty-mapping-v1 \
  --route-export ../full-features-20260927/ml/artifacts/full-features-v2/routes \
  --weather /Users/cute/MosTransport2026Hack/data/weather/2025/route-weather.csv.gz \
  --posts /Users/cute/MosTransport2026Hack/data/external-traffic/observations.jsonl \
  --events docs/analysis/2026-09-27-stop-source-variants/event_starts.csv \
  --events-manifest docs/analysis/2026-09-27-stop-source-variants/events-manifest.json \
  --output data/planning \
  --report docs/analysis/2026-09-27-stop-source-variants
```

The event capture remains partial; re-fetching later changes the source and must be
recorded as another experiment. CatBoost 1.2.10 is already declared and locked; no
dependency was introduced. Existing CatBoost categorical handling and fixed seed
follow the [official categorical feature contract](https://catboost.ai/docs/en/features/categorical-features)
and [training parameters](https://catboost.ai/docs/en/references/training-parameters/common).
The comparable repository implementation is `stop_models.py`; these variants use
smaller fixed training budgets for a bounded demo experiment, rather than a model
search. This makes quality loss a disclosed limitation, not a hidden fallback.

## Verification

- ML Ruff: passed.
- Mypy: passed, 134 source files.
- Full ML pytest suite: 1,488 passed in 68.74 seconds.
- Focused six source tests: passed, covering all masks, calendar-off isolation,
  future labels/weather/traffic/events invariance, 1 km spatial matching and hours.
- Saved all-off CBM reload: exact prediction-array equality observed.
- 48 chronological source fits and 16 final November–December fits: completed;
  each final file has 584 identities × 61 days × 24 hours = 854,976 values.
- Final publication audit: all 16 compressed files, 16 weight files and full
  prediction tensors match manifest hashes; all 854,976 values per file are finite
  and nonnegative. All 16 prediction tensors are distinct. Source-prefixed feature
  columns match each bit mask exactly. Total compressed forecast size: 66,708,462
  bytes (about 63.6 MiB), so serving uses a bounded per-mask cache.

## Integrated delivery and lead verification

Base `6ce7216db8535997bbf4d1e512603d19ff3d9fec`; branch
`agent/stop-source-variants-20260927`; dedicated worktree
`/Users/cute/MosTransport2026Hack-worktrees/stop-source-variants-20260927`.
Integration method: local fast-forward into `main` after the final checks.
The worktree and its isolated Docker preview are retained for demonstration;
the initial delivery was local; subsequent requested publication is recorded below. No other worktree or existing local preview was altered.

The initial screen retains the original `stop_model`. Choosing **Учёт источников**
selects `stop_sources` with all four flags enabled; each checkbox switches to the
exact corresponding fitted model. Manual factors remain independent. Requests
use the existing HTTP payload with one additive forecast-mode enum value. The
source flags default to false at the API boundary for backward compatibility;
the interface sends all four true when entering the new mode.

Lead verification on 2026-09-27:

- `make check`: exit 0; 430 backend passed / 52 SQL skipped, 1,489 ML passed,
  110 frontend passed, 119 contract tests passed, static checks/build/OpenAPI/Compose passed.
- `npm --prefix frontend run test:e2e`: 53 passed, including keyboard controls,
  390/768/1440px layout, errors/retry and obsolete-response protection.
- `make agent-up ID=stop-source-variants-20260927`: exit 0; isolated Docker stack
  built, clean database migrated and existing scored forecast published successfully.
- `make agent-smoke ID=stop-source-variants-20260927`: exit 0; database, API,
  forecast and frontend reachable.
- HTTP test exercised all 16 actual masks: 200, unique run IDs, exact mass sums,
  malformed flags rejected with 422, approved reference values unchanged. Maximum
  observed warm request 45.2ms and cold request 213.1ms on this workstation via
  TestClient; this is not a deployment load test or p95 claim. Cache retained two assets.
- Live browser against Docker: original baseline → all-on mask15 → weather-off
  mask13 changes actual predictions → baseline restores identical points and run ID.
  Desktop/mobile screenshots and no-overflow check passed.
- Independent backend/ML/frontend reviews found no remaining actionable issues.
- `git diff --exit-code -- ml/competition_submissions data/planning/stop-model.json.gz
  data/planning/stop-model-manifest.json`: exit 0. No dependency or lock changes.

The first frontend attempt used shell Node20 and could not start Vitest; using
installed Node24.19 resolved it. Initial live-smoke exact-text selector excluded
its nested forecast number; correcting that test selector passed without a
product-code change. Logs remain `/tmp/stop-sources-{check,final-e2e,stack,smoke,live}.log`.
Preview: http://localhost:47321 (API http://localhost:37321).

## Remote-main integration

The subsequent requested publication merges remote `3cf951d`, preserving its
anchored-v3 default stop model, timeline playback, direction arrows, insights and
range totals. Source variants retain the earlier inferred v1 training allocation;
the recorded calendar_stop_reference is historical, not the refreshed live model.
No new source-quality claim follows from comparing these different mappings.

The refreshed catalogue changes only first_date and ordering; all 574 physical
route/stop/direction/name/lat/lon tuples are identical. The serving manifest now
records the old training catalogue hash separately from the current serving
catalogue hash and binds their spatial projection. No model weight or source
variant prediction bytes changed. The original builder remains hash-reproducible;
after running it, bind its results to the current spatial catalogue explicitly:

```sh
git show 735d8a5:data/planning/forecast.json.gz > /tmp/stop-source-original-planning.json.gz
.venv/bin/python scripts/bind_stop_source_catalog.py \
  --previous-planning /tmp/stop-source-original-planning.json.gz \
  --current-planning data/planning/forecast.json.gz \
  --variants-manifest data/planning/stop-source-variants-manifest.json
```

The binding rejects changed identities, names or coordinates and never rewrites
training provenance. Source files/weights/forecasts are immutable for this delivery.

Publication follow-up includes remote commits through `61e2171`, with ordinary
merges preserving both histories. The compact demo layout and source defaults
coexist; the applicability card uses the selected response mode and does not
assign baseline accuracy/coverage to source experiments. Latest lead checks:
`make check` exit0 (434 backend,1506 ML,127 frontend,119 contracts;52 SQL skipped),
52 browser tests passed. Docker smoke and live baseline→mask15→mask13→baseline
passed after catalogue integration. Source weights, source forecasts and scored
competition files remain byte-identical to `735d8a5`. Independent review found no
blocking issue. No new dependency or migration. Logs:
`/tmp/stop-sources-push-gate.log`, `/tmp/stop-sources-push-e2e.log`.
The first push was refused because another normal commit reached remote main;
its changes were merged and rechecked, without forced history replacement.

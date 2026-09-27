# ADR 0008: isolated experimental route-hour forecasting

Status: accepted for offline experiments; not a production publication decision.

Organizer labels count successful validations per route/hour across all directions.
Existing forecast.v1 publication requires stop/direction identities unavailable in this
extract. Manufacturing those identities would misrepresent data. Keep forecast.v1 and
backend/frontend unchanged. Introduce `route-forecast.experimental.v1` only inside ML:
complete ten-route 61-day Moscow hourly grid, immutable values, explicit real-data target,
source digest, feature/model versions, origin/cutoff/generation timestamps.

Train and batch inference remain offline. Store binaries in ignored ml/artifacts with
manifest hashes; load only trusted local bundles. CatBoost is an optional ML extra.
A route projection supports hourly day and daily calendar-month aggregation and rejects
year, partial coverage and invalid dates. Tests verify the existing domain/HTTP response
can represent these counts with explicit route-number→internal-ID mapping, no stops,
no capacity and no uncertainty interval. This is not a SQL publisher or endpoint switch.

Temporal validation uses frozen origins with historical labels only. Missing label keys
mean zero in the supplied extract, not certified service/source coverage. Historical
availability timestamps are absent: retrospective perfect delivery is an assumption.
October confirmation follows model selection; September overlaps development. Earlier
EDA exposed October baseline, so this is not blind evaluation.

Before production: version the route-only publication contract, resolve real route IDs,
write atomically through repository infrastructure, check endpoint selection and UI target
labels, and implement temporally validated uncertainty intervals. Never send real counts
through the demo/synthetic stop contract or fill map positions with guessed telemetry.
Rollback offline experiment code has no DB or API migration consequences.

Evidence: [experiment report](../analysis/2026-09-25-route-ml/README.md).
Research: [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common),
[rolling-origin evaluation](https://otexts.com/fpp3/tscv.html).

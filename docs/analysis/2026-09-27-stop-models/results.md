# Real stop-target experiment results

Ten methods were evaluated on the complete regenerated duty/calendar/clock mapping,
with **30 real 61-day evaluations**. The selected independent stop model is the
seasonal calendar/weekday blend: scores **0.863650 / 0.821191 / 0.885933** at
May 1 / July 1 / September 1. It does **not** meet stable 0.88, and none of these
stop models supports a 0.93 claim. Boosted stop models did not improve the strong
calendar baseline. Selection uses mean May/July score; September is a non-blind
diagnostic. No hidden November–December score is available.

|Model|May|July|September|
|---|---:|---:|---:|
|Independent seasonal calendar profile|0.862620|0.816573|0.880647|
|Independent calendar/weekday 50:50 blend|**0.863650**|**0.821191**|**0.885933**|
|CatBoost MAE absolute count|0.000000|0.813814|0.652790|
|CatBoost RMSE absolute count|0.605284|0.795036|0.734461|
|CatBoost RMSE residual to profile56|0.733376|0.678665|0.715487|
|CatBoost RMSE relative to profile84|0.667202|0.312218|0.709941|
|Relative model, clipped and shrunk 25%|0.865830|0.749852|0.777680|
|Day-type profile28|0.825527|0.811985|0.790125|
|Day-type profile56|0.850745|0.779063|0.766999|
|Day-type profile84|0.860796|0.753193|0.794242|

Absolute MAE's May failure is severe **overprediction**, not evidence that the
learner merely returned zeros: its all-bucket pseudo-WAPE is 9.206 and known-stop
pseudo-WAPE 15.737. No post-hoc cap was fitted to hide that failure. Relative
regression is similarly unreliable; shrinking its correction limits damage but
does not improve the selected mean May/July objective.

## Target evidence and coverage

Source manifest SHA-256:
`0748c3c1a3f40b3f950c58b8cf33a62710dd0a20b246c5b59aa2c008ddc96e97`.
There are 1,796,352 nonzero/supplied source rows and 59,667,191 successful
validations, reconciled against the organizer labels separately for every
route-date-hour. The declared catalogue has 574 route/stop/direction identities,
439 distinct stop IDs, and 10 added route-specific unknown buckets. Its identity
universe is fixed independently of which target rows happen to exist.

The complete inferred grid has 304 × 584 × 24 = 4,260,864 rows. These are
**inferred counts/zeros**, not observed stop boarding labels. Training samples
at most 60,000 rows at each monthly origin with seed 20260927; the complete
unsampled feature export is provided by the parent full-dataset pipeline.
All source rows are eligible under this mapper, so no additional block is masked
in this run. The generic training policy excludes an entire route-hour if any
source row is ineligible; excluded mass remains in the audit and evaluation.

Only 21,060,982 validations (35.30%) receive inferred stop assignments.
38,606,209 remain in explicit `unallocated:<route>` buckets. Assignment rates
change substantially with feed availability:

|Month|Assigned to inferred stop|
|---|---:|
|January|0.97%|
|February|1.00%|
|March|1.22%|
|April|28.09%|
|May|56.01%|
|June|57.16%|
|July|54.60%|
|August|53.40%|
|September|53.07%|
|October|52.60%|

This is a changing **mapping-observation process**. A low known-stop target can
mean unavailable mapping rather than little passenger demand. Models trained on
these pseudo-counts cannot identify that distinction from passenger labels alone.
Calendar-trip uniqueness and clock-based nearest-stop assignment are not independently
verified true stop locations. Catalogue geometry is retrospective; no historical
point-in-time topology claim is made.

For the selected model, known-stop pseudo-WAPE is 0.888 / 0.654 / 0.739;
all-bucket pseudo-WAPE is 1.032 / 0.461 / 0.751. Those are agreement metrics
against the inferred mapper, **not real stop accuracy**. May contains 1,010,621
known-stop target mass on identities with no earlier positive count; July and
September unseen-identity mass is zero. Good aggregated route scores therefore
do not validate the geographic forecast. Monthly and lead-week route errors for
every model are included in `scores.json`.

## Final artifacts

`ml/artifacts/new-duty-stop-models-v3/stop_predictions.csv.gz` contains 854,976
continuous stop/bucket forecasts for November–December. It records date, route,
hour, stop ID, direction, model version, generation timestamp, label provenance
and unavailable uncertainty bounds. Total predicted count is 12,296,438.68;
unknown buckets contain 8,879,409.56 (**72.21%**). The historical non-summer
mapping gap heavily influences that unknown fraction. Never place this unknown
mass on a stop or present the map as observed onboard occupancy.

`submission.csv` contains exactly 14,640 unique route-date-hour rows. Independent
continuous predictions are summed first and rounded once. Reloading both files
and comparing every route-hour confirmed exact equality to rounded stop sums;
route 5 remains zero. This experimental upload candidate is weaker in local
validation than the best separate route statistical methods; it is not promoted
as the production scoring submission.

Small result receipts are committed here; large forecasts remain local, with
hashes and sizes in `artifacts.json`. The selected model is a deterministic
calendar statistic, so its reproducibility is bound by code/config/source hashes
rather than a serialized learned weight file. No new dependencies were added.

## Reproduction and checks

From this worktree, use its `ml/src` on `PYTHONPATH` and the installed project
runtime. A complete fresh run is:

```sh
python -m tramflow_ml.stop_models \
  --source ../duty-mapping-full-20260927/ml/artifacts/full-duty-mapping-v1 \
  --archive '/Users/cute/MosTransport2026Hack/dataset (1).zip' \
  --proof /tmp/tram-eda/results/reconciliation.json \
  --output ml/artifacts/reproduced-stop-models
```

The recorded v3 run reused the 24 unchanged CatBoost/recent-profile results
from v2 and added six strong-calendar evaluations. The receipts bind both code
versions and score hashes. Later CLI validation requires explicitly supplying
compatible code and score hashes when reusing results; the numerical prediction
functions were unchanged by that validation hardening.

Observed checks: focused stop-model tests 16 passed; Ruff passed; mypy passed.
The full final ML suite passed 1,029 tests in 23.12 seconds.
The strong-baseline linear aggregation comparison covers both variants and all
three origins to `1e-5` tolerance (the route implementation stores features as
float32). Full final integration checks belong to the parent worktree.

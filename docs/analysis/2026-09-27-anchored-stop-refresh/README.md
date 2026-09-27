# Anchored stop forecast refresh

The fixed incumbent `calendar_weekday_blend` now forecasts anchored-v3 inferred
counts. Unknown November–December forecast mass falls from **72.2112% to
22.3773%**. The 854,976 predictions cover 584 identities and 61 × 24 hours; all
are finite and nonnegative. Total predicted mass is 12,296,438.6845. These are
inferred allocations, not newly observed stop labels or evidence of spatial
accuracy. The scored competition model and submission are unchanged.

`results.json` records the complete 12-window comparison, both source manifests,
source file hashes, catalog hash, producer hash, exact configuration and timestamp.
Both baseline and refreshed stop errors are measured against the same new inferred
targets. They are diagnostic only. Historical GTFS and transferred calendars are
retrospective; count inputs end before every forecast origin. All future-count
poison checks passed.

| Origin | Horizon | Actual route score, refreshed |
| --- | ---: | ---: |
| 2025-05-01 | 61 days | 0.8636497452 |
| 2025-07-01 | 61 days | 0.8211894326 |
| 2025-09-01 | 61 days | 0.8859337358 |

Continuous route predictions differ by at most 2.73e-12 across evaluation windows,
and 1.82e-12 for November–December. The original raw rounded-score criterion
failed: changed floating-point summation order flips a few exact half-integer
rounding ties. The preregistration records this failure and the narrow numerical
amendment. No model, rounding rule or stop mass was modified to force equality.
All changed rounded cells differ by one passenger and both values lie within
1e-8 of the same half-integer; observed score deltas satisfy the measured
changed-cell-count / actual-mass bound. `results.json` retains raw scores and
quantization counts, including 17, 58 and 9 changed cells on the 61-day windows.

## Reproduce

From the dedicated worktree, with its locked Python environment:

```sh
PYTHONPATH=ml/src .venv/bin/python scripts/refresh_anchored_stop_model.py \
  --old-source ../duty-mapping-full-20260927/ml/artifacts/full-duty-mapping-v1 \
  --source ../stop-schedule-warp-20260927/ml/artifacts/full-anchored-transfer-v3 \
  --route-export ../full-features-20260927/ml/artifacts/full-features-v2/routes \
  --output ml/artifacts/anchored-stop-refresh-v1 \
  --report docs/analysis/2026-09-27-anchored-stop-refresh
```

Raw data and full experiment exports remain ignored. The serving serializer uses
`stop_predictions.csv.gz`, `run.json`, and `dataset_audit.json`; its compact map
artifact is managed by the lead integration task. Generated timestamps make
compressed CSV bytes run-specific; numeric predictions are deterministic.

Focused checks: `pytest ml/tests/test_anchored_stop_refresh.py -q` passed 3 tests;
`ruff check --no-cache scripts/refresh_anchored_stop_model.py
ml/tests/test_anchored_stop_refresh.py` passed. Final integration gates are recorded
in the lead task tracker.

## Serving integration and observed verification

Both committed planning assets were rebuilt with the new catalog and canonical
mapping method; the API warns about transferred calendars. Independent review
and lead comparison matched every 854,976 source CSV value exactly; see
`serving-check.json`. Old/new catalog rows are compared by identity, not order:
574 identities, names and coordinates match; first_date and ordering differ.
Scored route rows and approved CSV SHA edc07cca26ff21dd5228f744dea79940eb81a65c3ce5368aa743e2fc2fd0f11f are unchanged.

Lead `make check` exit0:404backend/52SQLskipped,1492ML,107frontend,119contracts;
Ruff/mypy/build/OpenAPI/golden/Compose pass. Focused planning tests30passed.
Docker build and smoke pass; live API day/month/year conservation and three
source cells/factor1.2 match (`live-api.json`). Browser map ready, factor1.5
changes1888.4→2832.6, no JS errors or390px overflow (`live-ui.json`). Five planning
browser tests pass, including accessibility and map-marker selection.
Initial build required Docker's credential-helper directory on PATH; retry passed.
Initial ad-hoc browser selector matched two model labels; narrowing the selector
passed without changing product code.

User preview remains on http://127.0.0.1:8081 with updated backend/frontend and
existing database volume. No dependency or database schema change. Publication
base5c83b22; branch agent/anchored-stop-refresh-20260927; worktree
/Users/cute/MosTransport2026Hack-worktrees/anchored-stop-refresh-20260927 retained
with ignored full exports for reproduction. Integration incorporates ML commit
11bb48a and external experiment cherry-pick98af184; final publication is a
normal fast-forward push of this branch to main.

The separate [external-source experiment](../2026-09-27-external-residual/README.md)
was rejected: selected correction .858660/.840260/.888174 loses May/September
to the incumbent .858811/.839799/.892071. No source model was promoted and no
new leaderboard score or full external-source points are claimed.

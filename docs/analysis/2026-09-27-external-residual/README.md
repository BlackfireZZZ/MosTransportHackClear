# External-source residual experiment

Status: completed experiment, **rejected for promotion**. No scored CSV or serving
asset changed. A fixed, bounded correction around the strong robust28 model still
fails the preregistered improvement requirement. No model/search extensions followed
inspection of the scores.

| Model / extra residual features | May 1 | July 1 | September 1 |
|---|---:|---:|---:|
| Robust28 reference | 0.865371 | 0.830310 | 0.891882 |
| Published hybrid, independently refitted | 0.858811 | 0.839799 | 0.892071 |
| Residual without extra source | 0.858108 | 0.822681 | 0.888754 |
| Calendar | 0.858367 | 0.839336 | 0.887952 |
| Weather | 0.858042 | 0.839615 | 0.888572 |
| Traffic | 0.858546 | 0.823272 | 0.889273 |
| News | 0.858108 | 0.833441 | 0.888488 |
| All four | 0.858660 | 0.840260 | 0.888174 |

The predeclared May/July mean selects `all`. It loses to robust28 in May and
September and to the published hybrid in both months. Its July improvement alone
does not qualify. All three windows contain 61 days and the same complete hourly
route grid. September has already been inspected in previous experiments and is
not a blind holdout. [Scores and route/month slices](scores.json),
[selection and incremental source effects](decision.json), [frozen plan](plan.json).

Traffic improves the residual-control score on all three windows by
0.000438 / 0.000591 / 0.000518. This is a measured **incremental effect inside a
weaker experimental model**, not an improvement over the deployed forecast.
The traffic variant remains below both strong references on every window. Calendar
and weather improve July but regress in September; news has zero May effect.
Consequently this result does not demonstrate four useful production sources or
justify a claim of four source points from the jury.

## Evidence and constraints

Evidence level: organizer label contract plus checksum-bound inspected aggregate
export; existing official-source factual extract; repository model contracts;
team-inference residual architecture. Original source limitations remain those in
[external-factors](../2026-09-27-external-factors/README.md): only 163 accepted
citywide traffic snapshots over 106 days, lagged news publication counts, revised
weather without original vintage evidence. No new data was collected.

The [official CatBoost fit documentation](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit)
supports real-valued regression targets. Here MAE fits absolute residuals from the
existing calendar robust28 anchor, in boarding-count units. The correction is
clipped to ±2.5% of each anchor, preserving zero anchors. This is an explicit,
reversible experimental choice; bounds limit changes but cannot guarantee quality.
Existing CatBoost 1.2.10 is reused. No dependency or lock changes.

The six variants share 200 trees, depth 5, learning rate .05, seed 20260927,
four threads, and L2=10. Training origins are 14 days apart. Each training anchor
uses only data preceding its own origin; each complete training target horizon
ends before the outer origin. Weather/traffic/news features reuse the existing
past-only transformation. Calendar already influences robust28, so the calendar
ablation measures extra residual features rather than removing all calendar input.
All models include the strong anchor as a feature. No random time-series split,
future actual weather, future labels, or target-period source observation is used.

Acceptance fixed before fitting: gain >=0.0001 on each development window and no
September regression against **both** robust28 and the independently recomputed
published hybrid. Source effects are separately compared with residual control.
Scores are calculated after rounding once. A route with no observed mass has null
score and retains its absolute error; route and month slices are recorded.

## Reproduce

From the repository root with the locked ML environment:

```sh
PYTHONPATH=ml/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  scripts/experiment_external_residual.py \
  --labels /private/full-features-v2/routes \
  --weather data/weather/2025/route-weather.csv.gz \
  --posts data/external-traffic/observations.jsonl \
  --plan docs/analysis/2026-09-27-external-residual/plan.json \
  --output /tmp/external-residual-reproduced
```

The label loader verifies all 72,960 January–October keys and total mass
59,667,191 against a checksum-bound manifest. [Run manifest](manifest.json) binds
labels, weather, posts, frozen plan and final script. Only aggregated metrics and
public numerical covariates are committed. No passenger-level record or fitted
model is stored. A rejected candidate has no new competition submission.

## Ownership and handoff

Owner: `ml_gap_review`, only this report directory, the new experiment script and
`ml/tests/test_external_residual.py`. Base
`5c83b22d5ef772849c75d3cd950191de658cede5`; branch
`agent/external-residual-20260927`; worktree
`/Users/cute/MosTransport2026Hack-worktrees/external-residual-20260927`.
The integration owner controls the shared tracker, final gate and publication.
The worktree is retained for review; no runtime services were created.

## Observed verification

- `pytest ml/tests/test_external_residual.py ml/tests/test_external_factors.py ml/tests/test_competition_hybrid.py -q`: **19 passed**. Includes future-source perturbation invariance, Moscow-date boundaries, missing/disabled source semantics, correction bounds and zero-mass metrics.
- `ruff check ml scripts/experiment_external_residual.py`: passed.
- `mypy ml/src`: passed, 133 source files.
- Two complete runs reproduced `scores.json` and `decision.json` byte-for-byte; all 24 model-origin records and route/month slices matched. The final manifest uses the final script with optional CatBoost imported only inside training.
- `git diff --check`: passed. Parent owns the combined `make check`; it has not been claimed here.

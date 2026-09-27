# Independent stop-hour forecasting protocol

Status: complete new duty/calendar/clock mapping was consumed; 10 methods and 30
real temporal evaluations completed. See [results](results.md), [scores](scores.json)
and [statistical EDA](statistical-eda.md). The older soft mapping was not fitted.
The first new-map pilot used target-derived identity support and is superseded;
reported results use only the declared catalogue plus explicit unknown buckets.

Hypothesis: independently predicting stop × route × direction × hour pseudo-counts
can improve held-out actual route-hour WAPE-score after unconstrained aggregation.
The organizer's route-hour target, 14,640-row submission grid, and raw mass remain
unchanged. Stop errors only assess agreement with an inferred target and cannot
establish boarding-location accuracy.

Predeclared comparison: 28/56/84-day day-type stop profiles; CatBoost MAE and RMSE
on absolute stop counts; CatBoost RMSE on residuals to the 56-day profile. All boosted
models use 250 depth-6 trees, learning rate 0.08, seed 20260927, four threads. Existing
optional CatBoost 1.2.10 dependency is reused. No dependencies are added.

Origins are May 1, July 1 and September 1, each with 61-day horizons. Select by mean
May/July actual route score. September is a disclosed non-blind diagnostic. Historical
monthly origins generate direct-horizon training examples, capped at 60,000 per
origin with a fixed seed. Each feature uses observations strictly before its origin;
training targets stop before the outer evaluation origin. Actual future weather is
excluded. Profiles, route profiles, hour, weekday, civil day type, holidays, month,
lead, identity, direction, geometry and explicit unknown-location status are inputs.

Known-stop counts plus a separately predicted `unallocated:<route>` category must
reconcile to every actual route-hour before any inferred zeros are filled. Missing
stop cells are never described as observed zeros. Any ineligible source row excludes
its whole route-hour block from training, including absent stop cells. Excluded mass
remains in the audit and full evaluation. Route 5 retains an explicit unknown bucket
and all submission hours. No unknown mass is redistributed to known stops.

The independently predicted continuous stop counts are summed and only then rounded
for official route scoring. No route model imposes their totals. Reports also include
known-stop pseudo-WAPE, all-bucket pseudo-WAPE, eligible pseudo-WAPE, unknown mass,
unseen-identity mass and per-route scores. Map coordinates may be retrospective and
are not proof of historical availability. Sequence is omitted because a stop can occur
in several patterns with conflicting sequence positions. Uncertainty bounds remain
null and marked unavailable; point predictions must not fabricate calibration.

Primary research: [CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression)
defines MAE/RMSE objectives; [categorical features](https://catboost.ai/docs/en/features/categorical-features)
supports route/stop/direction identities. Those methods apply to fractional pseudo-counts,
but do not correct mapping error. MAE matches absolute-error evaluation; RMSE and
residual regression test whether smoothing high-volume counts helps aggregated WAPE.

Verification: `pytest ml/tests/test_stop_models.py` checks future-perturbation invariance,
outer-origin target separation, continuous aggregation, explicit eligibility exclusions,
reconciliation guards, date boundaries and unknown-bucket separation. Synthetic fits
of all three CatBoost variants produced finite nonnegative `(5, 2, 24)` predictions.
Real score acceptance remains >=0.88 on every declared slice, with 0.93 aspirational.

Two predeclared extensions added RMSE relative-to-profile84 and a 25% shrunk,
clipped correction, plus the existing strong seasonal calendar and calendar/weekday
blend as independent per-stop means. The strong baseline sum matches its route
counterpart (tested before rounding); route totals are never imposed.

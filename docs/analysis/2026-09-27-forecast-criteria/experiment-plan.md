# Frozen comparison: stop residuals and actual next-day origins

Before model fitting: version stop-calendar-bounded.v1, CatBoost200iterations,
depth5,RMSE,seed20260927; at most20000rows per28-day training origin. Baseline
is calendar_weekday_blend per inferred stop, summed without route constraints.
New features:7/14/28-day day-type profiles, stop share of route mass and mapping
coverage; strongest calendar profile supplies residual anchor, correction capped
at5%. Existing scored model and CSV remain unchanged.

Development origins May1/July1; September1already used, explicitly diagnostic.
Each61-day window reports actual route WAPE-score, per-route/month/leadweek and
pseudo-stop error separately. Acceptance: improve both development windows and
no September regression before proposing promotion; no inferred spatial accuracy
claim. Next-day samples May1/8/15,July1/8/15,September1/8/15 refit at every origin,
compare stop baseline/refinement and current route CatBoost hybrid. Nine days
are a bounded diagnostic, not a complete operational benchmark.

Evidence: checksum-bound complete-features.v1 dense route labels and full-duty
mapping manifest. Labels are observed successful validations, stops inferred.
No missing-source zeros or source support from target-period observations.

Research: https://otexts.com/fpp3/tscv.html distinguishes rolling next-day from
fixed multi-step origins; https://otexts.com/fpp3/reconciliation.html explains
bottom-up sums. Our inferred lower-level targets violate any assumption of
observed stop truth; route score is the acceptance metric. Existing CatBoost
and clipped residual machinery are reused; no new dependency.

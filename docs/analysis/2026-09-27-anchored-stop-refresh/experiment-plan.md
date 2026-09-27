# Preregistered anchored stop forecast refresh

Recorded before running the refresh. Evidence: organizer target contract in
`docs/product/DATASET.md`, inspected mapping manifests, and repository contracts
`stop_models.prepare`, `calendar_stop_profile`, `evaluate` and
`scripts/build_direct_stop_artifact.py`. Stop allocations and directions are team
inference, not observed labels; all geometry and schedule transfer are retrospective.

Falsifiable claim: replacing old duty-mapped targets with anchored-v3 targets in
the fixed incumbent `calendar_weekday_blend` reduces final unknown forecast mass
below 30%, while route-level scores and continuous route forecasts remain within
1e-8 of the old mapping on identical chronological windows. Scored competition
artifacts remain unchanged. No search, fitted residual model, or route rescaling.

Compare 61-day horizons at 2025-05-01, 2025-07-01, 2025-09-01 and one-day
horizons at May/July/September 1, 8, 15. These are non-blind diagnostics. Report
per-route, month and lead-week scores; stop pseudo-label WAPE is diagnostic only.
Inputs end strictly before each origin. Test invariance to poisoning all later
counts. Frozen onset calibration ends April 30; later reference geometry does not
establish historical spatial accuracy. The final origin is November 1, history
ends October 31, and the horizon is 61 days in Europe/Moscow.

Acceptance: sources checksum verified; complete route-hour mass reconciliation;
identical identity universes and missingness; all 854,976 final stop-hour values
finite and nonnegative; unknown share <30%; route forecast max absolute difference
and route score difference ≤1e-8; no future count dependence. Uncertainty remains
explicitly unavailable. Compare both models against the same new inferred stop
targets for the diagnostic stop errors.

Smallest disproof: synthetic conservation and future-poison tests, then the full
12-window comparison and serialization-grid audit. Existing serving serializer
owns public-contract checks; final integration is performed by the lead agent.

## Numerical acceptance amendment

The first full run stopped at the original score gate although its continuous
route predictions passed 1e-8. The linear redistribution changes floating-point
summation order, which can place mathematically exact half-integers on opposite
sides of NumPy's round-to-even boundary. Before completing the run, the lead
approved a narrow amendment: continuous-route tolerance remains 1e-8; all cells
whose integer rounded prediction changes must be within 1e-8 of a half-integer
and differ by at most one passenger. Raw score changes must be bounded by the
measured number of such cells divided by actual mass (plus 1e-12 numerical
slack). Preserve both raw scores, the changed-cell count and implied absolute
error delta. No stop forecasts are rescaled or quantized to force equality.

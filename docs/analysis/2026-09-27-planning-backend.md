# Planning API: inferred spatial allocation and explicit scenarios

Evidence: organizer route/date/hour target and `message2081` in
[the task contract](../product/TASK.md); selected approved CSV; full duty mapping
catalog and v3 independent stop predictions. Acceptance: the unfiltered selected
CSV total equals known spatial mass plus unknown mass for every returned bucket;
scenario weights cannot alter the stored forecast. Stops and directions
remain inferred, never current vehicle positions or measured stop labels.

`POST /api/v1/planning/forecast` is additive. `route` is the organizer route
number string, not the SQL surrogate ID. Start dates are November–December 2025.
Day returns 24 hourly buckets; month returns daily buckets through the end of the
selected calendar month. Year requires the first day of a month and returns 12
calendar-month buckets. All boundaries are Moscow local half-open intervals.

The default `approved` mode uses the checksum-bound selected route CSV. Spatial
shares average the independent stop forecast over matching route/day-of-week/hour
cells. An unknown stop or missing coordinate catalogue match stays unknown mass;
shares are not renormalized over known stops. The route total remains the approved
model, not the weaker independent stop model. Stop IDs retain the GTFS namespace;
there is no implicit OSM crosswalk. The source catalogue itself is retrospective.

Stop/direction filters select only attributed mass. Unknown mass has no justified
membership in either filter. Every response therefore retains both selected totals
and complete `route_baseline`, `route_scenario`, `route_unallocated_baseline` and
`route_unallocated_scenario` denominators. The full stop catalogue is returned
independently of selection. An incompatible stop/direction pair is rejected.

`forecast_mode=external_experiment` reads the separate hash-bound five-branch
model asset. `source_enabled.events` maps to `news`: historical official incident
and closure messages, not KudaGo attendance. The exact offline mixing rule is
mean(base plus enabled branches), rounded once per route-hour using ties-to-even,
before aggregation. Scores remain an unpromoted experiment and are exposed in
`experimental_evaluation`. The run identity includes artifact hashes and selected
branches. The source collector's report carries coverage and temporal limitations.

`factors.*.multiplier` retains its API name but means a raw nonnegative model
weight in [0,3]. Disabled factors have weight zero. The selected baseline has
fixed weight 1. For each route-hour, `scenario = (baseline + sum(weight_i ×
source_forecast_i)) / (1 + sum(weight_i))`; thus weather 2 and events 1 yield
base 25%, weather 50%, events 25%. In `approved`, factors use the newly trained
`approved-source-variants.v1` branches; `events` means the incident/closure
notices branch. The frontend sends only the factor selected by the user;
`source_enabled` stays false there. API clients using source flags alone retain
the fixed 0.05 candidate path for compatibility. If both contracts are sent,
factor weights take precedence and each branch is mixed exactly once. In
`stop_model` and `external_experiment`, manual factors still use
the older external variant artifact (`events` maps to `news`). Missing or
invalid variants fail with 503 when their weight is positive. The route-hour scenario is distributed to
stops in proportion to the selected base's spatial cells, preserving unknown
mass; periods aggregate those hourly results. The raw preset and normalized
weights travel in the response and CSV provenance. The blend is a scenario,
not a validated improvement or a guarantee against overprediction. Source
flags in approved mode without factor overrides use the candidate blend and
identify its unsubmitted model version.

For dates beyond December, year repeats the mean weekday/hour profile of the two
forecast months. This is a qualitative scenario, not a trained annual model or
validated annual seasonality. The transparent repeat-profile approach is motivated
by [FPP3 simple seasonal and mean benchmarks](https://otexts.com/fpp3/simple-methods.html).
Our input differs: two months of predictions, not a year's observed seasons, so
annual accuracy is not claimed. New routes, structural changes and yearly seasonal
variation are unsupported.

## Artifact and recovery contract

`data/planning/forecast.json.gz` contains public aggregates only; its manifest binds
approved CSV, generator, stop prediction, stop run, source audit and catalogue hashes.
The source labels and private event identifiers are not published. The generator
requires the selected manifest identity in the approval registry, matching stop
catalogue SHA, nonnegative finite inferred predictions, and the exact Nov–Dec
forecast date range. Loader validates complete 14,640 route-hour grid, finite counts,
spatial share sums and coordinates. Altered or absent artifacts return HTTP 503;
bad user parameters return 422. No SQL writes or schema changes are involved.

The planning artifact is a separately versioned projection. Whenever the approved
model is promoted, regenerate this artifact and rebuild the container; it does not
silently track a changed SQL active pointer. The response exposes its exact model
and run hashes. The default artifact is cached per API process. Rollback restores
the preceding committed artifact and rebuilds/restarts the API.

Reproduction:

```sh
python3 scripts/build_planning_artifact.py \
  --predictions /path/to/new-duty-stop-models-v3/stop_predictions.csv.gz \
  --catalog /path/to/full-duty-mapping-v1/stop_catalog.csv
```

The adjacent source `run.json` and `dataset_audit.json` are required. Private source
location used here is the retained `stop-models-20260927` and
`duty-mapping-full-20260927` worktrees. No dependencies were added.

## Observed verification

Base `61bd6d9`; worktree `planning-backend-20260927`, branch
`agent/planning-backend-20260927`. Focused API/unit checks: 17 passed. Full backend:
391 passed, 52 SQL tests skipped (no database change). Ruff, mypy (52 files),
architecture check and `git diff --check` pass. Tests cover exact route total,
known+unknown conservation, filters, invalid pairs, day/month/year boundaries,
leap arithmetic, multipliers, corruption→503 and real ASGI requests. Experimental
mix tests cover ties-to-even and changed identity with enabled sources.

Real ASGI smoke with the independently produced model-variants asset
`a58994ebaaafec6b8e7116e243d0702b828900efedd4c3c78fe0aaabb59484ca` returned
HTTP 200 for approved and experimental day/month/year (24/30/12 points). Route 1
approved Nov 1 sum is 22,648, November 525,410. Local in-process timings were
8–55 ms, not deployment RPS/p95 measurements. Lead owns final merged OpenAPI,
frontend, Compose and complete repository gates; those are not claimed here.

# Reference data and forecast contracts

These are strict pre-runtime reference validators, independent of the demo HTTP
API. Production packaging belongs to TASK-029; online workers must not import
training code. Organizer raw columns, target and service-day policy remain unknown.
[ADR-0005](../docs/decisions/0005-provisional-data-calendar-contract.md) records the
provisional policy and compatibility boundary.

## Time and identity

`calendar_v1.py` implements `moscow-midnight.v1`: aware instants, Europe/Moscow
service days, half-open windows, exact-hour day origins (24 elapsed hourly buckets),
first-of-month midnight month origins (daily buckets), and January 1 midnight year
origins (12 calendar months). UTC-equivalent instants are equivalent inputs.
Arbitrary-date month/year origins require a different calendar policy version.

`data_v1.py` defines normalized `data.v1` rows, not an organizer adapter:

- `EntityCatalog` holds opaque route/stop IDs and direction-specific ordered visits.
  Equal display names remain different IDs. Repeated stop visits have explicit
  zero-based `stop_sequence` values.
- `ValidationEvent` and `TelemetryEvent` carry event/source/entity identity,
  `event_at`, `available_at`, vehicle and route/direction/stop visit. They contain no
  passenger/card IDs. Call `row.validate_entities(catalog)` before using a row;
  parsing alone cannot establish catalog membership.
- `event.visible_at(cutoff)` requires event_at < cutoff and available_at <= cutoff.
  Feature builders must call it: a historical timestamp does not prove the row was
  available to an earlier forecast. All time comparisons are by UTC instant.
- `ObservedAggregate` permits `observed` nonnegative integer counts (including zero)
  or `missing` null. Count targets may be summed only over disjoint windows/entity
  partitions with matching provenance; occupancy and rates are not count targets.
  Coverage must come from explicit source evidence, not from a missing event row.

Malformed raw rows and duplicates remain ingestion/quarantine inputs; normalization
must not silently fix, drop or reinterpret them. Raw data stays immutable, and
corrections use a new source/transformation version.

## Publication

`forecast_v1.py` validates `forecast.v1` artifacts and dataset manifests. Each
represented route/direction/stop tuple must have exactly the full calendar horizon;
selecting a subset of entities is allowed, partial temporal coverage is rejected.
One envelope owns run/model/source versions; per-point overrides are forbidden.
Bounds must exist for every point with shared interval method/level, or be absent
for every point without such metadata. This establishes interval structure only.

Synthetic flag and `synthetic_boardings` target agree in both directions. A
validation count is not a boarding/occupancy measurement. Consumers should use
`validate_forecast_json(payload, manifest=manifest)` to enforce matching dataset,
source/feature/entity/calendar versions, target/unit and synthetic provenance,
and cutoff within the source date range. Parsing an artifact alone cannot verify
which dataset it came from. Event availability checks remain separately required.

## Verification

```bash
make reference-contract-check
```

This gate runs Ruff, strict types, calendar/data/publication tests and generated
JSON-schema drift. It is included in `make check` and backend CI. Backend and ML
also validate the same complete forecast fixture. Regenerate the snapshot only
after reviewing a contract change:

```bash
uv run --package tramflow-backend python - <<'PY'
import json
from pathlib import Path
from contracts.forecast_v1 import ForecastArtifact
Path('contracts/forecast_v1.schema.json').write_text(
    json.dumps(ForecastArtifact.model_json_schema(), indent=2, sort_keys=True) + '\n'
)
PY
```

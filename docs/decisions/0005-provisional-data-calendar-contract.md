# ADR-0005: provisional normalized data and calendar policy

Accepted for pre-data fixtures, 2026-09-23, TASK-015/025. The later organizer
target and observed raw columns are recorded in `docs/product/DATASET.md`.

## Contract

Calendar version `moscow-midnight.v1` uses Europe/Moscow and midnight service days.
Day origins are exact local hours and cover 24 elapsed hours with hourly buckets.
Month origins are the first local midnight of a month and cover that whole month
with daily buckets. Year origins are January 1 local midnight and cover the whole
calendar year with monthly buckets. Every interval is half-open. These provisional
origins simplify independent fixtures; rolling arbitrary-date month/year horizons
require a future calendar version, not silent reinterpretation.

Every forecast entity tuple must cover the entire horizon exactly once. A run may
select a subset of entities, but cannot silently omit time buckets within one.
The envelope alone owns run/model/source/version metadata; per-point overrides
are forbidden. Dataset and artifact agree on source, feature, entity, calendar,
target/unit and synthetic identity. A synthetic boarding-event count is a fixture
target, not an observed passenger occupancy. Bounds and method/nominal level are
all present or all absent in a publication. Real interval calibration is separate.

Normalized validation and telemetry events carry immutable event ID, versions,
aware event/availability timestamps and explicit synthetic provenance. They carry
no passenger/card identifiers. Visibility at cutoff requires event_at < cutoff
and available_at <= cutoff; receiving an older event later never makes it visible
in an earlier feature snapshot. Catalog IDs, not names, bind routes, directions
and ordered stop visits. Repeated stop visits require their sequence position;
duplicate display names do not collapse entities.

Observed count aggregates distinguish coverage=observed with an integer value
(including zero) from coverage=missing with null. Missing input is not evidence
of zero demand. Counts may be summed only over disjoint time windows and entity
partitions with equal target, unit and provenance. Occupancies/rates must never be
summed across times/stops as counts. A source coverage assertion is needed to
produce observed-zero buckets; the event list alone cannot establish it.

Raw adapters are not defined here: malformed rows and duplicate records remain
raw inputs for future quarantine/deduplication; strict normalized models do not
silently repair or discard them. No organizer protocol is inferred.

## Evidence, alternatives and compatibility

- https://docs.python.org/3.13/library/datetime.html: aware instants and astimezone
  conversion; naive timestamps depend on machine local timezone and are rejected.
- https://docs.python.org/3.13/library/zoneinfo.html: IANA civil-time conversion.
- Calendar contracts require month/year boundaries, leap dates, availability
  protection and distinction between missing and zero.

Fixed 30/365-day horizons would violate those cases. No distributed storage or
production contract package is selected here. Existing forecast.v1 is a pre-runtime
reference only; strengthening its invalid-input rejection corrects implementation
against its original task. Fixtures and generated schema change together; unknown
versions stay rejected. HTTP API and persisted demo forecasts are unchanged.

Recovery: revert this contract/fixtures/schema/gate together; no published datasets
or production integrations depend on the reference yet. Organizer changes require
an explicit versioned policy/adapter and new boundary cases before real-data use.

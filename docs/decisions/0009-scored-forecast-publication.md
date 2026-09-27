# ADR-0009: Scored route-hour publication

## Context

The organizer evaluates one complete route × date × hour grid for November and
December 2025. The stored route-only counts have neither observed stop nor
direction; `forecast.v1` has a different entity contract. A publisher tied to
one CSV path and one model version would mislabel a replacement forecast. The
API's previous newest-run selection could select an unrelated partial run.

## Decision

Use `tramflow.scored-route-hour.v1` as a separate manifest for this fixed
competition grid. It binds the CSV, producer entrypoint, full Git source commit,
local evaluation report, model/configuration and source versions. The publisher
validates the hashes, chronology and exact organizer grid before a PostgreSQL
transaction inserts immutable day/hour and month/day runs. A route/horizon
pointer changes in the same transaction. The API uses it when present and
retains legacy selection for the synthetic demonstrator. The Compose publisher
receives a read-only artifact directory and an approval registry baked into its
image; the web worker receives neither raw history nor training code.

The quality report names at least three complete 61-day historical windows,
separated by 45 days and ending by the cutoff. The candidate must beat its
declared baseline on at least two without regressing on any reported origin.
Reported values are not recomputed in the serving container: a reviewer must
reproduce them against the private organizer archive before adding the exact
manifest identity to the baked approval registry. A changed mounted report
cannot gain approval by recomputing its own hash. A public leaderboard number
alone does not pass this review.

## Alternatives and consequences

Embedding model metadata in backend constants was rejected because a changed
CSV would retain stale provenance. Reusing `forecast.v1` was rejected because
it requires stop/direction identity unavailable for the observed target.
Picking the latest timestamp was rejected because a partial run can be newer.

The manifest version deliberately retains the organizer's fixed 2025 grid.
Later-period inference requires an independently evaluated grid/date contract,
not a reinterpretation of this one. The active pointer is small and protects
online selection, while older published runs remain immutable for audit and
manual recovery. To roll back a bad promotion, republish the prior verified
manifest; do not mutate points or run metadata. A database downgrade after
publication removes the pointer and would change selection semantics, so use a
forward fix for deployed systems.

PostgreSQL transaction atomicity and transaction-level advisory locks are
documented at https://www.postgresql.org/docs/17/tutorial-transactions.html
and https://www.postgresql.org/docs/17/explicit-locking.html. The existing
publisher already uses an advisory transaction lock; the pointer trigger and
foreign keys provide the database-side check that cooperative locking cannot.

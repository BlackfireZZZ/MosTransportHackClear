# Full duty-calendar-clock mapping

The ready enrollment rule from TASK-080 is applied to every successful source event:
route + hashed `bus_exit_no` matching the fourth GTFS `trip_id` component + service
calendar + inclusive trip interval. Multiple matching trips abstain. Source dates and
clocks are unchanged; previous-service-day trips retain extended GTFS clocks.

The additional stop candidate is explicitly a new approximation: unique nearest
scheduled arrival within 90 seconds, unchanged under shifts of -15/+15 seconds.
Every shifted match must also stay within 90 seconds. Ties, overlapping trip clocks,
unknown duty codes and dates without active service remain unallocated. Parameters
were frozen before the three-day pilot; no clock correction is fitted. The method
has no independently measured stop accuracy and does not repair service calendars.

[GTFS reference](https://gtfs.org/documentation/schedule/reference/) defines scheduled
arrival/departure times and service days beyond midnight. It does not establish
actual vehicle position or payment location. The duty suffix interpretation is
specific to this feed. The immutable community feed was captured in June 2026,
with embedded 2025 dates; historical operation is unverified. Official service
notice exclusions are not applied in this version. Experimental training eligibility
therefore does not certify that every scheduled pattern operated.

The earlier HMM soft distribution is not used. Each input boarding contributes
exactly one unit either to a `gtfs:<stop_id>` candidate or to
`unallocated:<route>` with direction `-1`. The latter is a modeling bucket without
coordinates, not a stop. All ten competition routes remain the route-model domain;
route 5 has no successful source validations and no matching feed route.

Output includes stop-hour CSVs for January–June, July–August and September–October,
original-hour mass ledgers, unique-trip counts, a separate GTFS stop catalog,
route/date/status coverage, and checksums. No passenger or event identifiers are
exported. Stop IDs must not be joined directly to the existing OSM weather map.

```sh
PYTHONPATH=ml/src python -m tramflow_ml.boarding.duty_mapping \
 --source ../boarding-real-alignment/ml/artifacts/boarding-date-shards \
 --archive ../boarding-absolute-clock/ml/artifacts/clock-sources/moscow-gtfs-20260630.zip \
 --audit-manifest ../boarding-dataset/ml/artifacts/boarding-2025-reviewed/manifest.json \
 --out ml/artifacts/full-duty-mapping-v1
PYTHONPATH=ml/src python scripts/rebuild_duty_mapping.py ml/artifacts/full-duty-mapping-v1
```

A new directory is required. An interrupted run is incomplete and has no final
manifest; it is not a training source. The completed source partition manifests and
every input shard are checked before use. The finalizer independently checks every
route-hour sum and binds all daily outputs with SHA-256. Dataset output is local and
ignored by Git; no runtime dependency, API, database or serving behavior changes.

## Observed full result

304 days, 59,667,191 successful validations, no mass lost. 36,657,551 events have a
recognized active duty; 32,141,999 have exactly one trip, with zero trip overlaps.
21,060,982 receive the stop approximation. 38,606,209 remain unallocated:
20,741,810 without active service, 2,267,830 unknown duty, 4,515,552 outside trip
intervals, 11,081,017 without a stable nearest stop. The stability check shifts only
the nearest-stop calculation, not the unique-trip enrollment boundary.

The three split files contain 733,318 / 534,474 / 528,560 aggregate rows and retain
all 36,195,057 / 10,717,653 / 12,754,481 source boardings respectively. These include
the explicit unallocated buckets. `source_decoded` means stop-assigned in this run,
not all events consumed; `source_success` is the complete consumed source mass.

The 36 route/day unique-trip counts on the prior 12-day experiment reproduce its
results exactly. Independent finalization checked all route-hour conservation keys
and generated hashes for 912 daily outputs. Eight focused tests cover overlap,
ties, perturbation stability, midnight, invalid clocks, count conservation and empty
date rejection. Ruff and mypy pass. Parent owns the final repository quality gate.

The full artifact preserves `implementation.py` matching its manifest hash. The
final source adds only an empty-date request rejection after this completed run;
valid-date allocation is unchanged. No raw identifiers were exported.

Base: `3dc1877`, branch `agent/duty-mapping-full-20260927`, worktree
`/Users/cute/MosTransport2026Hack-worktrees/duty-mapping-full-20260927`.
Result: `ml/artifacts/full-duty-mapping-v1`. Parent integrates the isolated commit;
worktree retained for ignored dataset artifacts. No push or leaderboard submission.

## Source reproducibility supplement

`provenance.json` supplements the existing dataset manifest without changing its
bytes or invalidating downstream references. It binds the exact executed mapper
snapshot, original finalizer and every boarding module plus `uv.lock`. All dependency
bytes were verified equal to base `3dc18775199a1aee2d791ea051047306453d2fae`;
the new mapper itself is bound by the original run's implementation hash. Exact
sources are retained in `implementation-sources/`. This receipt does not claim that
the later wrapper changes were executed in the original run.

Reproduce the supplemental audit on an original run without a receipt:

```sh
PYTHONPATH=ml/src python scripts/rebuild_duty_mapping.py ml/artifacts/full-duty-mapping-v1 \
 --provenance-base 3dc18775199a1aee2d791ea051047306453d2fae
```

Future mapper runs bind the same module/lock hashes and repository revision at start
and reject source changes before completion. The source revision supplements the
hashes; it does not substitute for them when the worktree contains edits.

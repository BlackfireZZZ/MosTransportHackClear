# AFC training-data preparation — 2026-09-26

Исследование восстановления остановки посадки по временным последовательностям валидаций.

The usable target is successful validation rows in their original Moscow payment
hour. The experiment never promotes inferred stops to observed labels. Historical
patterns and independent event/visit gold are missing; G3 remains unverified.

## Reproduction

Run from the dedicated worktree with locked dependencies (`make bootstrap`):

```sh
uv run --package tramflow-ml python -m tramflow_ml.boarding audit \
  --source '/Users/cute/MosTransport2026Hack/dataset (1).zip' \
  --out ml/artifacts/boarding-2025-reviewed
uv run --package tramflow-ml python -m tramflow_ml.boarding verify \
  --run ml/artifacts/boarding-2025-reviewed
uv run --package tramflow-ml python -m tramflow_ml.boarding catalog \
  --source '/Users/cute/MosTransport2026Hack/dataset (1).zip' \
  --out ml/artifacts/boarding-2025-analysis/catalog.json
uv run --package tramflow-ml python -m tramflow_ml.boarding evaluate \
  --out ml/artifacts/boarding-2025-analysis/synthetic.json
uv run --package tramflow-ml python -m tramflow_ml.boarding detect \
  --run ml/artifacts/boarding-2025-reviewed --out ml/artifacts/boarding-2025-pilot
uv run --package tramflow-ml python -m tramflow_ml.boarding export \
  --run ml/artifacts/boarding-2025-reviewed --out ml/artifacts/boarding-2025-training
```

Completed destinations refuse overwrite; choose a new directory for a new run.
An interrupted audit resumes from a source/configuration/implementation-bound
checkpoint. Existing shards are hash-checked, and the immutable ZIP is streamed
again to reach later chunks. Event keys bind archive/member identity and row
ordinal. Two equal transaction numbers remain two events. Shards contain opaque
local device/vehicle/exit keys; card and driver fields are not read. These local
pseudonymous artifacts are not a public data release.

`train.csv`, `development.csv`, `diagnostic.csv` in the training output use fixed
2025 boundaries (July 1 and September 1). Each target hour has the preceding civil
midnight as origin; 24/168/336-hour lags and previous-day aggregates use only earlier
event hours. Missing historical cells stay blank. No zero-filled service calendar
is fabricated. `available_at` is unknown: the dataset supports retrospective
experiments, not a certified online-availability backtest. The diagnostic period
has already been inspected in earlier work and is not a blind external holdout.

Pilot burst features describe completed selected windows. They cannot be joined
to same-hour forecasting targets as if known beforehand. Recompute `detect` on
strictly pre-origin events for a forecasting experiment. Existing route-model
artifacts, serving, API and the original `tramflow-ml` CLI are unchanged.

## Interpretation and gates

- P0/P1: source ledger, archive/member and code hashes, counts, identity audit,
  minute-boundary diagnostics and route/hour conservation.
- P2: versioned catalog inventory; no certified historical pattern or independent
  2025 anchor. The supplied route7/11/12 patterns start on 2025-12-20.
- P3: device-only reversible gap/max-span detectors, B0, anchored B2 and exact
  small monotone reference decoder. Open starts, missing visits, repeated stop IDs,
  split/many-to-one events, delay/clock scenarios and ambiguity are tested.
- P4/P5: four predeclared D0 configurations and two synthetic generator families.
  Real stop accuracy, calibration and external precision remain unverified.
  HMM/HSMM/beam adoption is deferred by G1/G3/G4; no evidence supports training them
  against guessed stop labels. Exact decoding has explicit budgets and returns
  undecoded on exhaustion; it is not represented as a scalable full-period solver.
- P6: conserved route-hour training export plus unassigned source ledger and
  descriptive pilot artifacts. G5 application publication is outside this task.
- P7: optional route-model burst-feature ablation is not adopted; existing model
  stays intact. Route forecast improvements would not validate stop geography.

The [assumption register](assumptions.json) and [independent-data request](missing-data-request.md)
state what would unlock stop-label training. There are no new dependencies.

## Research evidence

[Chen and Fan](https://d-nb.info/1164363093/34) motivate temporal clustering but
cannot establish our start stop or thresholds. [Newson and Krumm](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/)
justify topology-aware sequence constraints; their GPS emissions and accuracy do
not transfer to timestamp-only observations. [Johnson and Willsky](https://www.jmlr.org/papers/v14/johnson13a.html)
explain explicit duration models, which do not create independent anchors.
[Killick et al.](https://arxiv.org/abs/1101.1438) describe penalized changepoints,
not a way to establish stop geography. The implementation remains a reversible
stdlib/installed-pandas offline experiment without a production ADR or dependency.

## Observed result

The full audit reconciled **59,667,191 successful events** from 62,443,497 raw rows
against all 57,551 label keys with zero mismatches. Two complete scans produced
byte-identical normalized shard hashes. The reviewed pass took 489.54 seconds and
576,454,656 bytes peak RSS. No successful payment was deduplicated or dropped.

The exported dataset has 57,594 source-observed route/hour rows:
34,521 train, 11,742 development, 11,331 diagnostic. Its 43 extra cells relative to
labels contain rejected source rows and zero successes; they are not invented
calendar cells. Target counts sum back to the full successful source mass.

Pilot routes 1/12 cover 12 days and 449,635 rows (427,494 successful). The default
D0 scenario produces 265,804 bursts. Four configurations conserve all events;
9 perturbations run on a deterministic 10,000-event sample. The 24 synthetic cases
cover two generator families. Real stop accuracy stays unverified.

`make check` exit 0: 350 backend passed / 48 SQL skipped, 353 ML, 92 frontend,
119 contract tests; Ruff, mypy, ML golden evaluation, build, OpenAPI and Compose
checks passed. There are 54 new boarding tests. No production dependency, schema
or API changed; existing model artifacts were hash-checked without modification.

See [evidence.json](evidence.json), [synthetic results](synthetic.json),
[stability results](stability.json) and [catalog inventory](catalog.json).
The remaining stop-level work is explicitly blocked on independent data, not marked
complete. The local worktree is retained for ignored artifacts and its unpushed branch.

Downloadable artifact location:
`/Users/cute/MosTransport2026Hack-worktrees/boarding-dataset/ml/artifacts/boarding-training-2025.zip`.
Code/report integration: local fast-forward into main after parent verification; no push.
Base: `181a73bfd6a3b4e361229e148cd2decc2b9e922f`; branch: `agent/boarding-dataset`.

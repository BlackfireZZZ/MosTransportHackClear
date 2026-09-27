# Inference readiness and risk map

Evidence: organizer [target and grid](../product/DATASET.md), scored [model
record](../experiments/route-hour-2025.md), checksum-bound
[artifact](../../ml/competition_submissions/2026-09-27/README.md), backend SQL
integration and disposable Compose smoke. The 0.89195 public WAPE-score is a
user-reported leaderboard result for **November–December 2025**, not a measured
online, stop-level or later-year accuracy claim.

| Priority | Observed failure window | Control | Residual limit |
|---|---|---|---|
| P0 | The best model produced a CSV but the web service read only seeded synthetic SQL rows. | Compose publishes the selected checksum-bound bundle after migration and before API startup. | The serving contract covers the fixed scored 2025 horizon; new dates need a separate evaluated contract. |
| P0 | Only two demo route IDs existed; scored numbers are ten distinct route labels. | Publisher validates all 14,640 keys and resolves route **numbers** to serving surrogate IDs. | No observed stop or direction identity is attached. |
| P1 | Missing, duplicate, nonfinite or altered CSV cells could be presented as complete forecasts. | SHA-256 verification of the bytes being parsed, strict header/grid/count/prediction validation before a single SQL transaction. | The hash proves artifact identity, not hidden-label accuracy. |
| P1 | A newer partial run could hide the selected scored forecast. | Publication switches explicit route/horizon pointers in the same transaction; a trigger and pointer-free fallback both require a complete published scored run. | Other future publishers need their own completeness and selection contract. |
| P1 | Replacing only the CSV would silently retain the old model/version labels. | A versioned manifest binds CSV, producer file, local temporal evaluation and provenance by SHA-256; its canonical identity creates distinct immutable run IDs. Publication requires its identity in the image-baked approval registry. | The report's numbers require independent offline reproduction before approval; hashes alone cannot prove accuracy. |
| P1 | Aggregate quality gains could conceal a severe loss on one historical forecast origin. | The bundle gate requires gains on at least two complete 61-day origins and forbids a loss on any reported origin. The current report's three score pairs and selected CSV were reproduced from the organizer archive. | A reviewer must still check that future reports include representative origins and faithfully reproduce private-label scores. |
| P1 | Published real predictions could be changed under the same run ID. | New migration rejects insert/update/delete of points belonging to a published `validation_count` run and forbids demoting that run. | Older synthetic demo points retain legacy mutability for existing test fixtures. |
| P1 | Private ensemble proof and alignment assertions vanish with Python `-O`; recency script lacked proof verification. | Checked-in scripts use explicit, archive-bound reconciliation and alignment errors; verified re-generation reproduces the committed CSV hash. | The source raw timestamp timezone is not specified by the organizer. |
| P1 | Historical actual weather could enter a nominal pre-origin submission. | `submit` rejects retrospective weather input; offline `evaluate` remains explicitly retrospective. | Weather corrections are not part of the selected model. |
| P2 | An hourly grid would appear as 31 days in the month tab or leak December 1 into the default November view. | Publisher derives daily route sums; default month window ends at the next Moscow calendar month boundary. | Explicit windows may cross months by design. |
| P2 | The UI could imply stop loads, occupancy, capacity or intervals from route-only counts. | Published run declares `validation_count/event_count`; stop rows, capacity and bounds are absent; map matching remains unavailable. | Frontend wording and presentation of unavailable stop detail remain a separate UI responsibility. |

Publication is idempotent for the same approved manifest identity. A failed
validation or SQL insertion leaves no published run. Correcting a published
forecast requires a new checksum and publication; changing rows in place is
rejected. On a fresh Compose stack, numeric routes appear first in the catalog,
with the two synthetic demo routes still separately available. A route without
observed successful validations in the supplied history, such as route 5,
receives the model's zero counts; this does not assert that the route did not
operate.

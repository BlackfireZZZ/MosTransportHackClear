# Anchored mapping visualization

Open `index.html` in a browser. D3 7.9.0 loads from jsDelivr; an internet connection is required. `fragment.html` is the editable conversation source; `data.json` contains the same embedded aggregates.

Nine deterministic examples cover routes 11/12/17 on January 14, May 13 and September 9, 2025. Each is the first eligible morning trip (06:00–10:00, at least 100 validations), ordered by first validation, profile and anonymous vehicle group; selection does not use mapping coverage or fit. Times use Europe/Moscow. No passenger, card, device or vehicle identifiers are exported.

Assignments are inferred using `duty-calendar-clock.first-stop-local-anchors-transfer.v3`, not observed boarding labels. Transferred calendars remain visibly marked. The monthly totals come from `../full-audit.json`. This historical visualization does not replace the serving forecast.

Publication uses retained worktree `publish-mapping-viz-20260927`, branch `agent/publish-mapping-viz-20260927`, based on `ba8fc1404e77e9106b09e891a81bad8335dc1033`; remote main integrated through `222321d273db3f67dc29c7387b77873ed1812237`. Only documentation artifacts are added. Verification: browser route/date/stop selection, legend toggle and tooltip; desktop/mobile rendering and aggregate conservation. Worktree retained as the publication receipt.

Publication checks passed: `make check` exit 0; browser interaction checks with no runtime errors and mobile width 328/328 px; conservation and embedded-data equality for all 9 examples. The standalone page and fragment contain identical aggregates.

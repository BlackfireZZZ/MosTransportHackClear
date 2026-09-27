# KudaGo events pilot: what two probe months of 2025 actually contain

Run of 2026-09-27 over January and September 2025, Moscow city listings, under the
[`external-events.v1` contract](../../external-events-kudago.md). Collector:
`tramflow-kudago`.

`quality_report.json` and `manifest.json` in this directory are the run's own
artefacts, copied verbatim. The delivery itself is 125 MB and stays out of Git; it
is reproducible from the command in "Reproducing" below.

## The answer

**The data is retrospective-only and cannot enter the scored route-hour pipeline.**

Availability, not quality, is the blocker. Every snapshot was fetched in 2026, and
no archived copy of these pages exists — the Wayback Machine holds no KudaGo event
pages. So `available_at` is 2026 for every observation, and at each 2025 origin:

| Origin | Sessions provably available | Retrospective only |
|---|---|---|
| 2025-05-01 | 0 | 4047 |
| 2025-07-01 | 0 | 4047 |
| 2025-09-01 | 0 | 4047 |
| 2025-11-01 | 0 | 4047 |

A `publication_date` before the origin proves nothing on its own: the pilot found a
January 2025 card published in April 2026. Using this data in the competition
backtest would show an improvement that cannot reproduce, because at scoring time
the model would not have had the rows it trained against.

The path to usable data is daily snapshots accumulating from now on, not a deeper
crawl of the past.

## What the catalogue does contain

Where availability is not the question — for example descriptive work, or any use
after snapshots accumulate — the content itself is reasonable:

| Measure | Share | Numerator / denominator |
|---|---|---|
| Sessions with an exact start | 86.6% | 3504 / 4047 |
| Sessions with an exact interval | 33.9% | 1373 / 4047 |
| Sessions with a place | 83.1% | 3364 / 4047 |
| Sessions with coordinates | 73.2% | 2962 / 4047 |
| Unresolved | 13.4% | 543 / 4047 |
| Date only | 0.0% | 0 / 4047 |

Counts: 1960 event cards, 312 venues, 2272 observations, 78668 occurrences of which
4047 fall inside the two collection windows, 3 quarantined, 2328 requests.

2131 sessions give a start but no trustworthy end (`start_only`) — 52.7% of all
in-window sessions, and 60.8% of those that carry an exact start at all. An hourly
occupancy feature would therefore rest on an assumed duration for most sessions;
only 1373 carry both ends.

Resolved sessions by month, buffer days included:

| 2024-12 | 2025-01 | 2025-02 | 2025-08 | 2025-09 | 2025-10 |
|---|---|---|---|---|---|
| 90 | 2124 | 107 | 42 | 1115 | 26 |

## Reading the numbers correctly

- **Coverage is measured inside the window.** 74621 of 78668 occurrences fall
  outside it: one card can carry a decade of history, and those rows are correct
  data kept by design. Counting them in the denominator reported 1% exact intervals
  for data that was 71% on an earlier sample.
- **The 543 unresolved rows are kept, not dropped.** They are date records the
  normaliser refused to guess at, by flag: `schedule_weekdays_unconfirmed` 202 (a
  weekday set with no documented first day), `ambiguous_multiday_range` 133,
  `is_startless`/`unbounded_start` 107, `place_schedule_required` 94 (a venue
  timetable it would have to assume), `start_day_unknown` 49, `time_conflict` 38.
  Rule 8 keeps one row each so an unknown date never disappears.
- **`by_month` counts resolved sessions only.** An unresolved row keeps whatever
  date its card carried, which can be 2014; bucketing by the raw `start_date` put
  sessions years outside the window into a report about two months of 2025.
- **The run is `partial`, honestly.** Six of 2328 requests failed: three read
  timeouts, two HTTP 503, one HTTP 404. The manifest lists them rather than
  presenting the delivery as complete.
- **This is not the whole city.** Closed events, deleted cards and unregistered
  venues may be absent, and the share of real events covered is unknown without an
  external registry. No coverage percentage here is a claim about reality.

## Reproducing

```bash
uv run --package tramflow-ml tramflow-kudago collect \
  --month 2025-01 --month 2025-09 --out <dir> --owner <you>
```

About 50 minutes at the collector's one-request-per-second policy; the API itself
serves list pages in 2–18 s. Re-running the normaliser alone over an existing
capture is `--no-fetch`, and the same `raw/` yields byte-identical output.

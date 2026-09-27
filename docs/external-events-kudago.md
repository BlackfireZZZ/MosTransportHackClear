# External events: KudaGo

Contract `external-events.v1`. Collector: `ml/src/tramflow_ml/external/kudago/`,
CLI `tramflow-kudago`. [Пилотный отчёт](analysis/2026-09-27-kudago-events-pilot/README.md)
фиксирует покрытие и ограничения исторических данных.

Moscow city listings from the [KudaGo public API](https://docs.kudago.com/api/)
`v1.4`, collected offline. The online worker never calls it.

## What a delivery is

```text
manifest.json          written last, hashes every other file, never itself
requests.jsonl         one row per attempt, successful or not
raw/<request_id>.json  response bodies, byte for byte
dictionaries/          locations and event categories as fetched
events.jsonl           one row per content version of a card
places.jsonl           one row per content version of a venue
occurrences.jsonl      one row per resolved session, or per unresolved date record
observations.jsonl     one row per sighting, even when the payload repeats
quarantine.jsonl       what could not be used, and why
quality_report.json    coverage with numerators and denominators
```

UTF-8 without BOM, `null` for unknown, no NaN or Infinity, no empty string
standing in for a number, one JSON object per physical line. Source ids stay
strings: `event_id="3606"`. A KudaGo `place_id` shares no namespace with a
payment `place_id` or a GTFS/OSM stop id.

`snapshot_id` is SHA-256 over `json.dumps(payload, ensure_ascii=False,
allow_nan=False, sort_keys=True, separators=(',', ':'))`. Arrays keep their order
and numbers keep their type — re-sorting would make two genuinely different
versions of a card collide.

## The four timestamps

Confusing these is how event data leaks into a backtest.

| | |
|---|---|
| `published_at` | what the API claims about publication. Not evidence of anything since. |
| `fetched_at` | when we received the response. |
| `available_at` | when **this version of the fields we use** was provably available. |
| `start_at` / `end_at` | when the event runs. May be after the forecast origin. |

An ordinary snapshot gets `available_at = fetched_at`. Only an archived snapshot
containing the schedule, place and status actually used may set it earlier, and
then `availability_evidence` is mandatory. A `publication_date` before the origin
proves nothing on its own — the pilot found a January 2025 card published in
April 2026.

For an origin, take the last observation with `available_at <= origin`, then its
snapshot. Do not sort unique content hashes by first appearance: that erases a
card going A → B → A and silently reinstates a cancelled postponement.

A future event is legitimate. Unlike validation data, `start_at < origin` is **not**
required — an announcement made before the origin may describe a concert after it.
That is why `features/events.py` and the `data.v1` contract cannot be reused here.

## Sessions, not events

A card with ten sessions is not one ten-week event. `occurrences.jsonl` carries one
row per session, or one row for a date record that could not be resolved.

Eight rules govern the conversion; each exists because the obvious reading is
wrong. They are implemented in `timerules.py` and every one has a named test.

1. Unix `start`/`end` are seconds UTC. If they disagree with the local fields,
   the result is `unresolved` with `time_conflict` — not a silent choice.
2. `end == start` means unknown duration: `end_at=null`, `start_only`. Never a
   zero-length session.
3. A missing `end_date` with an `end_time` takes its day from a valid Unix end, or
   stays `start_only`. The day is never invented.
4. A date with no time is `date_only` with both moments null. Never 00:00.
5. A month-long range without hours does not become 24 hours a day. Unbounded
   repeats expand only inside a collected month plus its buffer. A record entirely
   outside every window is one row flagged `outside_collection_window` — correct
   data, not quarantine, and excluded from target-session counts.
6. Repeats expand only when the schedule is unambiguous. `days_of_week` has no
   documented first day, so a partial weekday set is refused rather than guessed.
7. Intervals are half-open `[start_at, end_at)`. An event ending at 20:00 does not
   occupy the 20:00–21:00 bucket. `start_only` feeds start features only.
8. An unknown date never disappears. A negative interval is quarantine.

## Traps confirmed against the live API

- **A projection is not a record.** The same entity arrives three ways: a bare
  `{"id": 2033}` reference on a list page, an object inlined by `expand=place`,
  and the full `places/{id}/` response. Only the last is a venue. Classify by the
  request that produced the object, never by its content — the inventory shape
  carries no `title` by construction, so checking required fields on it quarantines
  perfectly good cards.
- **Coordinates come from `place.coords` only.** `location.coords` is the city
  centre and must never substitute. Validate the ranges; a suspicious pair is
  flagged and nulled, never auto-transposed.
- **Overlapping windows are not drift.** A one-day overlap changes `count`
  legitimately. Compare id sets across passes before calling a catalogue unstable.
- **Coverage is measured inside the window.** One card can carry a decade of
  history; counting those rows in the denominator reported 1% exact intervals for
  data that was 71%.
- **Disjoint months are disjoint windows.** A run collecting January and September
  holds two spans, never the hull between them. Spanning `[Jan, Oct)` would expand
  repeats into eight months nobody fetched and put them in the coverage denominator,
  which reads as collected data. `manifest.json` therefore carries `targets` and
  `windows` as one bound pair per month, and touching spans are merged so no session
  is counted twice.

## What the pilot established

Availability, not quality, is the blocker. Sessions inside the window carry
coordinates and exact starts, but every snapshot was fetched in 2026 and no
archived copy exists — checked, the Wayback Machine holds no KudaGo event pages.
So **no session passes a 2025 origin cutoff**, and this data is retrospective-only
until daily snapshots accumulate. Using it in the competition backtest would show
an improvement that cannot reproduce.

## Scope

Not included: cinema sessions (`movie-showings/`), the tram crosswalk, the feature
join, audience-size enrichment. The API catalogue is not the whole city; closed
events, deleted cards and unregistered venues may be absent, and the share of real
events covered is unknown without an external registry. Never state a coverage
percentage against reality.

Large dumps stay out of Git. The repository holds this contract, the collector,
compact fixtures and the quality report.

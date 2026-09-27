# Historical Moscow transport data — 2026-09-27

## Result and scope

**No continuous, freely downloadable Moscow road-speed/congestion archive for January–October 2025 was verified.** Downloaded and retained the available public observations, with their limitations:

| Dataset | Retained data | Actual Moscow-time coverage |
|---|---|---|
| `data/external-traffic/moscow-agency-2025/` | 14 current citywide Yandex congestion reports; one also has CODD speed | March 6–October 30, 12 publication dates |
| `data/external-traffic/m24-2025/` | 5 manually verified news reports, CODD/Yandex separated; one CODD speed | February 24–September 25, 4 publication dates |
| `data/external-parking/2025/` | 1,642,514 API parking observations, 178 lots, 9,691 snapshot instants | April 1 02:22–October 31 23:42, 208 days |

The 19 news reports cover 14 distinct dates and add only **one publication date (February 24)** beyond the existing 163 Telegram observations across 106 dates. The publishers can repeat the same underlying reading; these are not 19 independent sensors. Do not present this extension as a dense new traffic dataset.

Parking is a separate mobility proxy, **not congestion, speed, vehicle flow or tram occupancy**. The new artifacts are not consumed by the published model or experimental source switches. No forecasting improvement is claimed. Raw organizer validations and the scored submission were not changed.

## Evidence and sources

Evidence level: inspected publisher HTML / dataset card and Parquet schema, with repository cutoff contract (`docs/product/DATASET.md`). Source values are reported measurements, not independently calibrated sensor ground truth.

- [Moscow Agency public search](https://www.mskagency.ru/search/?criteria=%D0%BF%D1%80%D0%BE%D0%B1%D0%BA%D0%B8&from=01.01.2025&to=31.10.2025&type=text): bounded January–October searches for `пробки`, `загруженность`, `средняя скорость`. Thirteen search pages, sixteen article candidates. Exact source URLs and SHA-256 receipts in the manifest. Strict current-measurement parsing rejects forecasts, ranges, local geography and mismatched title/lead. Nested article/photo HTML is handled by a parser rather than the first closing div.
- [Moscow24 public search](https://www.m24.ru/sphinx/?criteria=%D0%BF%D1%80%D0%BE%D0%B1%D0%BA%D0%B8): five article bodies independently read and manually curated. Each JSONL row records the exact URL, HTML SHA-256, publisher timestamp and provider. This is a reviewed selection, not exhaustive automated extraction. One article separately reports CODD 7 and Yandex 8; retained headline CODD 7. Another publishes at 20:21 while mentioning road conditions at 19:25; that caveat stays on the row.
- [Moscow Parking Occupancy](https://huggingface.co/datasets/matrosovdani/moscow-parking-occupancy), pinned revision `283c1a33466424b48aef042923c1f8e6bdff3d98`, CC BY 4.0. Attribution: **Danil Matrosov / ParkOut; original Moscow Department of Transport public parking data**. Original dataset card is `data/external-parking/2025/SOURCE_README.md`.

Other inspected leads:

| Source | Why it does not supply the requested archive here |
|---|---|
| [TrafficIndex Moscow](https://trafficindex.org/moscow/) | Advertises 2017–2026 history, explicitly Premium access; no bypass attempted |
| [TomTom Traffic Stats](https://docs.tomtom.com/traffic-stats/documentation/api/introduction) | Requires MOVE API credentials, not available locally |
| [YMArchive](https://sourceforge.net/projects/ymarchive/files/) | Public downloadable archive is from 2015, wrong period |
| [MeTS-10](https://github.com/iarai/MeTS-10) | Genuine segment speed data including Moscow, 2019–2021, wrong period |
| [UrbanTransportData](https://urbantransportdata.ru/) | No open route-hour download verified; platform participation/access differs from a public historical sensor archive |
| [probki-moskva.ru](https://probki-moskva.ru/) | A 2025 dissertation cites it with access date in 2022; current fetch failed and 2025 coverage remains unverified |
| [Transport portal](https://transport.mos.ru/traffic_situation) | Current/forecast traffic page; direct retrieval timed out, historical endpoint unverified |

## Data contract and traps

News timestamps are publication proxies. No row proves the original unedited historical version. Keep Yandex and CODD scores separate, and retain a separate provider for speeds. Missing reports do not imply free roads. Article text/images are not redistributed; the extracted facts have no separately established open-data licence.

Parking source files partition timestamps in UTC. Filtering uses `Europe/Moscow` and the half-open interval `[2025-01-01, 2025-11-01)`: **1,056 source rows were already November 1 in Moscow and were excluded**. The first retained measurement is April 1 in Moscow, despite the source's March 31 UTC start. There is no January–March coverage; April–October contains six fully missing dates and other gaps.

All values are retained unchanged, with quality flags: **14,444 occupancy rates outside [0,100]**, no negative counts, and six lots whose free-space counts are always zero throughout the retained history. Such zeros can be inactive feeds. No imputation, clipping, sensor-reliability inference or hourly synthetic reconstruction is performed. `free_spaces` includes accessible spaces; `occupancy_rate` refers to common-space capacity and is not `1-free_spaces/total_spaces`.

Latest metadata contains future `last_free_at`/`feed_silent` values from 2026. The prepared location file contains only identifiers, names and coordinates; capacity and future flags are excluded. Location metadata is still a retrospective snapshot, not a proven 2025 vintage. Upstream occupancy-rate capacity history is not independently established. Any future feature must use only observations before its own forecast origin and derive sensor support from that history.

## Reproduction and validation

No project dependency or lock file changed. Collector uses the Python standard library. Parking preparation used already installed pandas 2.3.2 / pyarrow 19.0.1 in `/Users/cute/miniconda3/bin/python`; this is an offline data preparation environment, not a new serving dependency.

```sh
python3 scripts/test_moscow_traffic_archive.py
python3 scripts/collect_moscow_traffic_archive.py \
  --cache /tmp/moscow-agency-traffic-2025 \
  --output data/external-traffic/moscow-agency-2025 --offline
```

Remove `--offline` to fetch missing public pages at a bounded rate (at least 0.5 seconds between requests). A missing page or changed article structure raises an error. Original source snapshots remain immutable outside Git in `/tmp/moscow-agency-traffic-2025`; future public edits may change source receipts.

```sh
hf download matrosovdani/moscow-parking-occupancy \
  README.md data/parking_spots.parquet \
  'data/occupancy_2025-0[3456789].parquet' data/occupancy_2025-10.parquet \
  --repo-type dataset --revision 283c1a33466424b48aef042923c1f8e6bdff3d98 \
  --local-dir /tmp/moscow-parking-source --max-workers 2
/Users/cute/miniconda3/bin/python scripts/prepare_parking_history.py \
  --source /tmp/moscow-parking-source --output data/external-parking/2025
```

Original source partitions are outside Git in `/tmp/moscow-parking-source`. All source and output hashes are in the parking manifest. The script checks source file inventory, UTC schema, nonempty primary keys, key uniqueness, location FK membership, date bounds, and Parquet read-back equality. Separate repeated runs produced byte-identical traffic JSONL/CSV and parking observations Parquet. Nine focused parser tests passed, including future tense, geography, ranges, conflicting measurements, nested markup, and cutoff boundaries. Independent reviewer verified all candidate agency articles and the five Moscow24 articles.

Full `make check` exited 0: 404 backend tests passed (52 SQL integration tests skipped by the standard fast gate), 1,483 ML tests passed, 108 frontend tests passed, 119 contract tests passed; lint, types, golden ML eval, production build, OpenAPI drift and Compose validation passed. Log: `/tmp/traffic-archive-make-check.log`. Focused collector Ruff check and `git diff --check` passed. Independent parking audit confirmed exact equality to all retained source rows and output hashes.

## Handoff

Owner: Codex lead; supporting agent performed read-only research and review. Base `ba8fc1404e77e9106b09e891a81bad8335dc1033`; branch `agent/traffic-archive-20260927`; worktree `/Users/cute/MosTransport2026Hack-worktrees/traffic-archive-20260927`.

Acceptance is achieved for the available-data collection; the requested continuous speed archive remains unavailable from inspected open sources. Next step for model use is a separately specified chronological ablation of parking history with source/support controls; this task does not promote a model. Worktree retained because the branch is unpushed. Integration: local fast-forward into primary `main` after the successful quality gate; no external publication. Existing untracked organizer ZIP, macOS files and unrelated algorithm-research plan are preserved.

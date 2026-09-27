---
license: cc-by-4.0
language:
  - en
  - ru
task_categories:
  - time-series-forecasting
  - tabular-regression
tags:
  - parking
  - smart-city
  - urban-mobility
  - moscow
  - occupancy
  - open-data
size_categories:
  - 1M<n<10M
configs:
  - config_name: occupancy
    data_files: data/occupancy_*.parquet
  - config_name: parking_spots
    data_files: data/parking_spots.parquet
---

# Moscow Parking Occupancy Dataset

4.6 million half-hourly occupancy snapshots for **210 municipal parking lots**
in Moscow, continuous since **31 March 2025** and refreshed every Monday.

Long, dense, public occupancy series are rare: most published work on parking
occupancy prediction still leans on the UCI *Parking Birmingham* set, which
covers October–December 2016 only.

## Configs

| Config | Rows | What it is |
|---|---|---|
| `occupancy` | ~4.6 M | Snapshots every 30 minutes |
| `parking_spots` | 210 | Static metadata: coordinates, capacity, nearest metro |

```python
from datasets import load_dataset

occ = load_dataset("matrosovcmtn/moscow-parking-occupancy", "occupancy", split="train")
spots = load_dataset("matrosovcmtn/moscow-parking-occupancy", "parking_spots", split="train")
```

## Schema

**`occupancy`** — primary key `(time, parking_id)`

| Column | Type | Description |
|---|---|---|
| `time` | timestamp[ns, UTC] | Snapshot timestamp |
| `parking_id` | int32 | Foreign key to `parking_spots.id` |
| `free_spaces` | int32 | Free spaces (common + accessible) |
| `free_handicapped_spaces` | int32 | Free accessible spaces |
| `occupancy_rate` | float64 | Percent 0–100 of **common** spaces occupied |

> `occupancy_rate` is computed as `(common_total - common_free) / common_total * 100`
> over declared **common** capacity only — it is *not* `1 - free_spaces / total_spaces`.

**`parking_spots`** — `id`, `external_id`, `name_ru`, `name_en`,
`address_street_ru`, `address_street_en`, `subway_ru`, `subway_en`,
`latitude`, `longitude`, `total_spaces`, `common_spaces`, `handicapped_spaces`,
`last_free_at`, `feed_silent`.

## Read this before modelling: 39 lots are not what they look like

**19% of the Moscow lots report a constant zero free spaces.** That looks like
"completely full". It is not — the city registered these lots but does not
publish their occupancy, or the sensor died.

Measured on 2026-09-20: 39 of 210 lots reported no free space at all in the
previous 30 days; 19 never reported one in the entire history. One 257-space
lot last showed a free space in May 2026, a 143-space lot at VDNKh in July
2025. A 606-space lot does not stay full for five months.

Kept in the data on purpose, but flagged:

```python
spots = spots.to_pandas()
live = set(spots.loc[~spots.feed_silent, "id"])
occ = occ.filter(lambda r: r["parking_id"] in live)   # drops ~4.8% of rows
```

`last_free_at` lets you rebuild the flag for any as-of date instead of
trusting ours.

**Separately, the data is censored at the boundaries:** 25.2% of all rows are
exactly 100 and 7.8% exactly 0 — mostly genuine, lots really do fill up.
Treat the target as bounded.

## Caveats

- **Municipal lots only.** Malls, private lots and unpaid curbside are not in
  the Moscow public API and are not here.
- **Reported, not observed.** Values come from the upstream API; occasional
  zero-spikes and stale values are preserved rather than cleaned.
- **Gaps exist** during upstream outages.
- **No PII.** Aggregate counts only — no vehicle, plate or driver data.

## Licence and citation

Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original values
come from Moscow Department of Transport public parking data; this is a
derivative work that aggregates and republishes them. Keep attribution to both.

```bibtex
@misc{matrosov2026moscowparking,
  author       = {Matrosov, Danil},
  title        = {Moscow Parking Occupancy Dataset},
  year         = {2026},
  howpublished = {\url{https://github.com/matrosovcmtn/moscow-parking-occupancy}}
}
```

Canonical source and full documentation:
[github.com/matrosovcmtn/moscow-parking-occupancy](https://github.com/matrosovcmtn/moscow-parking-occupancy)
· Maintained by [ParkOut](https://parkout.ru)

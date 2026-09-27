# Independent stop forecast asset

`forecast_mode=stop_model` serves the independent
`direct-stop-calendar-history.anchored-v3:calendar_weekday_blend` model. It is not a
CatBoost model and does not inherit the approved route score. Its targets
are inferred stop counts plus explicit unknown buckets, not observed boarding
stops, tram positions, unique passengers or cabin occupancy.

The artifact has 584 identities and 854,976 exact route/stop/direction/date/hour
values. Each identity contains 1,464 continuous predictions in day-major,
hour-minor order beginning 2025-11-01. No rounding, averaging, normalization to
another route model or conservation adjustment is applied. Names and coordinates
come from the separately hash-bound planning catalogue. Unknown identities have
no map point and remain in route totals.

Day exposes exact hours; month sums them. Dates beyond December in the optional
year view use each identity's mean weekday/hour profile of the two forecast
months. Those later values are qualitative, not the original model's forecasts.
All source-selection flags leave this mode unchanged because independently
trained stop-model source ablations do not exist. The API reports that explicitly.
Four manual factors remain available and multiply exact baseline counts.

Rebuild from the retained source directory (adjacent `run.json` and
`dataset_audit.json` are required):

```sh
python3 scripts/build_direct_stop_artifact.py --source /path/to/stop_predictions.csv.gz
```

The generator validates unique complete grids, nonnegative finite values, source
model identity and the catalogue checksum. Manifest binds source CSV, source
run/audit, producer and generator. Source checksum:
`1e79f596a6723de6b80dcdbe3915fa66561e0466d6299da8a179c760e7154bed`.
Every one of the 854,976 serialized values was independently compared with the
source CSV and matched exactly. No private passenger records are included.

Serving validates hash, schema, full identity set, cutoff, coordinate catalogue
binding and every array. Corrupt/absent assets return 503; invalid filters return
422. Deployment includes this asset through the existing `COPY data` instruction;
no database, dependency, SQL migration or existing forecast mode changes.

## Anchored mapping refresh (2026-09-27)

The same fixed estimator now uses `duty-calendar-clock.first-stop-local-anchors-transfer.v3`.
Forecast unknown share fell from 72.2112% to 22.3773%; historical mapping coverage
is separately 80.28%. First validation is assumed to be the first stop; strong
onsets correct subsequent timetable clocks; remaining events use previous-stop
intervals. Some history uses transferred calendars from other dates. GTFS geometry
and calendars are retrospective; no stop ground truth or calibrated intervals exist.

Both planning spatial shares and independent stop assets bind the new catalog.
All 574 physical identities, names and coordinates are unchanged; only first_date
and row ordering differ. Scored route rows and the competition CSV are unchanged.
See [refresh evidence](../../docs/analysis/2026-09-27-anchored-stop-refresh/README.md)
for temporal comparisons, quantization effects and the reproduction command.

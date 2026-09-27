"""Duty/calendar/clock candidates; unmatched mass is never forcibly assigned."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, write_json
from .clock_gtfs import profiles_for_day
from .direction_probe import duty_component
from .partitions import read_day, write_frame

ROUTES = ('1', '5', '7', '11', '12', '17', '25', '26', '28', '50')
FEED_SHA = 'dec3ec0ccc2efb7f1327ee6f26a0636a4a271494490e6debedcb2c1ccc579e76'
FIELDS = ['day', 'route', 'event_hour', 'stop_id', 'direction', 'expected_count',
          'applicable_pattern', 'hour', 'weekday', 'label_origin', 'training_eligible']
METHOD = 'duty-calendar-clock.nearest90-stable15.v1'


def extract_feed(path: Path) -> dict[str, Any]:
    if digest(path) != FEED_SHA:
        raise ValueError('unrecognized GTFS archive')
    with ZipFile(path) as archive:
        def rows(name: str) -> Any:
            with archive.open(name) as raw:
                yield from csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig'))
        routes = {r['route_id']: r for r in rows('routes.txt')
                  if r['route_type'] == '0' and r['route_short_name'] in ROUTES}
        trips = {r['trip_id']: {**r, 'stop_times': []} for r in rows('trips.txt')
                 if r['route_id'] in routes}
        for row in rows('stop_times.txt'):
            if row['trip_id'] in trips:
                trips[row['trip_id']]['stop_times'].append(row)
        ids = {r['stop_id'] for t in trips.values() for r in t['stop_times']}
        services = {t['service_id'] for t in trips.values()}
        return {'routes': list(routes.values()), 'trips': list(trips.values()),
                'calendar': [r for r in rows('calendar.txt') if r['service_id'] in services],
                'calendar_dates': list(rows('calendar_dates.txt'))
                if 'calendar_dates.txt' in archive.namelist() else [],
                'stops': {r['stop_id']: r for r in rows('stops.txt') if r['stop_id'] in ids}}


def nearest_stable(times: Any, schedule: Any) -> Any:
    """Unique closest arrival within 90s, same position at ±15s; not calibrated."""
    schedule = np.asarray(schedule, dtype=float)
    times = np.asarray(times, dtype=float)
    if not len(schedule) or not np.isfinite(schedule).all() or np.any(np.diff(schedule) < 0):
        raise ValueError('finite ordered stop clocks required')
    if not np.isfinite(times).all():
        raise ValueError('finite event clocks required')
    selected = np.full(len(times), -1, dtype=int)
    valid = np.ones(len(times), dtype=bool)
    for shift in (0, -15, 15):
        costs = np.abs(times[:, None] + shift - schedule[None, :])
        ix = costs.argmin(axis=1)
        best = costs[np.arange(len(times)), ix]
        unique = (costs == best[:, None]).sum(axis=1) == 1
        if shift == 0:
            selected = ix
        valid &= unique & (ix == selected) & (best <= 90)
    return np.where(valid, selected, -1)


def enroll(times: Any, profiles: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Audited inclusive trip interval rule; every overlap abstains."""
    times = np.asarray(times, dtype=float)
    count = np.zeros(len(times), dtype=int)
    selected = np.full(len(times), -1, dtype=int)
    for index, profile in enumerate(profiles):
        inside = (times >= profile['times'][0]) & (times <= profile['times'][-1])
        count += inside
        selected[inside] = index
    return count, np.where(count == 1, selected, -1)


def map_day(source: Path, feed: dict[str, Any], day: str, salt: str) -> tuple[Any, Any, Any]:
    frame = read_day(source, day)
    frame = frame.loc[frame.success.eq('1')].copy()
    stamps = pd.to_datetime(frame.event_at)
    frame['second'] = stamps.dt.hour * 3600 + stamps.dt.minute * 60 + stamps.dt.second + 86400
    frame['hour'] = stamps.dt.hour
    frame['status'] = 'no_active_feed'
    frame['stop_id'] = 'unallocated:' + frame.route
    frame['direction'] = '-1'
    frame['trip'] = ''
    frame['unique_trip'] = False
    catalog: dict[tuple[str, str, str], dict[str, Any]] = {}
    for route, block in frame.groupby('route', sort=True):
        profiles = profiles_for_day(feed, str(route), date.fromisoformat(day))
        duties: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for profile in profiles:
            duty = duty_component(profile['trip_id'])
            if duty is None:
                continue
            hashed = hashlib.sha256((salt + ':exit:' + duty).encode()).hexdigest()[:24]
            duties[hashed].append(profile)
            for stop, name, coords in zip(profile['stop_ids'], profile['stop_names'],
                                          profile['coordinates'], strict=True):
                key = str(route), str(profile['direction']), 'gtfs:' + stop
                catalog[key] = dict(route=str(route), direction=key[1], stop_id=key[2],
                                    name=name, lat=coords[0], lon=coords[1])
        if profiles:
            frame.loc[block.index, 'status'] = 'unknown_duty'
        for duty, events in block.groupby('exit_key', sort=False):
            candidates = duties.get(duty, [])
            if not candidates:
                continue
            times = events.second.to_numpy(dtype=float)
            counts, chosen = enroll(times, candidates)
            frame.loc[events.index, 'status'] = np.where(counts > 1, 'ambiguous_trip', 'no_trip')
            frame.loc[events.index[counts == 1], 'unique_trip'] = True
            for i in np.unique(chosen[chosen >= 0]):
                subset = events.index[chosen == i]
                profile = candidates[i]
                positions = nearest_stable(times[chosen == i], profile['times'])
                frame.loc[subset, 'status'] = 'stop_clock_unstable'
                frame.loc[subset, 'trip'] = profile['profile_id']
                accepted = positions >= 0
                selected = subset[accepted]
                frame.loc[selected, 'status'] = 'assigned'
                frame.loc[selected, 'direction'] = profile['direction']
                frame.loc[selected, 'stop_id'] = [
                    'gtfs:' + profile['stop_ids'][int(j)] for j in positions[accepted]]
    ledger = frame.groupby(['route', 'hour', 'status'], sort=True).size().reset_index(name='count')
    ledger.insert(0, 'day', day)
    counts = frame.groupby(['route', 'hour', 'stop_id', 'direction'], sort=True).size().reset_index(
        name='expected_count')
    counts.insert(0, 'day', day)
    counts['event_hour'] = [f'{day}T{int(h):02}:00:00+03:00' for h in counts.hour]
    counts['weekday'] = date.fromisoformat(day).weekday()
    counts['applicable_pattern'] = ~counts.stop_id.str.startswith('unallocated:')
    counts['training_eligible'] = True
    counts['label_origin'] = np.where(counts.applicable_pattern,
        'inferred_nearest_scheduled_arrival', 'unallocated_observed_route_mass')
    if int(ledger['count'].sum()) != len(frame) or int(counts.expected_count.sum()) != len(frame):
        raise ValueError('source mass lost')
    trips = frame.loc[frame.unique_trip].groupby(['route', 'hour', 'trip']).size()
    trip_rows = trips.reset_index(name='count')
    trip_rows.insert(0, 'day', day)
    return counts[FIELDS], ledger, (list(catalog.values()), trip_rows)


_WORK: tuple[Path, dict[str, Any], str, str] | None = None


def _initialize(source: Path, feed: dict[str, Any], salt: str, method: str = METHOD) -> None:
    global _WORK
    _WORK = source, feed, salt, method


def _day(day: str) -> tuple[Any, Any, Any]:
    assert _WORK is not None
    source, feed, salt, method = _WORK
    if method != METHOD:
        from .schedule_warp import TRANSFER_METHOD
        from .schedule_warp import map_day as anchored_map_day
        return anchored_map_day(source, feed, day, salt, method == TRANSFER_METHOD)
    return map_day(source, feed, day, salt)


def implementation_receipt() -> dict[str, Any]:
    """Bind all boarding modules and dependency lock, including transitive helpers."""
    root = Path(__file__).resolve().parents[4]
    paths = sorted(Path(__file__).parent.glob("*.py")) + [
        root / "uv.lock", root / "ml/src/tramflow_ml/route_models/features.py",
        root / "ml/src/tramflow_ml/route_models/data.py",
    ]
    hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"source_sha256": hashes, "git_revision": revision}


def rebuild(source: Path, archive: Path, audit_manifest: Path, out: Path,
            dates: list[str] | None = None, method: str = METHOD) -> dict[str, Any]:
    from .schedule_warp import METHOD as anchored_method
    from .schedule_warp import TRANSFER_METHOD
    if method not in (METHOD, anchored_method, TRANSFER_METHOD):
        raise ValueError("unknown mapping method")
    if out.exists():
        raise ValueError('new output directory required')
    source_meta = json.loads((source / 'manifest.json').read_text())
    if digest(audit_manifest) != source_meta['source_manifest_sha256']:
        raise ValueError('source audit differs')
    salt = json.loads(audit_manifest.read_text())['archive_sha256']
    days = sorted({p['date'] for p in source_meta['parts']}) if dates is None else sorted(dates)
    available = {p['date'] for p in source_meta['parts']}
    if not days or len(set(days)) != len(days) or not set(days) <= available:
        raise ValueError('unique source dates required')
    implementation = implementation_receipt()
    feed = extract_feed(archive)
    out.mkdir(parents=True)
    write_json(out / 'feed-coverage.json', {'routes': feed['routes'], 'trips': len(feed['trips']),
                                        'calendar': feed['calendar']})
    catalogs: dict[tuple[str, str, str], dict[str, Any]] = {}
    totals: dict[str, int] = defaultdict(int)
    files: dict[str, Any] = {}
    executor = ProcessPoolExecutor(max_workers=4, initializer=_initialize,
                                   initargs=(source, feed, salt, method))
    results = executor.map(_day, days)
    for day, result_day in zip(days, results, strict=True):
        rows, ledger, (catalog, trips) = result_day
        target = out / 'days' / day
        write_frame(target / 'soft.csv.gz', rows)
        write_frame(target / 'ledger.csv.gz', ledger)
        write_frame(target / 'trips.csv.gz', trips)
        for row in catalog:
            key = row['route'], row['direction'], row['stop_id']
            if key not in catalogs:
                catalogs[key] = {**row, 'first_date': day}
            catalogs[key]['last_date'] = day
        for status, count in ledger.groupby('status')['count'].sum().items():
            totals[str(status)] += int(count)
        print(day, int(ledger['count'].sum()), len(rows), flush=True)
    executor.shutdown()
    for split, begin, end in [('train', '2025-01-01', '2025-07-01'),
                               ('development', '2025-07-01', '2025-09-01'),
                               ('diagnostic', '2025-09-01', '2025-11-01')]:
        frames = [pd.read_csv(out / 'days' / day / 'soft.csv.gz')
                  for day in days if begin <= day < end]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)
        path = out / f'soft_{split}.csv.gz'
        write_frame(path, combined)
        files['soft_' + split] = {'name': path.name, 'sha256': digest(path),
                                 'rows': len(combined),
                                 'count': float(combined.expected_count.sum())}
    pd.DataFrame(catalogs.values()).to_csv(out / 'stop_catalog.csv', index=False)
    result = dict(schema_version='boarding-real-training.v1', complete=True, method=method,
                  feature_version='calendar.v1', timezone='Europe/Moscow', dates=days,
                  date_count=len(days), files=files, source_success=sum(totals.values()),
                  source_decoded=totals['assigned'],
                  source_unassigned=sum(totals.values())-totals['assigned'],
                  status_counts=dict(totals), observed_stop_labels=False, calibrated=False,
                  real_stop_accuracy='unverified', point_in_time_sources_verified=False,
                  diagnostic_is_blind=False, stop_namespace='gtfs',
                  source_manifest_sha256=digest(source / 'manifest.json'),
                  feed_sha256=digest(archive),
                  implementation_sha256=digest(Path(__file__)),
                  catalog_sha256=digest(out / 'stop_catalog.csv'),
                  target='validation_count',
                  label_origin=('inferred_nearest_scheduled_arrival' if method == METHOD
                                else ('inferred_active_or_transferred_first_stop'
                                      if method == TRANSFER_METHOD
                                      else 'inferred_first_stop_local_schedule')))
    expected = sum(p['successful'] for p in source_meta['parts'] if p['date'] in days)
    if expected != result['source_success']:
        raise ValueError('source total mismatch')
    if implementation_receipt() != implementation:
        raise ValueError('implementation changed during mapping')
    result['implementation_sources'] = implementation
    if method in (anchored_method, TRANSFER_METHOD):
        result['anchor_policy'] = {
            'first_validation_is_first_stop': True,
            'trip_padding_seconds': 900,
            'strong_detector': 'adaptive_supports+coalesce.adaptive95',
            'adaptive95': 3.6527760293461156,
            'calibration_period': '2025-01 through 2025-04',
            'threshold_source': 'boarding-multiscale/multiscale-v3/thresholds.json',
            'threshold_source_sha256':
                'def1c72288982e829e08e58ca59d2ad37fba7980f3827a238b26d3f6da7b777b',
            'clock_rule': 'strong onset shifts matched stop and all later stops',
            'allocation_rule': 'previous stop; exact arrival belongs to arriving stop',
            'source_calendar_relaxed': method == TRANSFER_METHOD,
            'calendar_transfer': ('nearest same route/duty/2025 day-type; past wins ties'
                                  if method == TRANSFER_METHOD else None),
        }
    write_json(out / 'manifest.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'archive', 'audit-manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dates', nargs='+')
    parser.add_argument('--anchored', action='store_true')
    parser.add_argument('--transfer-calendar', action='store_true')
    args = parser.parse_args()
    if args.transfer_calendar and not args.anchored:
        parser.error('--transfer-calendar requires --anchored')
    from .schedule_warp import METHOD as anchored_method
    from .schedule_warp import TRANSFER_METHOD
    method = TRANSFER_METHOD if args.transfer_calendar else anchored_method
    rebuild(args.source, args.archive, args.audit_manifest, args.out, args.dates,
            method if args.anchored else METHOD)


if __name__ == '__main__':
    main()

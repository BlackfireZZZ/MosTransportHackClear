"""Read-only GTFS coverage audit before allocating payments to historical trips."""

import argparse
import csv
import io
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .audit import digest, write_json
from .clock_gtfs import clock_seconds, profiles_for_day, service_active

ROUTES = ('1', '7', '11', '12', '17', '25', '26', '28', '50')


def read_feed(path: Path, routes: tuple[str, ...] = ROUTES) -> dict[str, Any]:
    """Select tram routes without changing validity; quarantine global orphan stop times."""
    with ZipFile(path) as archive:
        if 'calendar.txt' not in archive.namelist():
            raise ValueError('calendar_dates-only feeds unsupported by mapper preflight')
        def rows(name: str) -> Any:
            with archive.open(name) as stream:
                yield from csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig'))
        selected = {r['route_id']: r for r in rows('routes.txt')
                    if r['route_type'] == '0' and r['route_short_name'] in routes}
        identifiers: set[str] = set()
        trips = {}
        for row in rows('trips.txt'):
            if row['trip_id'] in identifiers:
                raise ValueError('duplicate trip identifier')
            identifiers.add(row['trip_id'])
            if row['route_id'] in selected:
                trips[row['trip_id']] = {**row, 'stop_times': []}
        orphan_stop_times = 0
        for row in rows('stop_times.txt'):
            if row['trip_id'] not in identifiers:
                orphan_stop_times += 1
                continue
            if row['trip_id'] in trips:
                trips[row['trip_id']]['stop_times'].append(row)
        feed = {'routes': list(selected.values()), 'trips': list(trips.values()),
                'calendar': list(rows('calendar.txt')),
                'calendar_dates': list(rows('calendar_dates.txt'))
                if 'calendar_dates.txt' in archive.namelist() else [],
                'stops': {r['stop_id']: r for r in rows('stops.txt')},
                'orphan_stop_times': orphan_stop_times}
    validate_feed(feed)
    return feed


def validate_feed(feed: dict[str, Any]) -> None:
    """Validate selected trips even when their service never intersects audit dates."""
    if len({t['trip_id'] for t in feed['trips']}) != len(feed['trips']):
        raise ValueError('duplicate trip identifier')
    route_ids = {r['route_id'] for r in feed['routes']}
    calendars = {c['service_id']: c for c in feed['calendar']}
    if len(calendars) != len(feed['calendar']):
        raise ValueError('duplicate service calendar')
    for calendar in calendars.values():
        service_active(calendar, date(2025, 1, 1))
    exception_keys = set()
    for exception in feed.get('calendar_dates', []):
        day = date.fromisoformat(exception['date'])
        key = (exception['service_id'], day)
        if key in exception_keys:
            raise ValueError('duplicate calendar exception')
        exception_keys.add(key)
        if exception['exception_type'] not in ('1', '2'):
            raise ValueError('invalid calendar exception type')
        if exception['service_id'] not in calendars:
            raise ValueError('calendar exception references absent service; '
                             'calendar_dates-only unsupported')
    for trip in feed['trips']:
        if trip['route_id'] not in route_ids or trip['service_id'] not in calendars:
            raise ValueError('trip references absent route or service')
        rows = sorted(trip['stop_times'], key=lambda r: int(r['stop_sequence']))
        if len(rows) < 2 or len({int(r['stop_sequence']) for r in rows}) != len(rows):
            raise ValueError('trip requires at least two unique ordered stop times')
        if any(r['stop_id'] not in feed['stops'] for r in rows):
            raise ValueError('trip references absent stop')
        arrivals = [clock_seconds(r['arrival_time']) for r in rows]
        departures = [clock_seconds(r['departure_time']) for r in rows]
        if any(t >= 172800 for t in departures):
            raise ValueError('service clocks beyond two days unsupported by mapper')
        if any(a > d for a, d in zip(arrivals, departures, strict=True)) or any(
                d > a for d, a in zip(departures, arrivals[1:], strict=False)):
            raise ValueError('backward trip clocks')
        if any(r.get('pickup_type', '') not in ('', '0') for r in rows):
            raise ValueError('nonstandard pickup requires explicit boarding policy')


def coverage(feed: dict[str, Any], start: date, end: date,
             routes: tuple[str, ...] = ROUTES) -> list[dict[str, Any]]:
    """Closed civil date range; only positive trip intervals intersecting the day count."""
    if end < start or not routes or len(set(routes)) != len(routes):
        raise ValueError('nonempty date range and unique routes required')
    validate_feed(feed)
    result = []
    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        for route in routes:
            candidates = profiles_for_day(feed, route, day)
            active = [p for p in candidates if
                      min(p['times'][-1], 172800) > max(p['times'][0], 86400)]
            intervals = sorted((max(p['times'][0], 86400), min(p['times'][-1], 172800))
                               for p in active)
            duration, last = 0, 86400
            for begin, finish in intervals:
                duration += max(0, finish - max(last, begin))
                last = max(last, finish)
            result.append({'date': str(day), 'route': route, 'trip_instances': len(active),
                           'previous_service_day_trips': sum(p['service_day'] < str(day)
                                                             for p in active),
                           'covered_seconds': duration, 'has_schedule': bool(active)})
    return result


def audit(archive: Path, output: Path, start: date, end: date) -> dict[str, Any]:
    if output.exists():
        raise ValueError('new output directory required')
    source_hash = digest(archive)
    feed = read_feed(archive)
    rows = coverage(feed, start, end)
    route_names = {r['route_id']: r['route_short_name'] for r in feed['routes']}
    groups = Counter((t['route_id'], t['service_id']) for t in feed['trips'])
    calendars = {r['service_id']: r for r in feed['calendar']}
    ranges = [{'route': route_names[r], 'trip_count': count,
               **{k: v for k, v in calendars[s].items() if k}}
              for (r, s), count in sorted(groups.items())]
    if digest(archive) != source_hash:
        raise ValueError('source changed during audit')
    output.mkdir(parents=True)
    with (output / 'route-day-coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'calendar-ranges.json', ranges)
    report = {'schema': 'gtfs-schedule-preflight.v1', 'source_sha256': source_hash,
              'date_range': [str(start), str(end)], 'timezone': 'Europe/Moscow',
              'routes': list(ROUTES), 'route_days': len(rows),
              'missing_route_days': sum(not r['has_schedule'] for r in rows),
              'selected_trips': len(feed['trips']),
              'integrity_scope': 'selected tram trips; orphan stop times quarantined',
              'global_orphan_stop_times': feed['orphan_stop_times'],
              'global_stop_times_integrity': feed['orphan_stop_times'] == 0,
              'historical_operation_confirmed': False, 'validity_extended': False,
              'coverage_definition': 'positive scheduled arrival interval intersects civil day',
              'implementation_sha256': {n: digest(Path(__file__).with_name(n))
                                       for n in ('schedule_preflight.py',
                                                 'clock_gtfs.py', 'audit.py')},
              'outputs_sha256': {p.name: digest(p) for p in output.iterdir()}}
    write_json(output / 'manifest.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument('--end', type=date.fromisoformat, default=date(2025, 10, 31))
    args = parser.parse_args()
    report = audit(args.archive, args.out, args.start, args.end)
    print(json.dumps(report, indent=2))
    if report['missing_route_days'] or report['global_orphan_stop_times']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

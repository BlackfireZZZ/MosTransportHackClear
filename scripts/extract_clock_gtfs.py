"""Extract three tram schedules from a hash-bound downloaded GTFS archive."""

import argparse
import csv
import io
from pathlib import Path
from zipfile import ZipFile

from tramflow_ml.boarding.audit import digest, write_json

SOURCE_URL = 'https://files.mobilitydatabase.org/mdb-3226/mdb-3226-202606301640/mdb-3226-202606301640.zip'
EXPECTED_SHA256 = 'dec3ec0ccc2efb7f1327ee6f26a0636a4a271494490e6debedcb2c1ccc579e76'


def extract(path):
    if digest(path) != EXPECTED_SHA256:
        raise ValueError('source hash mismatch; a different feed requires a new audit')
    with ZipFile(path) as z:
        def rows(name):
            with z.open(name) as raw:
                yield from csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig'))
        routes = {r['route_id']: r for r in rows('routes.txt')
                  if r['route_type'] == '0' and r['route_short_name'] in ['17', '12', '11']}
        trips = {t['trip_id']: {**t, 'stop_times': []} for t in rows('trips.txt') if t['route_id'] in routes}
        for r in rows('stop_times.txt'):
            if r['trip_id'] in trips:
                trips[r['trip_id']]['stop_times'].append(r)
        stop_ids = {r['stop_id'] for t in trips.values() for r in t['stop_times']}
        service_ids = {t['service_id'] for t in trips.values()}
        return {
            'routes': list(routes.values()), 'trips': list(trips.values()),
            'calendar': [c for c in rows('calendar.txt') if c['service_id'] in service_ids],
            'calendar_dates': [c for c in rows('calendar_dates.txt') if c['service_id'] in service_ids]
                              if 'calendar_dates.txt' in z.namelist() else [],
            'stops': {s['stop_id']: s for s in rows('stops.txt') if s['stop_id'] in stop_ids},
            'source_url': SOURCE_URL, 'source_sha256': EXPECTED_SHA256,
            'capture_date': '2026-06-30', 'timezone': 'Europe/Moscow',
            'embedded_file_dates': {i.filename: list(i.date_time) for i in z.infolist()},
            'historical_operation_confirmed': False,
            'calendar_exceptions_available': 'calendar_dates.txt' in z.namelist(),
            'source_kind': 'community_gtfs_later_capture_declared_2025_service',
            'feed_license_verified': False,
        }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new output path required')
    write_json(a.out, extract(a.archive))

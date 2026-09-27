"""Calendar-filtered, provenance-preserving experimental GTFS clock profiles."""

from datetime import date, timedelta
from typing import Any

WEEKDAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')


def clock_seconds(value: str) -> int:
    parts = value.split(':')
    if len(parts) != 3 or any(not p.isdigit() for p in parts):
        raise ValueError('GTFS HH:MM:SS required')
    h, m, s = map(int, parts)
    if not 0 <= h < 72 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError('invalid bounded GTFS service-day clock')
    return h * 3600 + m * 60 + s


def service_active(calendar: dict[str, str], day: date) -> bool:
    start = date.fromisoformat(calendar['start_date'])
    end = date.fromisoformat(calendar['end_date'])
    if end < start or any(calendar[k] not in ('0', '1') for k in WEEKDAYS):
        raise ValueError('invalid service calendar')
    return start <= day <= end and calendar[WEEKDAYS[day.weekday()]] == '1'


def profiles_for_day(feed: dict[str, Any], route: str, day: date) -> list[dict[str, Any]]:
    """Use a positive two-day axis: observations are 86400 + Moscow civil seconds.

    Prior-service-day trips remain candidates after midnight. Published service
    validity does not prove historical availability of a later-captured feed.
    No missing timestamps, unordered trips, or non-pickup stops are silently fixed.
    """
    routes = {r['route_id']: r for r in feed['routes']
              if r['route_short_name'] == route and r['route_type'] == '0'}
    calendars = {c['service_id']: c for c in feed['calendar']}
    exceptions = {(e['service_id'], date.fromisoformat(e['date'])): e['exception_type']
                  for e in feed.get('calendar_dates', [])}
    output = []
    for service_day, shift in [(day - timedelta(days=1), 0), (day, 86400)]:
        for trip in feed['trips']:
            if trip['route_id'] not in routes:
                continue
            if trip['service_id'] not in calendars:
                raise ValueError('missing service calendar')
            active = service_active(calendars[trip['service_id']], service_day)
            exception = exceptions.get((trip['service_id'], service_day))
            if exception not in (None, '1', '2'):
                raise ValueError('invalid calendar exception')
            active = active if exception is None else exception == '1'
            if not active:
                continue
            rows = sorted(trip['stop_times'], key=lambda r: int(r['stop_sequence']))
            seq = [int(r['stop_sequence']) for r in rows]
            if len(rows) < 2 or len(set(seq)) != len(seq):
                raise ValueError('unique ordered stop sequence required')
            arrivals = [clock_seconds(r['arrival_time']) for r in rows]
            departures = [clock_seconds(r['departure_time']) for r in rows]
            if any(a > d for a, d in zip(arrivals, departures, strict=True)) or any(
                a < d for d, a in zip(departures, arrivals[1:], strict=False)
            ):
                raise ValueError('backward trip clock; source must be quarantined')
            if any(r.get('pickup_type', '') not in ('', '0') for r in rows):
                raise ValueError('nonstandard pickup requires explicit boarding policy')
            if service_day < day and departures[-1] < 86400 - 1800:
                continue
            stops = [feed['stops'][r['stop_id']] for r in rows]
            output.append({
                'profile_id': f"{service_day}:{trip['trip_id']}", 'trip_id': trip['trip_id'],
                'times': [v + shift for v in arrivals], 'inferred': False,
                'direction': trip['direction_id'], 'service_day': str(service_day),
                'stop_ids': [r['stop_id'] for r in rows],
                'stop_names': [s['stop_name'] for s in stops],
                'coordinates': [[float(s['stop_lat']), float(s['stop_lon'])] for s in stops],
                'source_basis': 'later_snapshot_declared_service_dates',
                'historical_operation_confirmed': False,
            })
    return output

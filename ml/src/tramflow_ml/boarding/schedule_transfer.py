"""Explicit nearest-date schedule templates; never asserted historical service."""

from datetime import date, timedelta
from typing import Any

from tramflow_ml.route_models.features import OFF, WORK

from .clock_gtfs import profiles_for_day, service_active
from .direction_probe import duty_component


def day_type(day: date) -> int:
    """2025 production calendar: workday, Saturday, Sunday/public holiday."""
    if day.year != 2025:
        raise ValueError('template transfer supports the audited 2025 calendar only')
    if day in WORK or (day.weekday() < 5 and day not in OFF):
        return 0
    return 1 if day.weekday() == 5 and day not in OFF else 2


def templates(feed: dict[str, Any], route: str, target: date) -> list[dict[str, Any]]:
    """Fill missing duty/service-day profiles with nearest same-type 2025 calendar day.

    Existing active duty profiles take precedence. Equal-distance donor days prefer
    the past. Every trip retains donor service day, calendar gap and explicit basis.
    """
    cache: dict[str, list[dict[str, Any]]] = {}
    active = profiles_for_day(feed, route, target)
    result = [{**p, 'schedule_basis': 'active_calendar', 'template_day': p['service_day']}
              for p in active]
    route_ids = {r['route_id'] for r in feed['routes']
                 if r['route_short_name'] == route and r['route_type'] == '0'}
    calendars = {c['service_id']: c for c in feed['calendar']}
    exceptions = {(e['service_id'], date.fromisoformat(e['date'])): e['exception_type']
                  for e in feed.get('calendar_dates', [])}
    duties: dict[str, set[str]] = {}
    for trip in feed['trips']:
        duty = duty_component(trip['trip_id'])
        if trip['route_id'] in route_ids and duty is not None:
            duties.setdefault(duty, set()).add(trip['service_id'])
    for service_day, offset in [(target - timedelta(days=1), 0), (target, 86400)]:
        if service_day.year != 2025:
            continue
        existing = {duty_component(p['trip_id']) for p in active
                    if p['service_day'] == str(service_day)}
        ordered = sorted(
            (date(2025, 1, 1) + timedelta(days=i) for i in range(365)),
            key=lambda d: (abs((d - service_day).days), d > service_day, d),
        )
        ordered = [d for d in ordered if day_type(d) == day_type(service_day)]
        for duty, services in sorted(duties.items()):
            if duty in existing:
                continue
            donor = None
            for candidate in ordered:
                matches = []
                for service in services:
                    exception = exceptions.get((service, candidate))
                    if exception not in (None, '1', '2'):
                        raise ValueError('invalid calendar exception')
                    enabled = service_active(calendars[service], candidate)
                    if enabled if exception is None else exception == '1':
                        matches.append(service)
                if matches:
                    donor = candidate
                    break
            if donor is None:
                continue
            key = str(donor)
            if key not in cache:
                cache[key] = profiles_for_day(feed, route, donor)
            for profile in cache[key]:
                if (profile['service_day'] != str(donor)
                        or duty_component(profile['trip_id']) != duty):
                    continue
                clocks = [v - 86400 + offset for v in profile['times']]
                if service_day < target and clocks[-1] < 86400 - 1800:
                    continue
                result.append({
                    **profile, 'times': clocks, 'service_day': str(service_day),
                    'profile_id': f'{service_day}:template:{donor}:{profile["trip_id"]}',
                    'schedule_basis': 'transferred_calendar', 'template_day': str(donor),
                    'template_gap_days': (donor - service_day).days,
                    'historical_operation_confirmed': False,
                })
    return result

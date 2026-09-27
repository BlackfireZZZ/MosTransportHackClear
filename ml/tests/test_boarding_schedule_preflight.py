from copy import deepcopy
from datetime import date

import pytest

from tramflow_ml.boarding.clock_gtfs import WEEKDAYS
from tramflow_ml.boarding.schedule_preflight import coverage, validate_feed


def feed():
    return {'routes': [dict(route_id='r', route_short_name='11', route_type='0')],
            'calendar': [dict(service_id='s', start_date='20250401', end_date='20250401',
                              **dict.fromkeys(WEEKDAYS, '1'))], 'calendar_dates': [],
            'stops': {s: dict(stop_name=s, stop_lat='55', stop_lon='37') for s in ('a', 'b')},
            'trips': [dict(route_id='r', service_id='s', trip_id='r_s_1_1', direction_id='0',
                           stop_times=[dict(stop_id=s, stop_sequence=str(i), arrival_time=t,
                                            departure_time=t)
                                       for i, (s, t) in enumerate(
                                           [('a', '23:55:00'), ('b', '24:10:00')])])]}


def test_missing_january_is_explicit_without_extending_calendar():
    f = feed()
    before = deepcopy(f)
    rows = coverage(f, date(2025, 1, 1), date(2025, 1, 31), ('11',))
    assert len(rows) == 31
    assert all(r['trip_instances'] == r['covered_seconds'] == 0 for r in rows)
    assert f == before


def test_midnight_previous_service_day_counts_real_overlap_only():
    rows = coverage(feed(), date(2025, 4, 1), date(2025, 4, 3), ('11',))
    assert [r['covered_seconds'] for r in rows] == [300, 600, 0]
    assert [r['previous_service_day_trips'] for r in rows] == [0, 1, 0]


def test_empty_trip_rejected_even_outside_requested_month():
    f = feed()
    f['trips'][0]['stop_times'] = []
    with pytest.raises(ValueError, match='at least two'):
        coverage(f, date(2025, 1, 1), date(2025, 1, 1), ('11',))


@pytest.mark.parametrize('field', ['route_id', 'service_id'])
def test_missing_referential_keys_fail(field):
    f = feed()
    f['trips'][0][field] = 'missing'
    with pytest.raises(ValueError, match='absent route or service'):
        validate_feed(f)


def test_missing_stop_reference_fails():
    f = feed()
    del f['stops']['a']
    with pytest.raises(ValueError, match='absent stop'):
        validate_feed(f)


def test_zip_orphan_stop_times_are_reported_and_never_counted(tmp_path):
    import csv
    import io
    from zipfile import ZipFile

    from tramflow_ml.boarding.schedule_preflight import read_feed

    f = feed()
    tables = {name: f[name] for name in ('routes', 'calendar')}
    tables['trips'] = [{k: v for k, v in f['trips'][0].items() if k != 'stop_times'}]
    tables['stops'] = [{'stop_id': k, **v} for k, v in f['stops'].items()]
    tables['stop_times'] = [dict(trip_id='r_s_1_1', **r)
                            for r in f['trips'][0]['stop_times']]
    tables['stop_times'].append({**tables['stop_times'][0], 'trip_id': 'absent'})
    path = tmp_path / 'feed.zip'
    with ZipFile(path, 'w') as archive:
        for name, rows in tables.items():
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(name + '.txt', stream.getvalue())
    parsed = read_feed(path, ('11',))
    assert parsed['orphan_stop_times'] == 1
    assert len(parsed['trips'][0]['stop_times']) == 2
    assert coverage(parsed, date(2025, 4, 1), date(2025, 4, 1), ('11',))[0][
        'trip_instances'] == 1


@pytest.mark.parametrize('kind,match', [('duplicate', 'duplicate calendar'),
                                      ('type', 'invalid calendar'),
                                      ('date', 'Invalid isoformat'),
                                      ('service', 'absent service')])
def test_invalid_exception_rejected_even_outside_audit(kind, match):
    f = feed()
    row = dict(service_id='s', date='20251231', exception_type='1')
    f['calendar_dates'] = [row]
    if kind == 'duplicate':
        f['calendar_dates'].append({**row, 'exception_type': '2'})
    elif kind == 'type':
        row['exception_type'] = '9'
    elif kind == 'date':
        row['date'] = 'bad'
    else:
        row['service_id'] = 'absent'
    with pytest.raises(ValueError, match=match):
        coverage(f, date(2025, 4, 1), date(2025, 4, 1), ('11',))


def test_exception_removal_and_addition_apply_to_midnight_service_day():
    f = feed()
    f['calendar_dates'] = [dict(service_id='s', date='20250401', exception_type='2'),
                           dict(service_id='s', date='20250402', exception_type='1')]
    rows = coverage(f, date(2025, 4, 1), date(2025, 4, 3), ('11',))
    assert [r['covered_seconds'] for r in rows] == [0, 300, 600]
    assert [r['previous_service_day_trips'] for r in rows] == [0, 0, 1]


def test_duplicate_trip_rejected_by_in_memory_boundary():
    f = feed()
    f['trips'].append(deepcopy(f['trips'][0]))
    with pytest.raises(ValueError, match='duplicate trip'):
        validate_feed(f)

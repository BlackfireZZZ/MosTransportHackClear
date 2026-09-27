from copy import deepcopy
from datetime import date

import pytest

from tramflow_ml.boarding.clock_gtfs import clock_seconds, profiles_for_day


def feed():
    calendar = dict(service_id='s', start_date='20250501', end_date='20250531',
                    monday='1', tuesday='1', wednesday='1', thursday='1', friday='1',
                    saturday='0', sunday='0')
    rows = [dict(stop_sequence=str(i), stop_id=str(i), arrival_time=t, departure_time=t)
            for i, t in enumerate(['23:59:00', '24:01:00'])]
    return dict(routes=[dict(route_id='r', route_short_name='17', route_type='0')],
                calendar=[calendar], trips=[dict(route_id='r', service_id='s', trip_id='t',
                                                direction_id='0', stop_times=rows)],
                stops={str(i): dict(stop_name=str(i), stop_lat='55.8', stop_lon='37.6')
                       for i in range(2)})


def test_prior_service_day_survives_midnight_and_calendar_end():
    p = profiles_for_day(feed(), '17', date(2025, 5, 31))
    assert len(p) == 1
    assert p[0]['service_day'] == '2025-05-30'
    assert p[0]['times'] == [86340, 86460]
    assert not p[0]['historical_operation_confirmed']
    assert profiles_for_day(feed(), '17', date(2025, 4, 30)) == []


def test_calendar_exception_and_tram_mode():
    f = feed()
    f['calendar_dates'] = [dict(service_id='s', date='20250531', exception_type='1')]
    assert len(profiles_for_day(f, '17', date(2025, 5, 31))) == 2
    f['routes'][0]['route_type'] = '3'
    assert profiles_for_day(f, '17', date(2025, 5, 31)) == []


def test_equal_minute_stops_retained_but_backward_clocks_rejected():
    f = feed()
    row = f['trips'][0]['stop_times'][1]
    row.update(arrival_time='23:59:00', departure_time='23:59:00')
    assert profiles_for_day(f, '17', date(2025, 5, 15))[0]['times'][0] == 86340
    row.update(arrival_time='23:58:00', departure_time='23:58:00')
    with pytest.raises(ValueError):
        profiles_for_day(f, '17', date(2025, 5, 15))


@pytest.mark.parametrize('value', ['25:99:00', '-1:00:00', '24:00', 'nan:00:00'])
def test_bad_clocks(value):
    with pytest.raises(ValueError):
        clock_seconds(value)


def test_service_day_clock_and_input_immutability():
    f = feed()
    before = deepcopy(f)
    profiles_for_day(f, '17', date(2025, 5, 15))
    assert f == before
    assert clock_seconds('25:01:02') == 90062

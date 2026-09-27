import numpy as np
import pytest

from tramflow_ml.boarding.schedule_warp import correct_clock, previous_stop, trip_windows


def test_first_stop_is_mandatory_and_local_anchor_changes_only_later_clocks():
    corrected, anchors = correct_clock([100, 200, 300, 400], 120, [120, 240, 350])
    assert corrected.tolist() == [120, 240, 350, 450]
    assert anchors == [(0, 120.), (1, 240.), (2, 350.)]
    assert previous_stop([120, 239, 240, 349, 350, 450, 451], corrected).tolist() == [
        0, 0, 1, 1, 2, 3, -1]


def test_no_detected_onsets_keeps_original_intervals():
    corrected, anchors = correct_clock([100, 200, 300], 150, [])
    assert corrected.tolist() == [150, 250, 350]
    assert anchors == [(0, 150.)]


def test_ambiguous_onset_does_not_shift_clock():
    corrected, anchors = correct_clock([100, 200, 300], 100, [250])
    assert corrected.tolist() == [100, 200, 300]
    assert len(anchors) == 1


def test_duplicate_stop_clocks_abstain_except_explicit_first_anchor():
    assert previous_stop([100, 101, 200, 210, 300], [100, 100, 200, 300]).tolist() == [
        0, -1, 2, 2, 3]


def test_bounded_windows_split_gap_without_stealing_original_trip_events():
    profiles = [{'times': [1000, 1200]}, {'times': [1600, 1800]}]
    low, high = trip_windows(profiles)
    assert low.tolist() == [100, 1400]
    assert high.tolist() == [1400, 2700]
    assert sum(a <= 1400 < b for a, b in zip(low, high, strict=True)) == 1


def test_overlap_not_resolved_arbitrarily():
    low, high = trip_windows([{'times': [1000, 1700]}, {'times': [1600, 1800]}])
    assert sum(a <= 1650 < b for a, b in zip(low, high, strict=True)) == 2


def test_time_translation_and_midnight():
    schedule = np.array([86300, 86500, 86700])
    corrected, _ = correct_clock(schedule, 86310, [86310, 86530])
    assert previous_stop([86310, 86410, 86530], corrected).tolist() == [0, 0, 1]
    later, _ = correct_clock(schedule + 86400, 172710, [172710, 172930])
    np.testing.assert_array_equal(later, corrected + 86400)


@pytest.mark.parametrize('schedule', [[], [2, 1], [0, np.nan]])
def test_invalid_clocks_rejected(schedule):
    with pytest.raises(ValueError):
        correct_clock(schedule, 1, [])


def test_mapper_uses_prior_day_anchor_and_conserves_target_hours(tmp_path, monkeypatch):
    import hashlib
    import json

    import pandas as pd

    from tramflow_ml.boarding import schedule_warp as mapper

    duty = hashlib.sha256(b'salt:exit:3').hexdigest()[:24]
    def frame(stamps, exits):
        return pd.DataFrame({
            'event_key': stamps, 'event_at': stamps, 'route': ['11'] * len(stamps),
            'success': ['1'] * len(stamps), 'exit_key': exits,
            'device_key': ['device'] * len(stamps), 'vehicle_key': ['vehicle'] * len(stamps),
        })
    previous = frame(['2025-05-12T23:58:30+03:00'], [duty])
    current = frame(['2025-05-13T00:00:10+03:00', '2025-05-13T00:02:20+03:00',
                     '2025-05-13T01:00:00+03:00'], [duty, duty, 'unknown'])
    (tmp_path / 'manifest.json').write_text(json.dumps({'parts': [
        {'date': '2025-05-12'}, {'date': '2025-05-13'}]}))
    monkeypatch.setattr(mapper, 'read_day', lambda _, day: previous if day.endswith('12')
                        else current)
    profile = {'trip_id': '1_2_1_3', 'profile_id': '2025-05-12:1_2_1_3',
               'direction': '0', 'times': [86300, 86500, 86700],
               'stop_ids': ['a', 'b', 'c'], 'stop_names': ['A', 'B', 'C'],
               'coordinates': [[55., 37.]] * 3}
    monkeypatch.setattr(mapper, 'profiles_for_day', lambda *a: [profile])
    monkeypatch.setattr(mapper, '_onsets', lambda t, d: [float(t[0])])
    rows, ledger, (_, trips) = mapper.map_day(tmp_path, {}, '2025-05-13', 'salt')
    assert rows.expected_count.sum() == ledger['count'].sum() == 3
    assert rows.groupby('hour').expected_count.sum().to_dict() == {0: 2, 1: 1}
    assert rows.loc[rows.stop_id.eq('gtfs:a'), 'expected_count'].sum() == 1
    assert rows.loc[rows.stop_id.eq('gtfs:b'), 'expected_count'].sum() == 1
    assert rows.loc[rows.stop_id.eq('unallocated:11'), 'expected_count'].sum() == 1
    assert trips.prior_day_context.all()
    assert trips.first_offset.tolist() == [10.]


def test_separate_trips_keep_separate_offsets():
    first, _ = correct_clock([100, 200, 300], 120, [120, 230])
    second, _ = correct_clock([400, 500, 600], 390, [390, 480])
    assert first.tolist() == [120, 230, 330]
    assert second.tolist() == [390, 480, 580]


def test_template_transfer_keeps_duty_day_type_and_prefers_past_ties(monkeypatch):
    from datetime import date

    from tramflow_ml.boarding import schedule_transfer as transfer

    calendar = dict(service_id='s', start_date='20250101', end_date='20251231',
                    monday='0', tuesday='0', wednesday='0', thursday='0',
                    friday='0', saturday='0', sunday='0')
    feed = {'routes': [{'route_id': 'r', 'route_short_name': '11', 'route_type': '0'}],
            'trips': [{'route_id': 'r', 'trip_id': '1_2_1_3', 'service_id': 's'}],
            'calendar': [calendar], 'calendar_dates': [
                {'service_id': 's', 'date': '20250113', 'exception_type': '1'},
                {'service_id': 's', 'date': '20250115', 'exception_type': '1'}]}
    def profiles(_feed, _route, day):
        if day not in (date(2025, 1, 13), date(2025, 1, 15)):
            return []
        return [dict(service_day=str(day), trip_id='1_2_1_3',
                     profile_id=str(day)+':trip', times=[90000, 90600])]
    monkeypatch.setattr(transfer, 'profiles_for_day', profiles)
    result = transfer.templates(feed, '11', date(2025, 1, 14))
    assert len(result) == 1
    assert result[0]['template_day'] == '2025-01-13'
    assert result[0]['service_day'] == '2025-01-14'
    assert result[0]['schedule_basis'] == 'transferred_calendar'
    assert result[0]['times'] == [90000, 90600]
    assert not result[0]['historical_operation_confirmed']
    active = transfer.templates(feed, '11', date(2025, 1, 13))
    assert len(active) == 1
    assert active[0]['schedule_basis'] == 'active_calendar'
    assert transfer.day_type(date(2025, 1, 1)) == 2
    assert transfer.day_type(date(2025, 11, 1)) == 0


def test_real_gtfs_transferred_overnight_trip_does_not_reset_at_midnight(tmp_path, monkeypatch):
    import hashlib
    import json

    import pandas as pd

    from tramflow_ml.boarding import schedule_warp as mapper

    calendar = dict(service_id='s', start_date='20250513', end_date='20250513',
                    monday='1', tuesday='1', wednesday='1', thursday='1',
                    friday='1', saturday='0', sunday='0')
    feed = dict(
        routes=[dict(route_id='r', route_short_name='17', route_type='0')],
        calendar=[calendar], trips=[dict(
            route_id='r', service_id='s', trip_id='1_2_1_3', direction_id='0',
            stop_times=[dict(stop_sequence=str(i), stop_id=str(i),
                             arrival_time=t, departure_time=t)
                        for i, t in enumerate(['23:50:00', '24:00:00', '24:20:00'])])],
        stops={str(i): dict(stop_name=str(i), stop_lat='55.8', stop_lon='37.6')
               for i in range(3)},
    )
    duty = hashlib.sha256(b'salt:exit:3').hexdigest()[:24]
    stamps = ['2025-01-13T23:50:10+03:00', '2025-01-14T00:00:20+03:00']
    frame = pd.DataFrame(dict(event_key=stamps, event_at=stamps, route=['17'] * 2,
                             success=['1'] * 2, exit_key=[duty] * 2,
                             device_key=['d'] * 2, vehicle_key=['v'] * 2))
    (tmp_path / 'manifest.json').write_text(json.dumps({'parts': [
        {'date': '2025-01-13'}, {'date': '2025-01-14'}]}))
    monkeypatch.setattr(mapper, 'read_day', lambda _, day: frame.loc[
        frame.event_at.str.startswith(day)].copy())
    monkeypatch.setattr(mapper, '_onsets', lambda t, d: [float(t[0])])
    rows, ledger, (_, trips) = mapper.map_day(tmp_path, feed, '2025-01-14', 'salt', True)
    assert rows.expected_count.sum() == 1
    assert rows.stop_id.tolist() == ['gtfs:1']
    assert rows.label_origin.tolist() == ['inferred_transferred_calendar_first_stop']
    assert ledger.schedule_basis.tolist() == ['transferred_calendar']
    assert trips.template_day.tolist() == ['2025-05-13']
    assert trips.prior_day_context.all()

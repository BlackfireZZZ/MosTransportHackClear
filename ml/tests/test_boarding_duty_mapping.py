import numpy as np
import pytest

from tramflow_ml.boarding.duty_mapping import enroll, nearest_stable


def test_overlapping_trip_boundaries_abstain():
    count, selected = enroll([9, 10, 15, 20, 21], [{'times': [10, 15]}, {'times': [15, 20]}])
    assert count.tolist() == [0, 1, 2, 1, 0]
    assert selected.tolist() == [-1, 0, -1, 1, -1]


def test_stop_ties_repeated_clocks_and_sensitivity_abstain():
    assert nearest_stable([100, 160, 220, 300], [100, 220]).tolist() == [0, -1, 1, -1]
    assert nearest_stable([100], [100, 100, 220]).tolist() == [-1]
    assert nearest_stable([145], [100, 220]).tolist() == [-1]


def test_midnight_uses_extended_day_clock():
    counts, _ = enroll([86410], [{'times': [86300, 86600]}])
    assert counts.tolist() == [1]


@pytest.mark.parametrize('schedule', [[2, 1], [np.nan], []])
def test_invalid_stop_clock_rejected(schedule):
    with pytest.raises(ValueError):
        nearest_stable([1], schedule)


def test_day_conserves_unmatched_mass_and_original_hour(monkeypatch):
    import hashlib
    from pathlib import Path

    import pandas as pd

    from tramflow_ml.boarding import duty_mapping as mapping

    hashed = hashlib.sha256(b'salt:exit:3').hexdigest()[:24]
    frame = pd.DataFrame({'success': ['1', '1', '1', '0'],
        'event_at': ['2025-05-13T00:01:40+03:00', '2025-05-13T00:02:40+03:00',
                     '2025-05-13T01:00:00+03:00', '2025-05-13T00:01:40+03:00'],
        'route': ['11'] * 4, 'exit_key': [hashed, hashed, 'unknown', hashed]})
    profile = {'trip_id': '1_2_1_3', 'profile_id': 'test-trip', 'direction': '0',
               'times': [86500, 86620], 'stop_ids': ['a', 'b'],
               'stop_names': ['A', 'B'], 'coordinates': [[55., 37.], [55.1, 37.1]]}
    monkeypatch.setattr(mapping, 'read_day', lambda *a: frame)
    monkeypatch.setattr(mapping, 'profiles_for_day', lambda *a: [profile])
    rows, ledger, (_, trips) = mapping.map_day(Path('.'), {}, '2025-05-13', 'salt')
    assert rows.expected_count.sum() == ledger['count'].sum() == 3
    assert rows.groupby('hour').expected_count.sum().to_dict() == {0: 2, 1: 1}
    assert rows.loc[rows.stop_id.eq('gtfs:a'), 'expected_count'].sum() == 1
    assert rows.loc[rows.stop_id.eq('unallocated:11'), 'expected_count'].sum() == 2
    assert trips['count'].sum() == 2
    assert set(rows.label_origin) == {
        'inferred_nearest_scheduled_arrival', 'unallocated_observed_route_mass'}


def test_empty_dates_rejected_before_feed_extraction(tmp_path):
    import json

    from tramflow_ml.boarding.audit import digest
    from tramflow_ml.boarding.duty_mapping import rebuild

    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'archive_sha256': 'salt'}))
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_text(json.dumps({
        'source_manifest_sha256': digest(audit), 'parts': [{'date': '2025-01-01'}]}))
    with pytest.raises(ValueError, match='unique source dates'):
        rebuild(source, tmp_path / 'absent.zip', audit, tmp_path / 'out', [])


def test_implementation_receipt_binds_transitive_sources_and_lock():
    from tramflow_ml.boarding.duty_mapping import implementation_receipt

    receipt = implementation_receipt()
    assert 'uv.lock' in receipt['source_sha256']
    for name in ('duty_mapping.py', 'clock_gtfs.py', 'direction_probe.py', 'partitions.py'):
        key = 'ml/src/tramflow_ml/boarding/' + name
        assert len(receipt['source_sha256'][key]) == 64
    assert len(receipt['git_revision']) == 40

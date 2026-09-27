import pandas as pd
import pytest

from tramflow_ml.boarding.real_stability import (
    compare_assignments,
    perturb,
    select_sessions,
)


def frame():
    return pd.DataFrame({'event_key': [f'e{i:02}' for i in range(25)],
                         'second': [i * 10. for i in range(25)],
                         'event_at': ['2025-01-15T12:00:00'] * 25})


def test_perturbations_deterministic_order_independent_and_immutable():
    original = frame()
    saved = original.copy(deep=True)
    thinned = perturb(original, 'thin20pct', 20260926)
    assert len(thinned) == 20
    assert set(thinned.event_key) <= set(original.event_key)
    pd.testing.assert_frame_equal(thinned,
                                  perturb(original.iloc[::-1], 'thin20pct', 20260926))
    jitter = perturb(original, 'jitter5seconds', 20260926).set_index('event_key')
    unchanged = original.set_index('event_key')
    assert (jitter.second - unchanged.second).abs().max() <= 5
    assert (jitter.second - unchanged.second).abs().max() > 0
    pd.testing.assert_series_equal(jitter.event_at, unchanged.event_at)
    pd.testing.assert_frame_equal(original, saved)
    with pytest.raises(ValueError, match='unknown'):
        perturb(original, 'unrecognized', 1)


def test_session_selection_filters_and_tiebreaks_by_id():
    rows = []
    for key, count, kind in [('a', 20, 'inferred_vehicle_pool'),
                             ('b', 25, 'inferred_vehicle_pool'),
                             ('c', 25, 'inferred_vehicle_pool'),
                             ('d', 50, 'inferred_vehicle_pool'),
                             ('e', 100, 'device_only')]:
        rows += [{'group': key, 'identity_kind': kind}] * count
    selected = select_sessions(pd.DataFrame(rows))
    assert [(rank, len(data), data.group.iloc[0]) for rank, data in selected] == [
        ('low', 20, 'a'), ('median', 25, 'b'), ('high', 50, 'd')]


def test_comparison_nulls_weak_changes_and_mass_failures():
    baseline = [{'event_key': 'a', 'stop_id': '1', 'weak_label_eligible': True},
                {'event_key': 'b', 'stop_id': None, 'weak_label_eligible': False},
                {'event_key': 'c', 'stop_id': '3', 'weak_label_eligible': True}]
    changed = [{'event_key': 'a', 'stop_id': '2', 'weak_label_eligible': True},
               {'event_key': 'b', 'stop_id': None, 'weak_label_eligible': False}]
    summary = {'source_success': 2, 'soft_rows': [{'expected_count': 2.}]}
    result = compare_assignments(baseline, changed, summary)
    assert result['comparable_nonnull_stops'] == 1
    assert result['best_stop_agreement'] == 0
    assert result['weak_retained_changed_stop'] == 1
    assert result['weak_retention_rate'] == 1
    assert result['dropped_events'] == 1
    assert result['mass_per_retained_event'] == 1
    with pytest.raises(ValueError, match='mass'):
        compare_assignments(baseline, changed, {**summary, 'source_success': 3})
    with pytest.raises(ValueError, match='identity'):
        compare_assignments(baseline, changed + changed, summary)

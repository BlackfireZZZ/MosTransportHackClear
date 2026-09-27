"""Selective inference sees prefix costs, abstaining on weak or unstable evidence."""

from inspect import signature

import pytest

from tramflow_ml.boarding.direction_selective import select_direction


def profile(direction, times):
    return {'direction': direction, 'profile_id': direction, 'times': times,
            'stop_ids': [str(i) for i in range(len(times))]}


def profiles():
    return [profile('0', [0, 60, 120]), profile('1', [0, 90, 180])]


def test_selected_and_label_swap():
    source = profiles()
    result = select_direction([100, 160, 220], source)
    swapped = select_direction([100, 160, 220],
                               [{**p, 'direction': str(1-int(p['direction']))} for p in source])
    assert result['direction'] == '0'
    assert swapped['direction'] == '1'
    assert result['reason'] == swapped['reason'] == 'selected'
    assert not result['calibrated_probability']
    assert not result['direction_is_observed']
    for step in ('3', '6'):
        assert result['per_step'][step]['margin_per_interval'] == 30
        assert swapped['per_step'][step]['margin_per_interval'] == 30


def test_symmetric_costs_abstain_even_at_zero_threshold():
    result = select_direction([0, 60], [profile(d, [0, 60]) for d in ('0', '1')],
                              minimum_margin_per_interval=0)
    assert result['direction'] is None
    assert result['reason'] == 'tied_costs'


@pytest.mark.parametrize('source', [[], [profile('0', [0, 60])],
                                    [profile('0', [0, 60]), profile('1', [0])]])
def test_missing_direction_abstains(source):
    result = select_direction([0, 60], source, minimum_margin_per_interval=0)
    assert result['direction'] is None
    assert result['reason'] == 'missing_or_nonfinite_direction_cost'


@pytest.mark.parametrize(('threshold', 'direction'), [(0, '0'), (30, '0'), (30.01, None)])
def test_threshold_boundary_per_interval(threshold, direction):
    result = select_direction([0, 60, 120], profiles(), minimum_margin_per_interval=threshold)
    assert result['direction'] == direction
    assert result['reason'] == ('selected' if direction else 'insufficient_margin')


def test_different_step_winners_abstain():
    result = select_direction([0, 400], [profile('0', [0, 100, 200, 300, 400]),
                                          profile('1', [0, 320])],
                              minimum_margin_per_interval=0)
    assert result['per_step']['3']['direction'] == '1'
    assert result['per_step']['6']['direction'] == '0'
    assert result['direction'] is None
    assert result['reason'] == 'step_disagreement'


@pytest.mark.parametrize('threshold', [-1, float('nan'), float('inf')])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match='minimum margin'):
        select_direction([0, 60], profiles(), minimum_margin_per_interval=threshold)


@pytest.mark.parametrize('prefix', [[], [0], [0, 0], [1, 0], [0, float('nan')]])
def test_invalid_prefix(prefix):
    with pytest.raises(ValueError):
        select_direction(prefix, profiles())


def test_prefix_only_contract():
    assert list(signature(select_direction).parameters) == [
        'prefix_times', 'profiles', 'minimum_margin_per_interval',
    ]
    a = [0, 60, 120, 180, 240, 300]
    b = [0, 60, 120, 3000, 6000, 9000]
    assert select_direction(a[:3], profiles()) == select_direction(b[:3], profiles())


def test_profile_order_and_clock_translation_invariance():
    source = profiles()
    shifted = [{**p, 'times': [t+10000 for t in p['times']]} for p in source[::-1]]
    assert select_direction([0, 60, 120], source) == select_direction([99, 159, 219], shifted)

"""Frozen-prefix clock/prior ablation on real tram payments; no stop truth assumed."""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from tramflow_ml.boarding.absolute_clock import continue_profile, fit_profiles
from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.clock_gtfs import profiles_for_day

CONFIGS = {
    'interval_only': dict(clock_weight=0, start_prior=0, residual_weight=0),
    'clock_only': dict(clock_weight=.2, start_prior=0, residual_weight=.25),
    'clock_terminal60': dict(clock_weight=.2, start_prior=60, residual_weight=.25),
    'clock_terminal180': dict(clock_weight=.2, start_prior=180, residual_weight=.25),
    'strong_clock_terminal60': dict(clock_weight=1, start_prior=60, residual_weight=.25),
    'forced_terminal': dict(clock_weight=.2, start_prior=0, residual_weight=.25, start_mode='terminal'),
}
STABLE = ['clock_only', 'clock_terminal60', 'clock_terminal180', 'strong_clock_terminal60']


def describe(best, profile):
    return {**best, 'stop_ids': [profile['stop_ids'][i] for i in best['stop_indices']],
            'stop_names': [profile['stop_names'][i] for i in best['stop_indices']],
            'direction': profile['direction'], 'trip_id': profile['trip_id'],
            'service_day': profile['service_day']}


def evaluate(times, profiles, config):
    result = fit_profiles(times[:4], profiles, **config, alternatives=4)
    best = result['best']
    if best is None:
        return result
    by_id = {p['profile_id']: p for p in profiles}
    profile = by_id[best['profile_id']]
    result['best'] = describe(best, profile)
    result['alternatives'] = [describe(v, by_id[v['profile_id']]) for v in result['alternatives']]
    future = continue_profile(times[4:], profile, best, residual_weight=config['residual_weight'])
    result['continuation'] = describe(future, profile) if future else None
    i = best['stop_indices'][-1] + 1
    result['next_adjacent_error'] = float(times[4] - profile['times'][i] - best['initial_offset']) if i < len(profile['times']) else None
    result['full_stop_ids'] = (result['best']['stop_ids'] + result['continuation']['stop_ids'][1:]) if future else None
    return result


def stats(rows):
    out = {}
    for config in CONFIGS:
        group = [r['fits'][config] for r in rows]
        fits = [g for g in group if g['best'] is not None]
        successful = [g for g in fits if g.get('continuation') is not None]
        residuals = [abs(v) for g in successful for v in g['continuation']['local_residuals']]
        out[config] = {
            'sequences': len(group), 'prefix_feasible': len(fits), 'continuation_feasible': len(successful),
            'terminal_starts': sum(g['best']['start_is_terminal'] for g in fits),
            'median_abs_initial_offset': float(np.median([abs(g['best']['initial_offset']) for g in fits])) if fits else None,
            'future_intervals': len(residuals),
            'future_mae': float(np.mean(residuals)) if residuals else None,
            'future_median_abs': float(np.median(residuals)) if residuals else None,
            'future_within30': float(np.mean(np.array(residuals) <= 30)) if residuals else None,
            'median_prefix_score_gap': float(np.median([g['gap'] for g in fits if g['gap'] is not None])) if fits else None,
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--feed', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new output required')
    a.out.mkdir(parents=True)
    manifest = json.loads((a.inputs / 'manifest.json').read_text())
    assert digest(a.inputs / 'segments.json') == manifest['segments_sha256']
    segments = json.loads((a.inputs / 'segments.json').read_text())
    feed = json.loads(a.feed.read_text())
    rows, skipped = [], []
    profile_cache = {}
    for w in segments:
        key = (w['date'], w['route'])
        if key not in profile_cache:
            profile_cache[key] = profiles_for_day(feed, w['route'], date.fromisoformat(w['date']))
        profiles = profile_cache[key]
        for method, onsets in w['methods'].items():
            if len(onsets) < 6:
                skipped.append({**{k: w[k] for k in ['date', 'route', 'segment']}, 'method': method, 'onsets': len(onsets), 'reason': 'fewer_than_six_onsets'})
                continue
            times = [86400 + t for t in onsets[:8]]
            result = {**{k: v for k, v in w.items() if k != 'methods'}, 'method': method,
                      'onsets': onsets[:8], 'profiles_available': len(profiles), 'fits': {}}
            for name, config in CONFIGS.items():
                result['fits'][name] = evaluate(times, profiles, config)
            result['time_shift_controls'] = {
                str(shift): evaluate([t + shift for t in times], profiles, CONFIGS['clock_terminal60'])
                for shift in [420, 1020]
            }
            rows.append(result)
        print(w['date'], w['route'], w['segment'], len(rows), flush=True)
    summary = {'rows': len(rows), 'skipped': skipped, 'overall': stats(rows),
               'by_route_segment': {f'{route}/{segment}': stats([r for r in rows if r['route'] == route and r['segment'] == segment])
                                    for route in ['17', '12', '11'] for segment in ['dense_window', 'session_first_30m']},
               'by_month': {month: stats([r for r in rows if r['date'].startswith(month)]) for month in sorted({r['date'][:7] for r in rows})}}
    paired = {}
    base = 'interval_only'
    for name in CONFIGS:
        subset = [r for r in rows if r['fits'][base].get('continuation') and r['fits'][name].get('continuation')]
        paired[name] = {'sequences': len(subset)}
        for label in [base, name]:
            vals = [abs(v) for r in subset for v in r['fits'][label]['continuation']['local_residuals']]
            paired[name][label] = float(np.mean(vals)) if vals else None
    summary['paired_future_mae'] = paired
    controls = {}
    for shift in ['420', '1020']:
        subset = [r for r in rows if r['fits']['clock_terminal60'].get('continuation') and r['time_shift_controls'][shift].get('continuation')]
        controls[shift] = {'paired_sequences': len(subset)}
        for label in ['real', 'shifted']:
            vals = [abs(v) for r in subset for v in (r['fits']['clock_terminal60'] if label == 'real' else r['time_shift_controls'][shift])['continuation']['local_residuals']]
            controls[shift][label + '_future_mae'] = float(np.mean(vals)) if vals else None
    summary['time_shift_controls'] = controls
    station = defaultdict(lambda: {'observations': 0, 'stable_configurations': 0, 'dates': set(), 'future_observations': 0, 'future_local_error': [], 'other_method_matches': 0, 'other_method_same_stop': 0})
    indexed = {(r['date'], r['route'], r['segment'], r['method']): r for r in rows}
    for r in rows:
        if r['method'] != 'adaptive_scan':
            continue
        ref = r['fits']['clock_terminal60']
        ids = ref.get('full_stop_ids')
        if ids is None:
            continue
        profile = next(p for p in profile_cache[r['date'], r['route']] if p['profile_id'] == ref['best']['profile_id'])
        for i, stop_id in enumerate(ids):
            if i == 0:
                continue
            s = station[r['route'], profile['direction'], stop_id]
            s['name'] = profile['stop_names'][profile['stop_ids'].index(stop_id)]
            s['observations'] += 1
            s['dates'].add(r['date'])
            s['stable_configurations'] += all(r['fits'][cfg].get('full_stop_ids') is not None and r['fits'][cfg]['full_stop_ids'][i] == stop_id for cfg in STABLE)
            if i >= 4:
                s['future_observations'] += 1
                s['future_local_error'].append(abs(ref['continuation']['local_residuals'][i-4]))
            for method in ['consensus3_5s', 'feature_gmm']:
                other = indexed.get((r['date'], r['route'], r['segment'], method))
                if other is None or not other['fits']['clock_terminal60'].get('full_stop_ids'):
                    continue
                j = int(np.argmin(np.abs(np.array(other['onsets']) - r['onsets'][i])))
                if abs(other['onsets'][j] - r['onsets'][i]) <= 5:
                    s['other_method_matches'] += 1
                    s['other_method_same_stop'] += other['fits']['clock_terminal60']['full_stop_ids'][j] == stop_id
    stations = [{**s, 'route': route, 'direction': direction, 'stop_id': stop,
                 'dates': sorted(s['dates']), 'stable_fraction': s['stable_configurations']/s['observations'],
                 'future_mae': float(np.mean(s['future_local_error'])) if s['future_local_error'] else None}
                for (route, direction, stop), s in station.items()]
    first_observations = []
    for w in segments:
        if w['segment'] != 'session_first_30m':
            continue
        result = fit_profiles([86400 + w['session_first_clock']],
                              profile_cache[w['date'], w['route']],
                              start_mode='terminal', clock_weight=1, start_prior=0)
        b = result['best']
        first_observations.append({'date': w['date'], 'route': w['route'],
                                   'night_boundary': w['boundary_before_04'],
                                   'first_clock': w['session_first_clock'],
                                   'terminal_offset': b['initial_offset'] if b else None,
                                   'candidate_count': result['candidate_count']})
    write_json(a.out / 'first-observations.json', first_observations)
    write_json(a.out / 'rows.json', rows)
    write_json(a.out / 'summary.json', summary)
    write_json(a.out / 'stations.json', stations)
    write_json(a.out / 'manifest.json', {'schema_version': 'absolute-clock-experiment.v1',
               'timezone': 'Europe/Moscow', 'evaluation_dates': manifest['dates'],
               'prefix_onsets': 4, 'max_total_onsets': 8, 'max_offset': 900,
               'configs': CONFIGS, 'source_manifest_sha256': digest(a.inputs / 'manifest.json'),
               'feed_sha256': digest(a.feed), 'script_sha256': digest(Path(__file__)),
               'source_historical_capture_verified': False, 'stop_accuracy_verified': False,
               'evaluation': 'prefix-selected profile/offset, suffix forward stop matching still inferred',
               'module_hashes': {name: digest(Path(__file__).parents[1] / 'ml/src/tramflow_ml/boarding' / name) for name in ['absolute_clock.py', 'clock_gtfs.py']},
               'files': {p.name: digest(p) for p in a.out.iterdir() if p.is_file()}})
    print(json.dumps(summary['overall']))


if __name__ == '__main__':
    main()

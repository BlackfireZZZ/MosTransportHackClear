"""Frozen AM/PM bilateral matching and inferred GTFS duty-suffix controls."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.clock_gtfs import clock_seconds, profiles_for_day
from tramflow_ml.boarding.direction_probe import compare_directions, duty_component, shuffle_gaps
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import session_keys
from tramflow_ml.boarding.service_policy import load_service_policy
from tramflow_ml.boarding.wave_merge import adaptive_supports, coalesce

DATES = ['2025-05-13', '2025-05-17', '2025-06-10', '2025-06-14',
         '2025-07-08', '2025-07-12', '2025-08-12', '2025-08-16',
         '2025-09-09', '2025-09-13', '2025-10-14', '2025-10-18']
ROUTES = ('11', '12', '17')
HOURS = (7, 8, 17, 18)
THRESHOLD = 3.6527760293461156
METHODS = ('unrestricted', 'duty', 'duty_prior60', 'duty_prior180',
           'wrong_duty', 'shuffle_gaps', 'shift_clock')


def feed_structure(feed):
    groups = defaultdict(list)
    control = defaultdict(list)
    valid = 0
    for trip in feed['trips']:
        parts = trip['trip_id'].split('_')
        if duty_component(trip['trip_id']) is None:
            continue
        valid += int(parts[:2] == [trip['route_id'], trip['service_id']])
        stops = sorted(trip['stop_times'], key=lambda s: int(s['stop_sequence']))
        interval = (clock_seconds(stops[0]['departure_time']),
                    clock_seconds(stops[-1]['arrival_time']), trip['direction_id'], int(parts[2]))
        groups[tuple(parts[:2] + [parts[3]])].append(interval)
        control[tuple(parts[:3])].append(interval)
    def stats(grouped):
        pairs = [(a,b) for values in grouped.values() for a,b in
                 zip(sorted(values), sorted(values)[1:])]
        return {'groups': len(grouped), 'adjacent_pairs': len(pairs),
                'nonoverlap': sum(a[1] <= b[0] for a,b in pairs),
                'alternate_direction': sum(a[2] != b[2] for a,b in pairs),
                'ordinal_plus_one': sum(b[3] == a[3]+1 for a,b in pairs)}
    return {'trips': len(feed['trips']), 'route_service_components_match': valid,
            'duty_suffix': stats(groups), 'third_component_control': stats(control),
            'block_id_nonempty': sum(bool(t.get('block_id')) for t in feed['trips'])}


def clock_enrollment(block, profiles, hashed, wrong=False):
    duties = sorted(hashed.values())
    counts = Counter()
    by_period = {'AM': Counter(), 'PM': Counter()}
    for period, start, end in [('AM', 7, 9), ('PM', 17, 19)]:
        by_period[period]['all_events'] = int(block.second.between(start*3600, end*3600-1).sum())
    for key, events in block.groupby('exit_key', sort=True):
        duty = hashed.get(key)
        if duty is None:
            counts['unknown_duty'] += len(events)
            continue
        if wrong:
            duty = duties[(duties.index(duty)+1) % len(duties)]
        times = events.second.to_numpy(dtype=float) + 86400
        duty_profiles = [p for p in profiles if duty_component(p['trip_id']) == duty]
        all_directions = []
        all_matches = []
        for shift in (0, -300, 300, -900, 900):
            matches = np.zeros(len(times), dtype=int)
            directions = np.full(len(times), -1, dtype=int)
            for p in duty_profiles:
                inside = (times+shift >= p['times'][0]) & (times+shift <= p['times'][-1])
                matches += inside
                directions[inside] = int(p['direction'])
            all_directions.append(directions)
            all_matches.append(matches)
        matches, directions = all_matches[0], all_directions[0]
        stable300 = (matches == 1) & (all_matches[1] == 1) & (all_matches[2] == 1)
        stable300 &= (directions == all_directions[1]) & (directions == all_directions[2])
        stable900 = stable300 & (all_matches[3] == 1) & (all_matches[4] == 1)
        stable900 &= (directions == all_directions[3]) & (directions == all_directions[4])
        counts['unique_trip'] += int((matches == 1).sum())
        counts['no_trip'] += int((matches == 0).sum())
        counts['ambiguous_trip'] += int((matches > 1).sum())
        counts['stable300'] += int(stable300.sum())
        counts['stable900'] += int(stable900.sum())
        for period, start, end in [('AM', 7, 9), ('PM', 17, 19)]:
            scope = (times >= 86400+start*3600) & (times < 86400+end*3600)
            for d in (0, 1):
                by_period[period][str(d)] += int((scope & (matches == 1) & (directions == d)).sum())
                by_period[period]['stable300_' + str(d)] += int((scope & stable300 & (directions == d)).sum())
                by_period[period]['stable900_' + str(d)] += int((scope & stable900 & (directions == d)).sum())
    return {'counts': dict(counts), 'periods': {k: dict(v) for k, v in by_period.items()}}


def fit_window(job):
    row, profiles, duty, wrong = job
    times = [86400 + t for t in row['onsets']]
    selected = [p for p in profiles if duty_component(p['trip_id']) == duty]
    incorrect = [p for p in profiles if duty_component(p['trip_id']) == wrong]
    seed = int(hashlib.sha256(row['window_id'].encode()).hexdigest()[:8], 16)
    row['fits'] = {
        'unrestricted': compare_directions(times, profiles),
        'duty': compare_directions(times, selected),
        'duty_prior60': compare_directions(times, selected, start_prior=60),
        'duty_prior180': compare_directions(times, selected, start_prior=180),
        'wrong_duty': compare_directions(times, incorrect),
        'shuffle_gaps': compare_directions(shuffle_gaps(times, seed), selected),
        'shift_clock': compare_directions([t + 1020 for t in times], selected),
    }
    winners = [row['fits'][m]['winner'] for m in ('duty', 'duty_prior60', 'duty_prior180')]
    gap = row['fits']['duty']['margin']
    row['stable'] = len(set(winners)) == 1 and winners[0] is not None
    row['stable_margin30'] = row['stable'] and gap is not None and gap >= 30
    return row


def aggregate(rows):
    out = {}
    for method in METHODS:
        fits = [r['fits'][method] for r in rows]
        decided = [f for f in fits if f['winner'] is not None]
        paired = [f for f in decided if f['suffix_bilateral']]
        def mean(values):
            return float(np.mean(values)) if values else None
        out[method] = {
            'windows': len(rows), 'decided': len(decided),
            'prefix_bilateral': sum(f['prefix_bilateral'] for f in fits),
            'suffix_bilateral': sum(f['suffix_bilateral'] for f in fits),
            'winner_suffix_feasible': sum(f['fits'][f['winner']]['continuation'] is not None
                                          for f in decided),
            'winner_capped_loss': mean([f['fits'][f['winner']]['capped_loss'] for f in decided]),
            'all_window_capped_loss': mean([f['fits'][f['winner']]['capped_loss']
                                            if f['winner'] is not None else 900 for f in fits]),
            'paired_winner_mae': mean([f['fits'][f['winner']]['suffix_mae'] for f in paired]),
            'paired_opposite_mae': mean([f['fits'][str(1-int(f['winner']))]['suffix_mae']
                                         for f in paired]),
            'paired_positive_advantage': sum(f['chosen_suffix_advantage'] > 0 for f in paired),
            'direction_counts': dict(Counter(f['winner'] for f in decided)),
            'unresolved': len(rows) - len(decided),
        }
    return out


def period_stats(rows):
    result = []
    for route in ROUTES:
        for split in ('exploratory', 'heldout'):
            for period in ('AM', 'PM'):
                candidates = [r for r in rows if (r['route'], r['split'], r['period'])
                              == (route, split, period)]
                stable = [r for r in candidates if r['stable_margin30']]
                continued = [r for r in stable if r['fits']['duty']['fits'][r['fits']['duty']['winner']]['continuation'] is not None]
                weights = Counter()
                for r in stable:
                    weights[r['fits']['duty']['winner']] += r['segment_payments']
                result.append({'route': route, 'split': split, 'period': period,
                               'windows': len(candidates), 'stable_windows': len(stable),
                               'prefix_stable_with_feasible_suffix': len(continued),
                               'directions': dict(Counter(r['fits']['duty']['winner'] for r in stable)),
                               'segment_payment_weights': dict(weights),
                               'dates': sorted({r['date'] for r in stable})})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partitions', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--feed', type=Path, required=True)
    p.add_argument('--service-policy', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new output directory required')
    manifest = json.loads((a.partitions / 'manifest.json').read_text())
    if digest(a.audit_manifest) != manifest['source_manifest_sha256']:
        raise ValueError('partition source mismatch')
    archive_hash = json.loads(a.audit_manifest.read_text())['archive_sha256']
    feed = json.loads(a.feed.read_text())
    policy = load_service_policy(a.service_policy, source_root=a.source_root)
    a.out.mkdir(parents=True)
    jobs, excluded, enrollment, day_profiles = [], [], [], {}
    for day in DATES:
        frame = read_day(a.partitions, day, ROUTES)
        frame = frame.loc[frame.success.eq('1')].copy()
        clock = pd.to_datetime(frame.event_at)
        frame['second'] = clock.dt.hour*3600 + clock.dt.minute*60 + clock.dt.second
        frame = session_keys(frame)
        for route in ROUTES:
            profiles = profiles_for_day(feed, route, date.fromisoformat(day))
            duties = sorted({duty_component(p['trip_id']) for p in profiles})
            if None in duties:
                raise ValueError('unrecognized candidate duty namespace')
            hashed = {hashlib.sha256((archive_hash + ':exit:' + d).encode()).hexdigest()[:24]: d
                      for d in duties}
            block = frame.loc[frame.route.eq(route)].copy()
            known = block.exit_key.isin(hashed)
            enrollment.append({'date': day, 'route': route, 'successful_events': len(block),
                               'duty_code_matched_events': int(known.sum()),
                               'identity_conflict_events': int(block.identity_conflict.sum()),
                               'active_reference_duties': len(duties),
                               'clock': clock_enrollment(block, profiles, hashed),
                               'wrong_duty_clock': clock_enrollment(block, profiles, hashed, wrong=True)})
            day_profiles[day + '/' + route] = {
                d: sorted({(p['stop_names'][0], p['stop_names'][-1]) for p in profiles
                           if p['direction'] == d}) for d in ('0', '1')}
            eligible = block.loc[~block.identity_conflict & block.identity_kind.eq('inferred_vehicle_pool')]
            selected = []
            for group, session in eligible.groupby('group', sort=True):
                am = int(session.second.between(7*3600, 9*3600-1).sum())
                pm = int(session.second.between(17*3600, 19*3600-1).sum())
                if am >= 40 and pm >= 40:
                    selected.append((-(am+pm), group, session))
            for rank, (_, group, session) in enumerate(sorted(selected, key=lambda v: v[:2])[:2]):
                exit_key = session.exit_key.iloc[0]
                duty = hashed.get(exit_key)
                wrong = duties[(duties.index(duty)+1) % len(duties)] if duty is not None else None
                for hour in HOURS:
                    meta = {'window_id': f'{day}/{route}/{rank}/{hour}', 'date': day,
                            'route': route, 'rank': rank, 'hour': hour,
                            'period': 'AM' if hour < 12 else 'PM',
                            'split': 'exploratory' if day < '2025-07-01' else 'heldout',
                            'weekday': date.fromisoformat(day).weekday() < 5}
                    decision = policy.evaluate(route, datetime.fromisoformat(day).replace(
                        hour=hour, tzinfo=ZoneInfo('Europe/Moscow')), engine_applicable=True,
                        warnings=(), soft=True)
                    if not decision.eligible:
                        excluded.append({**meta, 'reason': 'service_policy', 'details': decision.reasons})
                        continue
                    if duty is None:
                        excluded.append({**meta, 'reason': 'duty_not_in_active_reference'})
                        continue
                    window = session.loc[session.second.between(hour*3600, (hour+1)*3600-1)]
                    window = window.sort_values(['second', 'event_key'])
                    times = window.second.to_numpy(dtype=float)
                    if len(times) < 20:
                        excluded.append({**meta, 'reason': 'fewer_than_20_payments'})
                        continue
                    devices = pd.factorize(window.device_key, sort=True)[0]
                    supports = adaptive_supports(times, devices, THRESHOLD)
                    onsets = coalesce(times, [s[0] for s in supports], supports)['times']
                    if len(onsets) < 8 or onsets[7]-onsets[0] > 1800:
                        excluded.append({**meta, 'reason': 'insufficient_compact_onsets',
                                         'onsets': len(onsets)})
                        continue
                    onsets = onsets[:8]
                    row = {**meta, 'onsets': onsets, 'window_payments': len(times),
                           'segment_payments': int(((times >= onsets[0]) & (times <= onsets[-1])).sum())}
                    jobs.append((row, profiles, duty, wrong))
        print(day, 'eligible windows', len(jobs), 'excluded', len(excluded), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(fit_window, jobs):
            rows.append(row)
            if len(rows) % 20 == 0:
                print('fitted', len(rows), '/', len(jobs), flush=True)
    splits = {s: [r for r in rows if r['split'] == s] for s in ('exploratory', 'heldout')}
    comparisons = {}
    for method in ('wrong_duty', 'shuffle_gaps', 'shift_clock', 'unrestricted'):
        matched = [r for r in splits['heldout'] if r['fits']['duty']['winner'] is not None
                   and r['fits'][method]['winner'] is not None]
        comparisons[method] = {'paired_windows': len(matched)}
        for label in ('duty', method):
            losses = [r['fits'][label]['fits'][r['fits'][label]['winner']]['capped_loss'] for r in matched]
            comparisons[method][label] = float(np.mean(losses)) if losses else None
    summary = {
        'windows': len(rows), 'excluded': excluded, 'enrollment': enrollment,
        'feed_structure': feed_structure(feed),
        'by_split': {s: aggregate(v) for s, v in splits.items()},
        'heldout_stable': aggregate([r for r in splits['heldout'] if r['stable_margin30']]),
        'by_route': {route: aggregate([r for r in splits['heldout'] if r['route'] == route])
                     for route in ROUTES},
        'periods': period_stats(rows), 'paired_control_losses': comparisons,
        'direction_endpoints': day_profiles,
        'accuracy_measured': False, 'sample_is_population_representative': False,
    }
    write_json(a.out / 'rows.json', rows)
    write_json(a.out / 'summary.json', summary)
    write_json(a.out / 'manifest.json', {
        'schema_version': 'direction-evidence.v1', 'timezone': 'Europe/Moscow',
        'dates': DATES, 'hours': HOURS, 'routes': ROUTES,
        'prefix': 4, 'suffix': 4, 'threshold': THRESHOLD,
        'source_archive_sha256': archive_hash, 'audit_manifest_sha256': digest(a.audit_manifest),
        'partitions_manifest_sha256': digest(a.partitions / 'manifest.json'),
        'feed_sha256': digest(a.feed), 'service_policy_sha256': digest(a.service_policy),
        'script_sha256': digest(Path(__file__)),
        'module_sha256': digest(Path(__file__).resolve().parents[1] /
                                'ml/src/tramflow_ml/boarding/direction_probe.py'),
        'files': {n: digest(a.out / n) for n in ('rows.json', 'summary.json')},
        'retrospective_only': True, 'direction_truth_available': False,
    })
    print('complete', len(rows), flush=True)


if __name__ == '__main__':
    main()

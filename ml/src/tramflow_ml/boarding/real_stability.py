"""Bounded retrospective perturbations measure repeatability, not stop accuracy."""

import hashlib
import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .audit import digest, write_json
from .composition import as_of, compose_route
from .historical import load_historical_catalog
from .partitions import read_day
from .real import RealConfig, _session, session_keys

SCENARIOS = ('gap20_span60', 'gap40_span120', 'thin20pct', 'jitter5seconds')


def perturb(frame: Any, scenario: str, seed: int) -> Any:
    """Keep original event identity/time; perturb only inference seconds or presence."""
    if scenario not in SCENARIOS:
        raise ValueError('unknown perturbation')
    if frame.event_key.duplicated().any() or frame.empty:
        raise ValueError('unique nonempty event keys required')
    result = frame.sort_values('event_key').copy(deep=True).reset_index(drop=True)
    random = np.random.default_rng(seed)
    if scenario == 'thin20pct':
        drop = random.choice(len(result), size=len(result) // 5, replace=False)
        result = result.drop(index=drop)
    elif scenario == 'jitter5seconds':
        result['second'] = result.second + random.uniform(-5, 5, len(result))
    return result.sort_values(['second', 'event_key']).reset_index(drop=True)


def select_sessions(frame: Any, minimum_events: int = 20) -> list[tuple[str, Any]]:
    """Select minimum, lower-median and maximum counts with group-ID tie breaking."""
    if minimum_events < 1:
        raise ValueError('positive minimum event count required')
    groups = [(str(key), group) for key, group in frame.groupby('group', sort=True)
              if len(group) >= minimum_events
              and group.identity_kind.eq('inferred_vehicle_pool').all()]
    groups.sort(key=lambda item: (len(item[1]), item[0]))
    if len(groups) < 3:
        raise ValueError('three eligible vehicle sessions required')
    return [(rank, groups[index][1].copy(deep=True)) for rank, index in
            [('low', 0), ('median', (len(groups) - 1) // 2), ('high', len(groups) - 1)]]


def compare_assignments(
    baseline: list[dict[str, Any]], changed: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Stop agreement excludes nulls; weak-label retention is on retained event keys."""
    before = {row['event_key']: row for row in baseline}
    after = {row['event_key']: row for row in changed}
    if (len(before) != len(baseline) or len(after) != len(changed)
            or not after.keys() <= before.keys()):
        raise ValueError('perturbation changed or duplicated original event identity')
    common = sorted(after)
    comparable = [key for key in common
                  if before[key]['stop_id'] is not None and after[key]['stop_id'] is not None]
    same = sum(before[key]['stop_id'] == after[key]['stop_id'] for key in comparable)
    baseline_weak = [key for key in common if before[key]['weak_label_eligible']]
    retained_weak = sum(after[key]['weak_label_eligible'] for key in baseline_weak)
    same_weak = sum(after[key]['weak_label_eligible']
                    and before[key]['stop_id'] == after[key]['stop_id'] for key in baseline_weak)
    weak_added = sum(after[key]['weak_label_eligible'] and not before[key]['weak_label_eligible']
                     for key in common)
    mass = sum(row['expected_count'] for row in summary['soft_rows'])
    if summary['source_success'] != len(changed) or not math.isclose(
            mass, len(changed), abs_tol=1e-7):
        raise ValueError('retained event mass is not conserved')
    return {'baseline_events': len(baseline), 'retained_events': len(changed),
            'dropped_events': len(baseline) - len(changed),
            'matched_event_keys': len(common), 'comparable_nonnull_stops': len(comparable),
            'same_best_stop': same,
            'best_stop_agreement': same / len(comparable) if comparable else None,
            'baseline_weak_on_retained_events': len(baseline_weak),
            'weak_retained': retained_weak, 'weak_removed': len(baseline_weak) - retained_weak,
            'weak_added': weak_added, 'weak_retained_same_stop': same_weak,
            'weak_retained_changed_stop': retained_weak - same_weak,
            'weak_retention_rate': retained_weak / len(baseline_weak) if baseline_weak else None,
            'scenario_weak_events': sum(row['weak_label_eligible'] for row in changed),
            'soft_expected_count': mass, 'mass_per_retained_event': mass / len(changed),
            'mass_conserved': True}


def run_sensitivity(
    partitions: Path, histories: Path, graph: Path, timetables: Path, out: Path,
    *, seed: int = 20260926,
) -> dict[str, Any]:
    """Run six frozen sessions sequentially without changing the reconstruction engine."""
    schedules = json.loads(timetables.read_text())
    results = []
    for day, route in [('2025-01-15', '1'), ('2025-07-29', '7')]:
        core = read_day(partitions, day)
        events = core.loc[core.success.eq('1')].copy()
        events['second'] = (pd.to_datetime(events.event_at).dt.tz_localize('Europe/Moscow')
                            .dt.as_unit('ns').astype('int64') / 1e9)
        events['core'] = True
        events = session_keys(events)
        patterns = load_historical_catalog(histories, graph, as_of(date.fromisoformat(day)))
        evidence = compose_route(patterns, schedules, route, date.fromisoformat(day))
        config = RealConfig(dates=(day,), routes=(route,), seed=seed)
        for rank, frame in select_sessions(events.loc[events.route.eq(route)]):
            original = frame.copy(deep=True)
            baseline, base_summary = _session(frame, evidence, config)
            case: dict[str, Any] = {'day': day, 'route': route, 'count_rank': rank,
                    'session_id': base_summary['session_id'], 'source_events': len(frame),
                    'event_keys_sha256': hashlib.sha256(
                        '\n'.join(sorted(frame.event_key)).encode()).hexdigest(),
                    'baseline_weak_events': sum(row['weak_label_eligible'] for row in baseline),
                    'baseline_burst_groups': base_summary['observed_payment_groups'],
                    'schedule_kinds': sorted(set(evidence.schedule_kinds)),
                    'applicable_pattern': evidence.applicable_pattern,
                    'warnings': evidence.warnings, 'sources': evidence.source_urls,
                    'scenarios': []}
            for scenario in SCENARIOS:
                altered = perturb(frame, scenario, seed)
                variant = (replace(config, gap_seconds=20, max_span_seconds=60)
                           if scenario == 'gap20_span60' else
                           replace(config, gap_seconds=40, max_span_seconds=120)
                           if scenario == 'gap40_span120' else config)
                rows, summary = _session(altered, evidence, variant)
                case['scenarios'].append({'scenario': scenario,
                                          'burst_groups': summary['observed_payment_groups'],
                                          **compare_assignments(baseline, rows, summary)})
            pd.testing.assert_frame_equal(original, frame)
            results.append(case)
            print(json.dumps({'day': day, 'route': route, 'rank': rank,
                              'events': len(frame), 'complete_scenarios': len(SCENARIOS)}),
                  flush=True)
    report = {'schema_version': 'boarding-real-sensitivity.v1', 'seed': seed,
              'timezone': 'Europe/Moscow', 'mode': 'retrospective_offline',
              'selection': 'six vehicle sessions; minimum/lower-median/maximum event counts >=20',
              'perturbed_time_field': 'second only; original event_at remains the mass ledger',
              'source_partition_manifest_sha256': digest(partitions / 'manifest.json'),
              'history_manifest_sha256': digest(histories / 'history-manifest.json'),
              'graph_sha256': digest(graph), 'timetables_sha256': digest(timetables),
              'implementation': {name: digest(Path(__file__).with_name(name)) for name in
                                 ['real.py', 'sequence.py', 'composition.py', 'real_stability.py']},
              'accuracy_measured': False, 'meaning': 'sensitivity, not empirical stop accuracy',
              'sessions': results}
    write_json(out, report)
    return report

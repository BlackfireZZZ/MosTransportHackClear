"""First-validation interval profiles; inferred endpoints are not observed stop truth."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date
from statistics import median
from typing import Any

Row = dict[str, Any]


def _day(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('dates must use YYYY-MM-DD')
    return parsed


def _quantiles(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    result: dict[str, float | None] = {}
    for name, fraction in [('p10', .1), ('p50', .5), ('p90', .9), ('p95', .95)]:
        if not ordered:
            result[name] = None
            continue
        position = fraction * (len(ordered) - 1)
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        result[name] = ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return result


def _dense(row: Row) -> bool:
    return min(int(row['from_events']), int(row['to_events'])) >= 3


def _rejection(row: Row) -> str | None:
    if not row.get('anchor_eligible', True):
        return 'ineligible_anchor'
    if int(row['steps']) != 1:
        return 'not_adjacent'
    if not _dense(row):
        return 'sparse_endpoints'
    if not math.isfinite(float(row['posterior'])) or float(row['posterior']) < .5:
        return 'low_posterior'
    if not math.isfinite(float(row['agreement'])) or float(row['agreement']) < .75:
        return 'scenario_disagreement'
    if row['identity_conflict']:
        return 'identity_conflict'
    if not row['applicable_pattern']:
        return 'inapplicable_pattern'
    if not row['vehicle_key']:
        return 'missing_vehicle'
    observed, expected = float(row['observed_seconds']), float(row['expected_seconds'])
    if not all(math.isfinite(x) and x > 0 for x in (observed, expected)):
        return 'invalid_interval'
    if abs(observed - expected) > max(90., .75 * expected):
        return 'large_local_residual'
    return None


def fit_dense_profiles(rows: list[Row], cutoff_date: str) -> Row:
    """Fit confident adjacent transitions strictly before cutoff, without volume weighting."""
    cutoff = _day(cutoff_date)
    groups: dict[tuple[str, int], list[Row]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    fit_vehicles: set[str] = set()
    fit_dates: set[str] = set()
    for row in rows:
        if _day(row['date']) >= cutoff:
            rejected['outside_fit_dates'] += 1
            continue
        reason = _rejection(row)
        if reason:
            rejected[reason] += 1
        else:
            groups[(str(row['pattern_key']), int(row['from_state']))].append(row)
            fit_vehicles.add(str(row['vehicle_key']))
            fit_dates.add(str(row['date']))
    edges: dict[str, dict[str, Row]] = defaultdict(dict)
    unsupported: list[Row] = []
    for (pattern, state), candidates in sorted(groups.items()):
        values = [float(row['observed_seconds']) for row in candidates]
        center = median(values)
        mad = median([abs(value - center) for value in values])
        threshold = max(30., 3. * 1.4826 * mad)
        kept = [row for row in candidates
                if abs(float(row['observed_seconds']) - center) <= threshold]
        rejected['robust_outlier'] += len(candidates) - len(kept)
        dates = sorted({str(row['date']) for row in kept})
        vehicles = sorted({str(row['vehicle_key']) for row in kept})
        failures = [name for name, condition in [
            ('observations_lt_5', len(kept) < 5), ('dates_lt_2', len(dates) < 2),
            ('vehicles_lt_2', len(vehicles) < 2),
        ] if condition]
        if failures:
            unsupported.append({'pattern_key': pattern, 'from_state': state,
                                'support': len(kept), 'reasons': failures})
            rejected['insufficient_edge_support'] += len(kept)
            continue
        values = [float(row['observed_seconds']) for row in kept]
        center = median(values)
        edges[pattern][str(state)] = {
            'median_seconds': center,
            'mad_seconds': median([abs(value - center) for value in values]),
            'support': len(kept), 'dates': dates, 'vehicles': vehicles,
            'fitted_through': dates[-1], 'interval_quantiles': _quantiles(values),
        }
    return {
        'version': 'dense-first-validation-v1', 'timezone': 'Europe/Moscow',
        'target': 'interval_between_first_validations_at_inferred_adjacent_visits',
        'cutoff_date': cutoff_date, 'fitted_through': max(fit_dates, default=None),
        'fit_vehicles': sorted(fit_vehicles), 'fit_dates': sorted(fit_dates),
        'edges': dict(edges), 'edge_count': sum(len(group) for group in edges.values()),
        'input_rows': len(rows), 'rejected_rows': dict(sorted(rejected.items())),
        'unsupported_edges': unsupported,
        'thresholds': {'minimum_endpoint_events': 3, 'minimum_posterior': .5,
                       'minimum_agreement': .75, 'minimum_support': 5,
                       'minimum_dates': 2, 'minimum_vehicles': 2,
                       'local_residual_floor_seconds': 90., 'local_residual_fraction': .75,
                       'outlier_floor_seconds': 30., 'outlier_scaled_mad_multiplier': 3.},
        'limitation': 'Inferred endpoints; agreement is timing consistency, not stop accuracy.',
    }


def lookup_profile(profile: Row, day: str, pattern_key: str, state: int) -> float | None:
    """Reject application before cutoff or on/before any fitted observation date."""
    requested = _day(day)
    if requested < _day(profile['cutoff_date']):
        return None
    fitted = profile.get('fitted_through')
    if fitted is not None and requested <= _day(fitted):
        return None
    edge = profile['edges'].get(pattern_key, {}).get(str(state))
    if edge is None or requested <= _day(edge['fitted_through']):
        return None
    return float(edge['median_seconds'])


def _summary(rows: list[Row]) -> Row:
    moving = [row for row in rows if int(row['steps']) > 0]
    skipped = [max(0, int(row['steps']) - 1) for row in moving]
    residuals = [float(row['observed_seconds']) - float(row['expected_seconds'])
                 for row in moving
                 if math.isfinite(float(row['observed_seconds']))
                 and math.isfinite(float(row['expected_seconds']))]
    steps = sum(int(row['steps']) for row in moving)
    plausible = sum(abs(float(row['observed_seconds']) - float(row['expected_seconds']))
                    <= max(90., .75 * float(row['expected_seconds'])) for row in moving)
    return {
        'transitions': len(rows), 'moving_transitions': len(moving),
        'resets': sum(int(row['steps']) < 0 for row in rows),
        'same_visit_transitions': sum(int(row['steps']) == 0 for row in rows),
        'inferred_edge_advances': steps, 'unobserved_intermediate_visits': sum(skipped),
        'unobserved_visit_fraction': sum(skipped) / steps if steps else None,
        'maximum_skipped_chain': max(skipped, default=0),
        'skip_chain_histogram': dict(sorted(Counter(map(str, skipped)).items())),
        'residual_seconds': _quantiles(residuals),
        'absolute_residual_seconds': _quantiles([abs(value) for value in residuals]),
        'locally_plausible_count': plausible,
        'locally_plausible_fraction': plausible / len(moving) if moving else None,
        'dates': len({row['date'] for row in rows}),
        'vehicles': len({row['vehicle_key'] for row in rows if row['vehicle_key']}),
    }


def summarize_transitions(rows: list[Row]) -> Row:
    """Describe inferred traversal consistency; missing validations do not prove skipped stops."""
    routes: dict[str, list[Row]] = defaultdict(list)
    edges: dict[tuple[str, int, int], list[Row]] = defaultdict(list)
    hotspots: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        routes[str(row['route'])].append(row)
        edges[(str(row['pattern_key']), int(row['from_state']), int(row['to_state']))].append(row)
        if int(row['steps']) > 1:
            for stop in row['skipped_stops']:
                hotspots[(str(row['route']), str(row['pattern_key']), str(stop))].append(row)
    return {
        **_summary(rows),
        'dense': _summary([row for row in rows if _dense(row)]),
        'sparse': _summary([row for row in rows if not _dense(row)]),
        'per_route': {key: _summary(group) for key, group in sorted(routes.items())},
        'per_edge': [{'pattern_key': key[0], 'from_state': key[1], 'to_state': key[2],
                      **_summary(group)} for key, group in sorted(edges.items())],
        'repeated_skipped_stop_hotspots': [
            {'route': key[0], 'pattern_key': key[1], 'stop_id': key[2],
             'inferred_unobserved_visits': len(group),
             'sessions': len({(row['date'], row['session_id']) for row in group}),
             'vehicles': len({row['vehicle_key'] for row in group if row['vehicle_key']})}
            for key, group in sorted(hotspots.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
        'interpretation': 'Traversal counts are inferred, not independent stop-count evidence; '
                          'no-payment visits do not imply the tram did not stop.',
    }


def validate_dense_holdout(rows: list[Row], profiles: Row) -> Row:
    """Compare fixed adjacent assignments on future, globally disjoint vehicles without refit."""
    cutoff = _day(profiles['cutoff_date'])
    fit_vehicles = set(profiles['fit_vehicles'])
    rejected: Counter[str] = Counter()
    baseline: list[float] = []
    learned: list[float] = []
    future_rows = independent_rows = 0
    for row in rows:
        if _day(row['date']) < cutoff:
            rejected['before_holdout'] += 1
            continue
        future_rows += 1
        if not row['vehicle_key']:
            rejected['missing_vehicle'] += 1
            continue
        if row['vehicle_key'] in fit_vehicles:
            rejected['fit_vehicle'] += 1
            continue
        independent_rows += 1
        if int(row['steps']) != 1 or row['identity_conflict'] or not row['applicable_pattern']:
            rejected['ineligible_transition'] += 1
            continue
        observed, expected = float(row['observed_seconds']), float(row['expected_seconds'])
        if not all(math.isfinite(x) and x > 0 for x in (observed, expected)):
            rejected['invalid_interval'] += 1
            continue
        prediction = lookup_profile(profiles, row['date'], row['pattern_key'], row['from_state'])
        if prediction is None:
            rejected['no_profile'] += 1
            continue
        baseline.append(abs(observed - expected))
        learned.append(abs(observed - prediction))
    return {
        'cutoff_date': profiles['cutoff_date'], 'future_rows': future_rows,
        'independent_rows': independent_rows, 'evaluated_rows': len(learned),
        'coverage_of_independent_rows': (len(learned) / independent_rows
                                         if independent_rows else None),
        'baseline_mae_seconds': sum(baseline) / len(baseline) if baseline else None,
        'profile_mae_seconds': sum(learned) / len(learned) if learned else None,
        'baseline_absolute_residual_seconds': _quantiles(baseline),
        'profile_absolute_residual_seconds': _quantiles(learned),
        'rejected_rows': dict(sorted(rejected.items())),
        'interpretation': 'Future-date and unseen-vehicle timing consistency on fixed inferred '
                          'adjacent assignments; not independent stop-label accuracy.',
    }

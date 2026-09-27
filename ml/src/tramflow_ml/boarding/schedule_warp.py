"""First-stop anchored, trip-local inferred allocation; coverage is not accuracy."""

import hashlib
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .clock_gtfs import profiles_for_day
from .direction_probe import duty_component
from .duty_mapping import FIELDS, nearest_stable
from .partitions import read_day
from .real import session_keys
from .wave_merge import adaptive_supports, coalesce

METHOD = 'duty-calendar-clock.first-stop-local-anchors.v2'
TRANSFER_METHOD = 'duty-calendar-clock.first-stop-local-anchors-transfer.v3'
ADAPTIVE95 = 3.6527760293461156
MAX_PADDING = 900.0


def _clock(values: Any) -> Any:
    result = np.asarray(values, dtype=float)
    if (result.ndim != 1 or not len(result) or not np.isfinite(result).all()
            or np.any(np.diff(result) < 0)):
        raise ValueError('nonempty finite ordered clocks required')
    return result


def correct_clock(schedule: Any, first: float, onsets: Any) -> tuple[Any, list[tuple[int, float]]]:
    """Pin first payment to stop zero; only stable strong onsets change later clocks."""
    original = _clock(schedule)
    if not np.isfinite(first):
        raise ValueError('finite first clock required')
    onsets = np.asarray(onsets, dtype=float)
    if (onsets.ndim != 1 or not np.isfinite(onsets).all()
            or np.any(np.diff(onsets) <= 0) or np.any(onsets < first)):
        raise ValueError('ordered unique onsets at or after first payment required')
    corrected = original + first - original[0]
    anchors = [(0, float(first))]
    for onset in onsets:
        if onset <= anchors[-1][1]:
            continue
        position = int(nearest_stable([onset], corrected)[0])
        if position <= anchors[-1][0] or onset <= corrected[position - 1]:
            continue
        corrected[position:] += onset - corrected[position]
        anchors.append((position, float(onset)))
    return corrected, anchors


def previous_stop(times: Any, corrected: Any) -> Any:
    """Half-open stop intervals, exact terminal included; equal clocks are ambiguous."""
    clock = _clock(corrected)
    values = np.asarray(times, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('finite event clocks required')
    positions = np.searchsorted(clock, values, side='right') - 1
    valid = (values >= clock[0]) & (values <= clock[-1])
    repeated = np.flatnonzero(np.r_[False, np.diff(clock) == 0])
    valid &= ~np.isin(positions, repeated)
    positions = np.where(valid, positions, -1)
    return np.where(values == clock[0], 0, positions)


def trip_windows(profiles: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Extend by at most 15 min, splitting gaps; overlapping original trips stay ambiguous."""
    clocks = [_clock(p['times']) for p in profiles]
    starts = np.array([v[0] for v in clocks])
    ends = np.array([v[-1] for v in clocks])
    low, high = starts - MAX_PADDING, ends + MAX_PADDING
    for i in range(len(clocks)):
        preceding = ends[ends <= starts[i]]
        following = starts[starts >= ends[i]]
        if len(preceding):
            low[i] = max(low[i], (starts[i] + preceding.max()) / 2)
        if len(following):
            high[i] = min(high[i], (ends[i] + following.min()) / 2)
    return low, high


def _onsets(times: Any, devices: Any) -> Any:
    supports = adaptive_supports(times, devices, ADAPTIVE95)
    return coalesce(times, [s[0] for s in supports], supports,
                    first_observation_anchor=True)['times']


def map_day(source: Path, feed: dict[str, Any], day: str, salt: str,
            transfer: bool = False) -> tuple[Any, Any, Any]:
    """Map successful events once, using previous civil day only as trip-start context."""
    import json

    meta = json.loads((source / 'manifest.json').read_text())
    available = {p['date'] for p in meta['parts']}
    previous = str(date.fromisoformat(day) - timedelta(days=1))
    frame = read_day(source, day)
    if previous in available:
        frame = pd.concat([read_day(source, previous), frame], ignore_index=True)
    frame = session_keys(frame.loc[frame.success.eq('1')].copy())
    stamp = pd.to_datetime(frame.event_at)
    frame['second'] = (stamp.dt.hour * 3600 + stamp.dt.minute * 60 + stamp.dt.second
                       + np.where(frame.event_at.str[:10].eq(day), 86400, 0))
    frame['hour'] = stamp.dt.hour
    frame['status'] = 'no_active_feed'
    frame['stop_id'] = 'unallocated:' + frame.route
    frame['direction'] = '-1'
    frame['trip'] = ''
    frame['unique_trip'] = False
    frame['allocation_basis'] = 'unallocated'
    frame['schedule_basis'] = 'unallocated'
    catalog: dict[tuple[str, str, str], dict[str, Any]] = {}
    diagnostics = []
    for route, block in frame.groupby('route', sort=True):
        if transfer:
            from .schedule_transfer import templates
            profiles = templates(feed, str(route), date.fromisoformat(day))
        else:
            profiles = profiles_for_day(feed, str(route), date.fromisoformat(day))
        duties: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for profile in profiles:
            duty = duty_component(profile['trip_id'])
            if duty is None:
                continue
            hashed = hashlib.sha256((salt + ':exit:' + duty).encode()).hexdigest()[:24]
            duties[hashed].append(profile)
            for stop, name, coords in zip(profile['stop_ids'], profile['stop_names'],
                                          profile['coordinates'], strict=True):
                key = str(route), str(profile['direction']), 'gtfs:' + stop
                catalog[key] = dict(route=key[0], direction=key[1], stop_id=key[2],
                                    name=name, lat=coords[0], lon=coords[1])
        if profiles:
            frame.loc[block.index, 'status'] = 'unknown_duty'
        for duty, events in block.groupby('exit_key', sort=False):
            candidates = duties.get(duty, [])
            if not candidates:
                continue
            low, high = trip_windows(candidates)
            times = events.second.to_numpy(float)
            eligible = (times[:, None] >= low) & (times[:, None] < high)
            counts = eligible.sum(axis=1)
            chosen = np.where(counts == 1, eligible.argmax(axis=1), -1)
            frame.loc[events.index, 'status'] = np.where(counts > 1, 'ambiguous_trip', 'no_trip')
            frame.loc[events.index[counts == 1], 'unique_trip'] = True
            for i in np.unique(chosen[chosen >= 0]):
                profile = candidates[int(i)]
                trip = events.loc[events.index[chosen == i]]
                for _, group in trip.groupby('group', sort=True):
                    group = group.sort_values(['second', 'event_key'], kind='stable')
                    target = group.event_at.str[:10].eq(day).to_numpy()
                    if not target.any():
                        continue
                    ix = group.index
                    frame.loc[ix, 'trip'] = profile['profile_id']
                    frame.loc[ix, 'schedule_basis'] = profile.get(
                        'schedule_basis', 'active_calendar')
                    if (group.identity_conflict.any()
                            or group.identity_kind.eq('identity_missing').any()):
                        frame.loc[ix, 'status'] = 'identity_conflict'
                        continue
                    t = group.second.to_numpy(float)
                    if abs(t[0] - profile['times'][0]) > MAX_PADDING:
                        frame.loc[ix, 'status'] = 'first_anchor_outside_bound'
                        continue
                    if profile['times'][0] < 86400 and previous not in available:
                        frame.loc[ix, 'status'] = 'missing_previous_day_context'
                        continue
                    devices = pd.factorize(group.device_key, sort=True)[0]
                    onsets = _onsets(t, devices)
                    corrected, anchors = correct_clock(profile['times'], float(t[0]), onsets)
                    positions = previous_stop(t, corrected)
                    selected = positions >= 0
                    frame.loc[ix, 'status'] = 'outside_corrected_trip_or_equal_clocks'
                    frame.loc[ix[selected], 'status'] = 'assigned'
                    frame.loc[ix[selected], 'direction'] = str(profile['direction'])
                    frame.loc[ix[selected], 'stop_id'] = [
                        'gtfs:' + profile['stop_ids'][j] for j in positions[selected]]
                    frame.loc[ix[selected], 'allocation_basis'] = 'schedule_previous_stop'
                    first_mask = selected & (t == t[0])
                    strong_mask = selected & np.isin(t, [a[1] for a in anchors[1:]])
                    frame.loc[ix[first_mask], 'allocation_basis'] = 'mandatory_first_stop'
                    frame.loc[ix[strong_mask], 'allocation_basis'] = 'strong_onset_anchor'
                    diagnostics.append(dict(
                        day=day, route=str(route), trip=profile['profile_id'],
                        count=int(target.sum()), assigned=int((selected & target).sum()),
                        anchors=len(anchors), first_offset=float(t[0] - profile['times'][0]),
                        max_correction=float(np.max(np.abs(corrected - profile['times']))),
                        prior_day_context=bool(t[0] < 86400),
                        schedule_basis=profile.get('schedule_basis', 'active_calendar'),
                        template_day=profile.get('template_day', profile.get('service_day', day)),
                        template_gap_days=profile.get('template_gap_days', 0),
                    ))
    frame = frame.loc[frame.event_at.str[:10].eq(day)]
    ledger = frame.groupby(
        ['route', 'hour', 'status', 'allocation_basis', 'schedule_basis']
    ).size().reset_index(name='count')
    ledger.insert(0, 'day', day)
    counts = frame.groupby(['route', 'hour', 'stop_id', 'direction']).agg(
        expected_count=('status', 'size'),
        schedule_basis=('schedule_basis', lambda x: ','.join(sorted(set(x)))),
    ).reset_index()
    counts.insert(0, 'day', day)
    counts['event_hour'] = [f'{day}T{int(h):02}:00:00+03:00' for h in counts.hour]
    counts['weekday'] = date.fromisoformat(day).weekday()
    counts['applicable_pattern'] = ~counts.stop_id.str.startswith('unallocated:')
    counts['training_eligible'] = True
    counts['label_origin'] = np.where(counts.applicable_pattern,
        np.where(counts.schedule_basis.eq('transferred_calendar'),
                 'inferred_transferred_calendar_first_stop',
                 np.where(counts.schedule_basis.eq('active_calendar'),
                          'inferred_first_stop_local_schedule',
                          'inferred_mixed_active_transferred_calendar')),
        'unallocated_observed_route_mass')
    expected = frame.groupby(['route', 'hour']).size()
    actual = counts.groupby(['route', 'hour']).expected_count.sum()
    if not expected.equals(actual):
        raise ValueError('original route-hour mass changed')
    trips = pd.DataFrame(diagnostics, columns=[
        'day', 'route', 'trip', 'count', 'assigned', 'anchors', 'first_offset',
        'max_correction', 'prior_day_context', 'schedule_basis', 'template_day',
        'template_gap_days'])
    return counts[FIELDS], ledger, (list(catalog.values()), trips)

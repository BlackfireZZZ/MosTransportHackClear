"""Restore absolute clocks of frozen windows and sample first observed session segments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.dense_experiment import busiest
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import session_keys
from tramflow_ml.boarding.wave_merge import adaptive_supports, coalesce


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new output required')
    a.out.mkdir(parents=True)
    manifest = json.loads((a.frozen / 'manifest.json').read_text())
    for name in ['windows.json', 'rows.json', 'thresholds.json']:
        assert digest(a.frozen / name) == manifest['files'][name]
    windows = json.loads((a.frozen / 'windows.json').read_text())
    rows = json.loads((a.frozen / 'rows.json').read_text())
    threshold = json.loads((a.frozen / 'thresholds.json').read_text())['adaptive']['95']
    selected = {(w['date'], w['route'], w['rank']): (w, r) for w, r in zip(windows, rows, strict=True)
                if w['date'] >= '2025-05-01' and w['route'] in ['17', '12', '11'] and w['rank'] == 0}
    output = []
    for day in sorted({k[0] for k in selected}):
        prior = json.loads((a.dense / (day + '.json')).read_text())['rows']
        keys = {r['session_key']: r for r in prior
                if r['method'] == 'gap30' and (day, r['route'], r['rank']) in selected}
        frame = read_day(a.source / 'boarding-date-shards', day, ('17', '12', '11'))
        frame = frame.loc[frame.success.eq('1')].copy()
        at = pd.to_datetime(frame.event_at)
        frame['second'] = at.dt.hour * 3600 + at.dt.minute * 60 + at.dt.second
        frame = session_keys(frame)
        for key, block in frame.groupby('group', sort=True):
            if key not in keys:
                continue
            assert not block.identity_conflict.any()
            assert block.identity_kind.eq('inferred_vehicle_pool').all()
            prior_row = keys[key]
            w, r = selected[day, prior_row['route'], prior_row['rank']]
            block = block.sort_values('second', kind='stable')
            t = block.second.to_numpy(dtype=float)
            roi = busiest(t)
            start = float(np.floor(roi[0] / 300) * 300)
            assert np.array_equal(roi - start, w['times'])
            supports = adaptive_supports(w['times'], w['devices'], threshold)
            metadata = {'date': day, 'route': w['route'], 'rank': w['rank'],
                        'trip_start_confirmed': False, 'session_first_clock': float(t[0]),
                        'session_events': len(t), 'boundary_before_04': bool(t[0] < 14400)}
            methods = {m: (np.array(coalesce(w['times'], r['methods'][m]['times'], supports,
                        first_observation_anchor=True)['times']) + start).tolist()
                        for m in ['adaptive_scan', 'consensus3_5s', 'feature_gmm']}
            output.append({**metadata, 'segment': 'dense_window', 'window_start': start,
                           'payments': len(roi), 'methods': methods})
            early = block.loc[block.second < t[0] + 1800]
            et = early.second.to_numpy(dtype=float)
            ed = pd.factorize(early.device_key, sort=True)[0]
            es = adaptive_supports(et, ed, threshold)
            onsets = coalesce(et, [v[0] for v in es], es, first_observation_anchor=True)['times']
            output.append({**metadata, 'segment': 'session_first_30m', 'window_start': float(t[0]),
                           'payments': len(et), 'methods': {'adaptive_scan': onsets}})
        print(day, len(output), flush=True)
    write_json(a.out / 'segments.json', output)
    write_json(a.out / 'manifest.json', {
        'schema_version': 'absolute-clock-inputs.v1', 'timezone': 'Europe/Moscow',
        'dates': sorted({k[0] for k in selected}), 'routes': ['17', '12', '11'],
        'detector_training': '2025-01-01..2025-04-30', 'evaluation': '2025-05-15..2025-10-18',
        'selection': 'frozen top-density rank0 sessions; dense window plus first30m of same session',
        'trip_start_confirmed': False, 'source_manifest_sha256': digest(a.source / 'boarding-date-shards/manifest.json'),
        'frozen_manifest_sha256': digest(a.frozen / 'manifest.json'),
        'script_sha256': digest(Path(__file__)), 'segments_sha256': digest(a.out / 'segments.json'),
        'segment_count': len(output), 'first_session_segments_are_not_independent_of_dense_segments': True,
    })


if __name__ == '__main__':
    main()

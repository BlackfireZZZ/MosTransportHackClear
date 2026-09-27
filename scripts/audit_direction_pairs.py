"""Schedule-labelled direction versus opposite under equal clock-free matching."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tramflow_ml.boarding.audit import digest, write_json
from tramflow_ml.boarding.clock_gtfs import profiles_for_day
from tramflow_ml.boarding.direction_comparison import compare_relative, scheduled_direction
from tramflow_ml.boarding.direction_probe import duty_component, shuffle_gaps
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import session_keys


def recover_duties(windows, feed, partitions, archive_hash):
    result = {}
    for day in sorted({r['date'] for r in windows}):
        routes = tuple(sorted({r['route'] for r in windows if r['date'] == day}))
        frame = read_day(partitions, day, routes)
        frame = frame.loc[frame.success.eq('1')].copy()
        clock = pd.to_datetime(frame.event_at)
        frame['second'] = clock.dt.hour*3600+clock.dt.minute*60+clock.dt.second
        frame = session_keys(frame)
        for route in routes:
            profiles = profiles_for_day(feed,route,date.fromisoformat(day))
            codes = {duty_component(p['trip_id']) for p in profiles}
            hashed = {hashlib.sha256((archive_hash+':exit:'+d).encode()).hexdigest()[:24]:d
                      for d in codes if d is not None}
            block = frame.loc[frame.route.eq(route) & ~frame.identity_conflict
                              & frame.identity_kind.eq('inferred_vehicle_pool')]
            ranked=[]
            for key, session in block.groupby('group',sort=True):
                am=int(session.second.between(7*3600,9*3600-1).sum())
                pm=int(session.second.between(17*3600,19*3600-1).sum())
                if am >= 40 and pm >= 40:
                    ranked.append((-(am+pm),key,session))
            ranked.sort(key=lambda v:v[:2])
            for row in [r for r in windows if r['date']==day and r['route']==route]:
                session=ranked[row['rank']][2]
                count=int(session.second.between(row['hour']*3600,(row['hour']+1)*3600-1).sum())
                if count != row['window_payments']:
                    raise ValueError('frozen window population changed')
                result[row['window_id']]=hashed.get(session.exit_key.iloc[0])
                if result[row['window_id']] is None:
                    raise ValueError('frozen duty not recovered from immutable ledger')
        print('verified source groups',day,flush=True)
    return result


def cluster_interval(rows, values):
    if not rows:
        return None
    days = sorted({r['date'] for r in rows})
    if len(days) < 2:
        return {'unit':'date','clusters':len(days),'percentile95':None,
                'reason':'fewer_than_two_date_clusters'}
    groups = [[v for r,v in zip(rows, values) if r['date'] == d] for d in days]
    rng = np.random.default_rng(20260927)
    means = [np.mean([v for i in rng.integers(0,len(days),len(days)) for v in groups[i]])
             for _ in range(2000)]
    return {'unit': 'date', 'clusters': len(days), 'resamples': 2000,
            'percentile95': np.quantile(means,[.025,.975]).tolist(),
            'exploratory_small_cluster_count': len(days) < 10}


def summarize(rows, method):
    labelled = [r for r in rows if r['scheduled']['direction'] is not None]
    out = {'windows': len(rows), 'labelled': len(labelled),
           'unknown_labels': len(rows)-len(labelled)}
    for part in ('suffix','full'):
        pairs = []
        for row in labelled:
            d = row['scheduled']['direction']
            a,b = row['fits'][method][d],row['fits'][method][str(1-int(d))]
            pairs.append((row,a[part],b[part]))
        feasible = [(r,a,b) for r,a,b in pairs if a['feasible'] and b['feasible']]
        deltas = [b['capped_loss']-a['capped_loss'] for _,a,b in pairs]
        def avg(values):
            return float(np.mean(values)) if values else None
        nulls = [float(np.mean(r['shuffle_deltas'][part])) for r in labelled]
        out[part] = {
            'assigned_feasible': sum(a['feasible'] for _,a,b in pairs),
            'opposite_feasible': sum(b['feasible'] for _,a,b in pairs),
            'both_feasible': len(feasible),
            'assigned_paired_mae': avg([a['mae'] for _,a,b in feasible]),
            'opposite_paired_mae': avg([b['mae'] for _,a,b in feasible]),
            'paired_wins_ties_losses': [sum(a['mae'] < b['mae']-1e-9 for _,a,b in feasible),
                                        sum(abs(a['mae']-b['mae']) <= 1e-9 for _,a,b in feasible),
                                        sum(a['mae'] > b['mae']+1e-9 for _,a,b in feasible)],
            'assigned_all_capped': avg([a['capped_loss'] for _,a,b in pairs]),
            'opposite_all_capped': avg([b['capped_loss'] for _,a,b in pairs]),
            'mean_opposite_minus_assigned': avg(deltas),
            'delta_date_cluster_interval': cluster_interval(labelled,deltas),
            'assigned_paired_skips': avg([a['skipped_stops'] for _,a,b in feasible]),
            'opposite_paired_skips': avg([b['skipped_stops'] for _,a,b in feasible]),
            'assigned_paired_cost': avg([a['cost_per_interval'] for _,a,b in feasible]),
            'opposite_paired_cost': avg([b['cost_per_interval'] for _,a,b in feasible]),
        }
        if method == 'duty_step6':
            contrast = [a-b for a,b in zip(deltas,nulls)]
            out[part]['shuffle_mean_delta'] = avg(nulls)
            out[part]['real_minus_shuffle_delta'] = avg(contrast)
            out[part]['contrast_date_cluster_interval'] = cluster_interval(labelled,contrast)
    out['prefix_ties_assigned'] = sum(r['fits'][method][r['scheduled']['direction']]['prefix_tied'] for r in labelled)
    out['prefix_ties_opposite'] = sum(r['fits'][method][str(1-int(r['scheduled']['direction']))]['prefix_tied'] for r in labelled)
    return out


def fit(job):
    row, profiles, duty, shuffles = job
    times = [86400+t for t in row['onsets']]
    selected = [p for p in profiles if duty_component(p['trip_id']) == duty]
    label = scheduled_direction(times[0],selected)
    result = {k:row[k] for k in ('window_id','route','date','split','period','segment_payments')}
    result['scheduled'] = label
    result['spans_scheduled_endpoint'] = times[-1] > label['scheduled_end'] if label['scheduled_end'] else None
    result['previous_prefix_winner'] = row['fits']['duty']['winner']
    result['fits'] = {
        'duty_step6': compare_relative(times,selected),
        'duty_step3': compare_relative(times,selected,max_step=3),
        'all_duties_step6': compare_relative(times,profiles),
    }
    result['shuffle_deltas'] = {'full':[],'suffix':[]}
    if label['direction'] is not None:
        d,other = label['direction'],str(1-int(label['direction']))
        for index in range(shuffles):
            seed = int(hashlib.sha256(f"{row['window_id']}/{index}".encode()).hexdigest()[:8],16)
            null = compare_relative(shuffle_gaps(times,seed),selected)
            for part in ('full','suffix'):
                result['shuffle_deltas'][part].append(null[other][part]['capped_loss']-null[d][part]['capped_loss'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--feed',type=Path,required=True)
    p.add_argument('--partitions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--shuffles',type=int,default=19)
    a = p.parse_args()
    if a.out.exists() or a.shuffles < 1:
        raise ValueError('new output and positive shuffle count required')
    manifest = json.loads((a.source_run/'manifest.json').read_text())
    if digest(a.feed) != manifest['feed_sha256'] or digest(a.source_run/'rows.json') != manifest['files']['rows.json']:
        raise ValueError('source hash mismatch')
    if digest(a.partitions/'manifest.json') != manifest['partitions_manifest_sha256']:
        raise ValueError('partition manifest mismatch')
    feed = json.loads(a.feed.read_text())
    windows = json.loads((a.source_run/'rows.json').read_text())
    duty_map = recover_duties(windows,feed,a.partitions,manifest['source_archive_sha256'])
    jobs=[]
    for row in windows:
        profiles=profiles_for_day(feed,row['route'],date.fromisoformat(row['date']))
        jobs.append((row,profiles,duty_map[row['window_id']],a.shuffles))
    a.out.mkdir(parents=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(fit,jobs):
            rows.append(row)
            if len(rows)%5 == 0:
                print('compared',len(rows),'/',len(jobs),flush=True)
    methods = ('duty_step6','duty_step3','all_duties_step6')
    heldout=[r for r in rows if r['split']=='heldout']
    summary={
        'schema_version':'direction-paired.v1','windows':len(rows),
        'all_windows':{m:summarize(rows,m) for m in methods},
        'splits':{split:{m:summarize([r for r in rows if r['split']==split],m) for m in methods}
                  for split in ('exploratory','heldout')},
        'heldout_by_route':{route:summarize([r for r in heldout if r['route']==route],'duty_step6')
                            for route in ('11','12','17')},
        'heldout_by_date':{day:summarize([r for r in heldout if r['date']==day],'duty_step6')
                           for day in sorted({r['date'] for r in heldout})},
        'heldout_no_endpoint_crossing':summarize([r for r in heldout if r['spans_scheduled_endpoint'] is False],'duty_step6'),
        'endpoint_crossing':dict(Counter(str(r['spans_scheduled_endpoint']) for r in heldout)),
        'geographic_accuracy_measured':False,
        'shuffles_per_window':a.shuffles,
    }
    write_json(a.out/'rows.json',rows)
    write_json(a.out/'summary.json',summary)
    write_json(a.out/'manifest.json',{
        'schema_version':'direction-paired.v1','timezone':'Europe/Moscow',
        'source_manifest_sha256':digest(a.source_run/'manifest.json'),
        'source_rows_sha256':digest(a.source_run/'rows.json'),'feed_sha256':digest(a.feed),
        'partitions_manifest_sha256':digest(a.partitions/'manifest.json'),
        'script_sha256':digest(Path(__file__)),
        'module_sha256':digest(Path(__file__).resolve().parents[1]/'ml/src/tramflow_ml/boarding/direction_comparison.py'),
        'files':{n:digest(a.out/n) for n in ('rows.json','summary.json')},
        'config':{'prefix':4,'suffix':4,'skip_penalty':15,'clock_weight':0,
                  'residual_weight':0,'terminal_prior':0,'max_steps':[6,3],
                  'shuffles':a.shuffles,'failure_cost':900},
        'retrospective_only':True,'direction_truth_available':False,
    })
    print('complete',len(rows),flush=True)


if __name__=='__main__':
    main()

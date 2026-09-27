import json,sys
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import session_keys
from tramflow_ml.boarding.wave_merge import adaptive_supports,coalesce
from tramflow_ml.boarding.absolute_clock import continue_profile
from tramflow_ml.boarding.clock_gtfs import profiles_for_day
from tramflow_ml.boarding.audit import digest
root=Path('/Users/cute/MosTransport2026Hack-worktrees')
base=root/'boarding-absolute-clock/ml/artifacts'
out=Path(__file__).parent
old=json.loads((base/'absolute-clock-v2/rows.json').read_text())
feedpath=base/'clock-sources/gtfs.json'
feed=json.loads(feedpath.read_text())
threshold=json.loads((root/'boarding-multiscale/ml/artifacts/multiscale-v3/thresholds.json').read_text())['adaptive']['95']
source=root/'boarding-real-alignment/ml/artifacts/boarding-date-shards'
frame=read_day(source,'2025-05-15',('17','12','11'))
frame=frame.loc[frame.success.eq('1')].copy()
at=pd.to_datetime(frame.event_at);frame['second']=at.dt.hour*3600+at.dt.minute*60+at.dt.second
frame=session_keys(frame)
prior=json.loads((root/'boarding-dense-audit/ml/artifacts/dense-onset-audit-v2/2025-05-15.json').read_text())['rows']
results={}
for route in ['17','12','11']:
 key=next(r['session_key'] for r in prior if r['route']==route and r['rank']==0 and r['method']=='gap30')
 block=frame.loc[frame.group.eq(key)].sort_values('second',kind='stable');assert not block.identity_conflict.any()
 t=block.second.to_numpy(float);devices=pd.factorize(block.device_key,sort=True)[0]
 supports=adaptive_supports(t,devices,threshold)
 detected=coalesce(t,[s[0] for s in supports],supports,first_observation_anchor=True)['times']
 row=next(r for r in old if r['route']==route and r['date']=='2025-05-15' and r['segment']=='dense_window' and r['method']=='adaptive_scan')
 profiles={p['profile_id']:p for p in profiles_for_day(feed,route,date(2025,5,15))}
 fits={}
 for name,priorfit in row['fits'].items():
  b=priorfit['best'];p=profiles[b['profile_id']];assert p['times'][b['stop_indices'][0]]==b['schedule_times'][0]
  end=p['times'][-1]+b['initial_offset']-86400
  onsets=row['onsets'][:4]+[s for s in detected if row['onsets'][3]<s<=end]
  c=continue_profile([86400+s for s in onsets[4:]],p,b,residual_weight=0 if name=='interval_only' else .25)
  ix=b['stop_indices']+(c['stop_indices'][1:] if c else [])
  errors=b['local_residuals']+(c['local_residuals'] if c else [])
  start=min(row['window_start'],p['times'][0]+b['initial_offset']-86400)-120
  stop=end+600
  edges=np.arange(np.floor(start/30)*30,np.ceil(stop/30)*30+1,30)
  counts=np.histogram(t,bins=edges)[0].tolist()
  assert sum(counts)==int(((t>=edges[0])&(t<=edges[-1])).sum())
  fits[name]={'trip':p['trip_id'],'direction':p['direction'],'offset':b['initial_offset'],'schedule':[s-86400 for s in p['times']], 'names':p['stop_names'],'onsets':onsets,'indices':ix,'errors':errors,'feasible':c is not None,'expected_end':end,'window_start':float(edges[0]),'counts':counts,'tail_onsets':[s for s in detected if end<s<=stop],'last_matched_position':ix[-1]+1,'stops_total':len(p['times']),'future_mae':float(np.mean(np.abs(c['local_residuals']))) if c and c['local_residuals'] else None,'prefix_onsets':4}
  sensitivity=[]
  for grace in [0,120,300]:
   extra=row['onsets'][:4]+[s for s in detected if row['onsets'][3]<s<=end+grace]
   ext=continue_profile([86400+s for s in extra[4:]],p,b,residual_weight=0 if name=='interval_only' else .25)
   sensitivity.append({'grace':grace,'onsets':len(extra),'times':extra,'indices':b['stop_indices']+(ext['stop_indices'][1:] if ext else []),'errors':b['local_residuals']+(ext['local_residuals'] if ext else []),'last_position':ext['stop_indices'][-1]+1 if ext else None,'mae':float(np.mean(np.abs(ext['local_residuals']))) if ext and ext['local_residuals'] else None})
  fits[name]['endpoint_sensitivity']=sensitivity
  print(route,name,'events',len(onsets),'positions',ix[-1]+1,'/',len(p['times']),'MAE',fits[name]['future_mae'],flush=True)
 full_edges=np.arange(0,86401,60);full_counts=np.histogram(t,bins=full_edges)[0];assert int(full_counts.sum())==len(t)
 results[route]={'session_first':float(t[0]),'session_last':float(t[-1]),'session_payments':len(t),'session_counts60':full_counts.tolist(),'session_onsets':detected,'fits':fits,'old_window_end':row['window_start']+1800}
(out/'full-runs-data.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
(out/'full-runs-manifest.json').write_text(json.dumps({'date':'2025-05-15','timezone':'Europe/Moscow','identity':'inferred_vehicle_pool; not independently confirmed vehicle identity','method':'existing adaptive detector on whole source session; first 4 onsets and fitted trip/offset frozen from previous experiment; continuation until shifted scheduled terminal; 10 min tail displayed unassigned','source_manifest_sha256':digest(source/'manifest.json'),'feed_sha256':digest(feedpath),'script_sha256':digest(Path(__file__)),'result_sha256':digest(out/'full-runs-data.json'),'terminal_truth':False},indent=2))

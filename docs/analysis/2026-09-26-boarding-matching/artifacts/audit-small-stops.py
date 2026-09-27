import json,sys
from tramflow_ml.boarding.multiscale import representation,peaks,SCALES,local_rate
from tramflow_ml.boarding.multiscale import controls as generate_controls
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from tramflow_ml.boarding.partitions import read_day
from tramflow_ml.boarding.real import session_keys
from tramflow_ml.boarding.wave_merge import adaptive_supports,coalesce
from tramflow_ml.boarding.absolute_clock import continue_profile,fit_profiles
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
thresholds=json.loads((root/'boarding-multiscale/ml/artifacts/multiscale-v3/thresholds.json').read_text())
previous=json.loads((out/'full-runs-data.json').read_text())
def signal_times(t,dev):
 r=representation(t,dev);at=r['times'];best=np.argmax(r['sync'][:,:4],axis=1)
 sync=peaks(at,r['sync'][np.arange(len(at)),best],SCALES[best],thresholds['sync'])
 before=np.searchsorted(t,at)-np.searchsorted(t,at-40)
 ratios=(r['counts'][:,:4]+.5)/(before[:,None]*SCALES[None,:4]/40+.5)
 ratios=np.where(r['counts'][:,:4]>=3,ratios,0)
 rb=np.argmax(ratios,axis=1);rise=peaks(at,ratios[np.arange(len(at)),rb],SCALES[rb],4)
 out={}
 for name,ids in [('synchrony',sync),('rise',rise)]:
  ss=adaptive_supports(t,dev,thresholds['adaptive']['90'])
  out[name]=coalesce(t,at[ids],ss)['times']
 out['density']=[a for a,b in adaptive_supports(t,dev,thresholds['adaptive']['90'])]
 return out
results={}
for route,d in previous.items():
 key=next(r['session_key'] for r in prior if r['route']==route and r['rank']==0 and r['method']=='gap30')
 block=frame.loc[frame.group.eq(key)].sort_values('second',kind='stable');assert not block.identity_conflict.any()
 t=block.second.to_numpy(float);dev=pd.factorize(block.device_key,sort=True)[0]
 rep=representation(t,dev);supports=adaptive_supports(t,dev,threshold)
 signals=signal_times(t,dev)
 def details(at):
  counts=[int(((t>=at)&(t<at+w)).sum()) for w in [2,3,5,10]]
  devices=[int(len(np.unique(dev[(t>=at)&(t<at+w)]))) for w in [2,3,5,10]]
  before=int(((t>=at-40)&(t<at)).sum())
  return {'time':at,'counts':counts,'devices':devices,'before40':before,'rise10':(counts[-1]+.5)/(before/4+.5),'expected10':float(local_rate(t,np.array([at]))[0]*10)}
 d['signals']={k:[details(a) for a in vals] for k,vals in signals.items()}
 d['signal_controls']=[]
 for seed in [20260926,20260927,20260928]:
  nt,nd=generate_controls(t,dev,seed,'device_shift');d['signal_controls'].append(signal_times(nt,nd))
 candidate_methods={};candidate_ends={}
 for method in ['synchrony','rise']:
  for at in signals[method]:
   candidate_methods.setdefault(at,[]).append(method);candidate_ends[at]=at+10

 for j,width in enumerate([2,3,5,10]):
  ix=peaks(rep['times'],rep['scan'][:,j],np.full(len(rep['times']),width),thresholds[f'scan{width}'])
  for i in ix:
   x=float(rep['times'][i]);candidate_methods.setdefault(x,[]).append(str(width)+'s');candidate_ends[x]=max(candidate_ends.get(x,x),x+width)
 for a,b in adaptive_supports(t,dev,thresholds['adaptive']['90']):
  candidate_methods.setdefault(a,[]).append('adaptive90');candidate_ends[a]=max(candidate_ends.get(a,a),b)
 strong=set(d['session_onsets']) | {x for f in d['fits'].values() for x in f['onsets'][:4]}
 merged=coalesce(t,sorted(set(candidate_methods)|strong),adaptive_supports(t,dev,thresholds['adaptive']['90']))
 grouped={}
 for group in merged['groups']:
  if any(x in strong for x in group['members']):continue
  support_end=max(candidate_ends.get(x,x) for x in group['members'])
  if any(group['time']<=x<support_end for x in strong):continue
  candidate_ends[group['time']]=support_end
  grouped[group['time']]=sorted({m for x in group['members'] for m in candidate_methods.get(x,[])})
 candidates=[]
 for at,methods in sorted(grouped.items()):
  if any(a<=at<b for a,b in supports):continue
  mask=(t>=at)&(t<at+10);n=int(mask.sum());devices=int(len(np.unique(dev[mask])))
  if n<2:continue
  candidates.append({'time':at,'support_end':candidate_ends[at],'n10':n,'devices10':devices,'methods':methods,'gap_before':float(at-t[np.searchsorted(t,at)-1]) if at>t[0] else None})
 original=next(r for r in old if r['route']==route and r['date']=='2025-05-15' and r['segment']=='dense_window' and r['method']=='adaptive_scan')
 profiles={p['profile_id']:p for p in profiles_for_day(feed,route,date(2025,5,15))}
 for name,f in d['fits'].items():
  b=original['fits'][name]['best'];p=profiles[b['profile_id']]
  for case in f['endpoint_sensitivity']:
   times=case['times'];relative=continue_profile([s+86400 for s in times[4:]],p,b,residual_weight=0)
   case['relative']={'indices':b['stop_indices']+(relative['stop_indices'][1:] if relative else []),'errors':b['local_residuals']+(relative['local_residuals'] if relative else []),'mae':float(np.mean(np.abs(relative['local_residuals']))) if relative and relative['local_residuals'] else None}
   for mode in ['mixed','relative']:
    target=case if mode=='mixed' else case['relative'];indices=target['indices'];observed=times[:len(indices)]
    matched=dict(zip(indices,observed));weak=[];used=set();controls={30:[],60:[]}
    for stop in range(indices[0],len(f['schedule'])):
     if stop in matched:continue
     left=max((i for i in indices if i<stop),default=None);right=min((i for i in indices if i>stop),default=None)
     if left is None:continue
     expected_left=matched[left]+f['schedule'][stop]-f['schedule'][left]
     expected_right=matched[right]-(f['schedule'][right]-f['schedule'][stop]) if right is not None else None
     choices=[]
     for cand in candidates:
      at=cand['time']
      if at<=matched[left] or (right is not None and at>=matched[right]) or at in used or (weak and at<weak[-1]['support_end']):continue
      le=at-expected_left;re=at-expected_right if right is not None else None
      if abs(le)>60 or (re is not None and abs(re)>60):continue
      choices.append((max(abs(le),abs(re or 0)),at,cand,le,re))
     if choices:
      _,at,cand,le,re=min(choices,key=lambda z:(z[0],-z[2]['devices10'],z[1]));used.add(at)
      weak.append({**cand,'stop':stop,'name':f['names'][stop],'left_error':le,'right_error':re,'two_sided':right is not None,'within30':abs(le)<=30 and re is not None and abs(re)<=30,'left':left,'right':right})
    for frac in [.25,.5,.75]:
     for tol in [30,60]:
      count=0;used_null=set();last_null=-1
      for stop in range(indices[0],indices[-1]):
       if stop in matched:continue
       left=max(i for i in indices if i<stop);right=min(i for i in indices if i>stop)
       a,z=matched[left],matched[right];el=a+f['schedule'][stop]-f['schedule'][left];er=z-(f['schedule'][right]-f['schedule'][stop]
       )
       for ci,cand in enumerate(candidates):
        if not a<cand['time']<z or ci in used_null:continue
        shifted=a+(cand['time']-a+frac*(z-a))%(z-a)
        if shifted>last_null and abs(shifted-el)<=tol and abs(shifted-er)<=tol:count+=1;used_null.add(ci);last_null=shifted;break
      controls[tol].append(count)
    target['weak']=weak;target['null_matches']=controls
    anchor=max((j for j,i in enumerate(indices) if i<len(f['schedule'])-1),default=0)
    target['relative_terminal']=observed[anchor]+f['schedule'][-1]-f['schedule'][indices[anchor]]
    assert len(used)==len(weak)
    assert all(a['time']<b['time'] for a,b in zip(weak,weak[1:]))
    assert all(not any(a<=w['time']<b for a,b in supports) for w in weak)
    assert all(w['time'] in t for w in weak)
  f['candidate_pool']=[c for c in candidates if f['window_start']<=c['time']<=f['expected_end']+600]
 terminal=d['fits']['clock_terminal180'];end=terminal['expected_end']
 terminal_check={'expected':end,'payments_within60':int(((t>=end-60)&(t<=end+60)).sum()),'payments_within120':int(((t>=end-120)&(t<=end+120)).sum())}
 after=[x for x in d['session_onsets'] if x>end][:8]
 if len(after)>=4:
  opposite=[p for p in profiles.values() if p['direction']!=terminal['direction']]
  back=fit_profiles([86400+x for x in after[:4]],opposite,start_mode='terminal',clock_weight=.2,residual_weight=.25)
  if back['best']:
   b2=back['best'];p2=profiles[b2['profile_id']];fut=continue_profile([86400+x for x in after[4:]],p2,b2) if len(after)>4 else None
   terminal_check['return_hypothesis']={'start':after[0],'times':after,'name':p2['stop_names'][0],'trip':p2['trip_id'],'offset':b2['initial_offset'],'prefix_errors':b2['local_residuals'],'future_errors':fut['local_residuals'] if fut else None}
 d['terminal_check']=terminal_check
 results[route]=d
 f=d['fits']['clock_terminal180'];c=f['endpoint_sensitivity'][0]
 print(route,'mixed weak',[(w['name'],w['time'],w['n10'],w['devices10'],w['left_error'],w['right_error']) for w in c['weak']], 'null',c['null_matches'],'relative_terminal',c['relative']['relative_terminal'],flush=True)
(out/'small-stops-data.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
(out/'small-stops-manifest.json').write_text(json.dumps({'date':'2025-05-15','timezone':'Europe/Moscow','sharp_rise':'exploratory ratio>=4 and >=3 events over 2/3/5/10s against previous40s with .5 pseudocount; sensitivity not trained; not stop truth','thresholds':'frozen Jan-Apr2025 synchrony and scan2/3/5/10 plus adaptive90; exclude inside adaptive95 support; n10>=2','matching':'fixed strong anchor identities; optional weak candidates <=60s from both neighbor projections; after last anchor one-sided only; no promotion','null':'candidate times circularly shifted inside fixed bracketing intervals by .25/.5/.75; conditional coincidence diagnostic, not accuracy','source_sha256':digest(source/'manifest.json'),'parent_sha256':digest(out/'full-runs-data.json'),'script_sha256':digest(Path(__file__)),'result_sha256':digest(out/'small-stops-data.json')},indent=2))

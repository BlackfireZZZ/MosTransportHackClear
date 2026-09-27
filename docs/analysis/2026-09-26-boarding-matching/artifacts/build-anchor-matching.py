import json
from pathlib import Path
from functools import lru_cache
from hashlib import sha256
P=Path(__file__).parent
source=P/'small-stops-data.json'
data=json.loads(source.read_text())
def waves(candidates):
 groups=[];end=-float('inf');group=-1
 for c in candidates:
  if c['time']>=end:group+=1
  groups.append(group);end=max(end,c['support_end'])
 return groups

def solve(stops,candidates,tolerance):
 groups=waves(candidates)
 @lru_cache(None)
 def dp(i,j):
  if i==len(stops) or j==len(candidates):return (0,0,())
  options=[dp(i+1,j),dp(i,j+1)]
  s,c=stops[i],candidates[j];err=c['time']-s['expected']
  if abs(err)<=tolerance and s['left_time']<c['time'] and (s['right_time'] is None or c['support_end']<=s['right_time']):
   k=j+1
   while k<len(candidates) and groups[k]==groups[j]:k+=1
   n,cost,path=dp(i+1,k);options.append((n+1,cost+abs(err),((i,j),)+path))
  return min(options,key=lambda x:(-x[0],x[1],x[2]))
 return dp(0,0)
# Cardinality must outrank greedy nearest distance; overlapping support cannot split a boarding.
st=[dict(expected=10,left_time=0,right_time=50),dict(expected=20,left_time=0,right_time=50)]
cs=[dict(time=4,support_end=5),dict(time=11,support_end=12)]
assert solve(st,cs,10)[0]==2
assert solve(st,[dict(time=10,support_end=25),dict(time=20,support_end=26)],10)[0]==1
assert solve(st,cs,0)[0]==0
assert solve(st,[dict(time=9,support_end=25),dict(time=10,support_end=11),dict(time=20,support_end=21)],10)[0]==1
out={}
for route,d in data.items():
 f=d['fits']['clock_terminal180'];case=f['endpoint_sensitivity'][0];a=case['relative'];times=case['times'];indices=a['indices'];anchors=[dict(stop=i,time=t,shift=t-f['schedule'][i],ordinal=j+1) for j,(i,t) in enumerate(zip(indices,times))]
 stops=[]
 for i,t in enumerate(f['schedule']):
  left=next((x for x in reversed(anchors) if x['stop']<i),None)
  right=next((x for x in anchors if x['stop']>i),None)
  anchor=next((x for x in anchors if x['stop']==i),None)
  expected=left['time']+t-f['schedule'][left['stop']] if left else t+f['offset']
  stops.append(dict(stop=i,name=f['names'][i],expected=expected,scheduled=t,anchor=anchor,left_time=left['time'] if left else -1,right_time=right['time'] if right else None,source_anchor=left['ordinal'] if left else None))
 pool=[{**c,'wave':g} for c,g in zip(f['candidate_pool'],waves(f['candidate_pool']))];missing=[s for s in stops if s['anchor'] is None and s['source_anchor'] is not None]
 matches={}
 for tol in [15,30,45,60,90]:
  n,cost,path=solve(missing,pool,tol)
  pairs=[dict(stop=missing[i]['stop'],candidate=j,error=pool[j]['time']-missing[i]['expected']) for i,j in path]
  assert len({pool[m['candidate']]['wave'] for m in pairs})==n
  assert n==len(pairs)==len({m['stop'] for m in pairs})==len({m['candidate'] for m in pairs})
  assert all(pool[b['candidate']]['time']>=pool[a['candidate']]['support_end'] for a,b in zip(pairs,pairs[1:]))
  assert all(abs(m['error'])<=tol for m in pairs)
  matches[str(tol)]=dict(pairs=pairs,count=n,mae=cost/n if n else None)
 assert list(m['count'] for m in matches.values())==sorted(m['count'] for m in matches.values())
 # Every forecast is frozen before weak matching; translating all timestamps must preserve pair choices.
 translated_stops=[{**s,'expected':s['expected']+420,'left_time':s['left_time']+420,'right_time':s['right_time']+420 if s['right_time'] is not None else None} for s in missing]
 translated_pool=[{**c,'time':c['time']+420,'support_end':c['support_end']+420} for c in pool]
 assert solve(missing,pool,60)==solve(translated_stops,translated_pool,60)
 out[route]=dict(stops=stops,anchors=anchors,candidates=pool,matches=matches,counts=f['counts'],start=f['window_start'],tail=f['tail_onsets'],terminal=stops[-1]['expected'],fixed_terminal=f['expected_end'],date='2025-05-15',trip=f['trip'])
 print(route,{k:v['count'] for k,v in matches.items()},flush=True)
(P/'anchor-matching-data.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
(P/'anchor-matching-manifest.json').write_text(json.dumps({'source_sha256':sha256(source.read_bytes()).hexdigest(),'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'result_sha256':sha256((P/'anchor-matching-data.json').read_bytes()).hexdigest(),'date':'2025-05-15','timezone':'Europe/Moscow','contract':'Frozen strong anchors, causal previous-anchor schedule projection; optional weak matches maximize cardinality then minimize total absolute timing error; monotone, one-to-one, nonoverlapping supports, no weak clock updates; terminal payments optional','checks':'cardinality counterexample; overlapping support veto; no-match boundary; all real pairs unique/ordered/in tolerance; counts monotone across tolerance; time-shift invariance'},indent=2))

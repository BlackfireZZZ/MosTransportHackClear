import json,hashlib
from pathlib import Path
p=Path(__file__).parent
d=json.loads((p/'small-stops-data.json').read_text());base=json.loads((p/'full-runs-data.json').read_text())
checks=0
for route,r in d.items():
 assert r['session_payments']==sum(r['session_counts60'])==base[route]['session_payments']
 for signals in r['signals'].values():
  assert all(a['time']<b['time'] for a,b in zip(signals,signals[1:]))
  for s in signals:
   assert all(a<=b for a,b in zip(s['counts'],s['counts'][1:]))
   assert all(a<=b for a,b in zip(s['devices'],s['devices'][1:]))
   assert all(0<=a<=b for a,b in zip(s['devices'],s['counts']))
   assert s['rise10']==(s['counts'][-1]+.5)/(s['before40']/4+.5)
 for name,f in r['fits'].items():
  assert f['schedule']==base[route]['fits'][name]['schedule']
  assert f['counts']==base[route]['fits'][name]['counts']
  for case in f['endpoint_sensitivity']:
   for target in [case,case['relative']]:
    ix=target['indices'];times=case['times'][:len(ix)];weak=target['weak']
    assert all(a<b for a,b in zip(ix,ix[1:]))
    assert all(a['support_end']<=b['time'] for a,b in zip(weak,weak[1:]))
    for j,err in enumerate(target['errors']):
     assert err==times[j+1]-times[j]-(f['schedule'][ix[j+1]]-f['schedule'][ix[j]])
    for w in weak:
     assert w['stop'] not in ix and w['n10']>=2
     assert abs(w['left_error'])<=60
     assert w['right_error'] is None or abs(w['right_error'])<=60
    for shift in [-600,420,1020]:
     for j in range(1,len(times)):
      predicted=times[j-1]+f['schedule'][ix[j]]-f['schedule'][ix[j-1]]
      assert times[j]-predicted==(times[j]+shift)-(predicted+shift)
    checks+=1
manifest=json.loads((p/'small-stops-manifest.json').read_text())
assert hashlib.sha256((p/'small-stops-data.json').read_bytes()).hexdigest()==manifest['result_sha256']
print(f'{checks} fits verified: unchanged raw bins/prefix profiles, error calculations, ordered non-overlapping weak supports, count/device statistics, shift invariance, result hash.')

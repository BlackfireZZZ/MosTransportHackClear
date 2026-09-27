import json
from pathlib import Path
from datetime import date
import numpy as np
from tramflow_ml.boarding.composition import as_of,compose_route
from tramflow_ml.boarding.historical import load_historical_catalog
from tramflow_ml.boarding.schedule_intervals import estimate_schedule_edges
from tramflow_ml.boarding.timing_v2 import edge_durations
from tramflow_ml.boarding.dense_audit import align
source=Path('/Users/cute/MosTransport2026Hack-worktrees/boarding-real-alignment/ml/artifacts')
folder=Path('/Users/cute/.codex/visualizations/2026/09/26/01a0dd42-a56a-7ec1-8d69-3d6bad11ee6e')
data=json.loads((folder/'tram-examples-data.json').read_text())
patterns=load_historical_catalog(source/'real-sources/osm',source.parent.parent/'data/tram_graph.json',as_of(date(2025,5,15)))
schedules=json.loads((source/'real-sources/timetables/timetables.json').read_text())
results=[]
for w in data:
 evidence=compose_route(patterns,schedules,w['route'],date(2025,5,15));estimates=estimate_schedule_edges(evidence);edges,basis=edge_durations(evidence,estimates);mask=np.array([v['boardable'] for v in evidence.visits])
 result={'route':w['route'],'start':w['start'],'end':w['end'],'schedule_edges':basis.count('inferred_schedule_interval'),'total_edges':len(edges),'applicable_pattern':evidence.applicable_pattern,'edge_estimates':estimates,'methods':{}}
 for method in ['consensus3_5s','adaptive_scan','feature_gmm']:
  gaps=np.diff(w['methods'][method]);fit=align(gaps,edges,mask,scales=(1.,));direct=align(gaps,edges,mask,max_step=1,scales=(1.,));flex=align(gaps,edges,mask)
  rng=np.random.default_rng(20260926);null=[align(gaps,rng.permutation(edges),mask,scales=(1.,))['mae_seconds'] for _ in range(9)]
  intervals=[]
  for i,k in enumerate(fit['steps']):
   states=[(fit['path'][i]+j)%len(edges) for j in range(k)]
   intervals.append({'observed':float(gaps[i]),'expected':fit['predicted_seconds'][i],'steps':k,'sources':[basis[s] for s in states]})
  result['methods'][method]={'intervals':intervals,'fit':fit,'consecutive_mae':direct['mae_seconds'],'rescaled_fit':flex,'shuffled_mae':null,'shuffled_median':float(np.median(null))}
  print(w['route'],method,'n',len(gaps),'mae',round(fit['mae_seconds'],1),'within30',round(fit['within_30'],2),'skip',fit['skipped_visits'],'phases',fit['competing_phases_within_10s'],'null',round(np.median(null),1),'free',round(flex['mae_seconds'],1),flex['scale'])
 print('SOURCE',w['route'],result['schedule_edges'],len(edges));results.append(result)
(folder/'displayed-window-schedule-check.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

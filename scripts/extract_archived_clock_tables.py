"""Extract source schedules without claiming the capture date is service validity."""
import argparse,csv,hashlib,html,io,json,re,zipfile,collections
from pathlib import Path

def clocks(v):
 h,m,s=map(int,v.split(':'));return h*3600+m*60+s

def gtfs(path):
 z=zipfile.ZipFile(path)
 def rows(n):return csv.DictReader(io.TextIOWrapper(z.open(n),encoding='utf-8-sig'))
 routes={x['route_id']:x for x in rows('routes.txt') if x['route_short_name'] in ['17','12','11'] and x['route_type']=='0'}
 calendar={x['service_id']:x for x in rows('calendar.txt')};trips={x['trip_id']:dict(x,route_ref=routes[x['route_id']]['route_short_name'],stop_times=[]) for x in rows('trips.txt') if x['route_id'] in routes}
 stops={x['stop_id']:x for x in rows('stops.txt')}
 for x in rows('stop_times.txt'):
  if x['trip_id'] not in trips:continue
  s=stops[x['stop_id']]
  trips[x['trip_id']]['stop_times'].append(dict(stop_sequence=int(x['stop_sequence']),stop_id=x['stop_id'],stop_name=s['stop_name'],lat=float(s['stop_lat']),lon=float(s['stop_lon']),arrival_seconds=clocks(x['arrival_time']),departure_seconds=clocks(x['departure_time']),pickup_type=x.get('pickup_type',''),drop_off_type=x.get('drop_off_type','')))
 anomalies=collections.Counter()
 for t in trips.values():
  t['stop_times'].sort(key=lambda x:x['stop_sequence']);ss=t['stop_times']
  for a,b in zip(ss,ss[1:]):
   gap=b['arrival_seconds']-a['departure_seconds'];anomalies['negative_edges']+=gap<0;anomalies['zero_edges']+=gap==0
  t['calendar']=calendar[t['service_id']]
 return dict(schema='historical-clock-source-research-v1',source_kind='community_gtfs_retrospective_calendar',capture_date='2026-06-30',source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),source_url='https://files.mobilitydatabase.org/mdb-3226/mdb-3226-202606301640/mdb-3226-202606301640.zip',calendar_dates_available='calendar_dates.txt' in z.namelist(),calendar_dates=list(rows('calendar_dates.txt')) if 'calendar_dates.txt' in z.namelist() else [],timezone='Europe/Moscow',trips=list(trips.values()),anomalies=dict(anomalies),caveat='Capture is 2026; service applicability is declared in calendar, not evidence of a contemporaneous 2025 capture.')

def text(s):return html.unescape(re.sub('<[^>]+>',' ',s)).strip()
def archive(path,route,capture,url):
 s=Path(path).read_text();tables=[]
 for tid,body in re.findall(r'<table[^>]*id="(table\d+)"[^>]*>(.*?)</table>',s,re.S):
  head=re.search(r'<thead>(.*?)</thead>',body,re.S)[1];names=[text(v) for v in re.findall(r'<th\b[^>]*>(.*?)</th>',head,re.S)][1:];rows=[]
  for attr,row in re.findall(r'<tr\b([^>]*)>(.*?)</tr>',body,re.S):
   cells=[text(x) for x in re.findall(r'<td\b[^>]*>(.*?)</td>',row,re.S)]
   if not cells:continue
   times=[clocks(x+':00') if re.fullmatch(r'\d{2}:\d{2}',x) else None for x in cells[1:]]
   negatives=[i for i,(a,b) in enumerate(zip(times,times[1:])) if a is not None and b is not None and b<a]
   zeros=[i for i,(a,b) in enumerate(zip(times,times[1:])) if a is not None and b is not None and a==b]
   rows.append(dict(service_label=cells[0],source_row_attributes=attr,seconds=times,width_matches=len(times)==len(names),negative_edges=negatives,zero_edges=zeros))
  tables.append(dict(table_id=tid,stops=names,rows=rows))
 return dict(schema='historical-clock-source-research-v1',source_kind='archived_third_party_timetable',route_ref=route,capture_date=capture,source_url=url,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),timezone='Europe/Moscow',service_validity='unknown',tables=tables,caveat='Rows preserve original clocks; negative edges are quarantined, not silently unwrapped; captured schedule is not proof of operation or exact service validity.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['gtfs','archive']);p.add_argument('input');p.add_argument('output');p.add_argument('--route');p.add_argument('--capture');p.add_argument('--url');a=p.parse_args();d=gtfs(a.input) if a.kind=='gtfs' else archive(a.input,a.route,a.capture,a.url);Path(a.output).write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')));print(a.output,len(json.dumps(d)),d.get('anomalies'))

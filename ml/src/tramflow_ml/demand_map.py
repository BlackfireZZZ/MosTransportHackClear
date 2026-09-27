"""Offline geographic boarding forecast with explicit unmapped demand and no remote assets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tramflow_ml.complete_features import STOP_KEYS, catalog_features, digest

TEMPLATE = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TramFlow · прогноз посадок</title><style>
:root{font-family:Inter,Arial,sans-serif;color:#1f2529;background:#f5f4ef;font-size:16px}
body{max-width:1450px;margin:0 auto;padding:24px}h1{font-size:26px;margin:0 0 12px;font-weight:600}
p{line-height:1.5;max-width:1100px}label{display:inline-flex;flex-direction:column;gap:6px;margin:8px 16px 12px 0}
select{min-height:40px;padding:8px;background:#fbfaf6;border:1px solid #d8dad7;border-radius:6px;font:inherit}
.panel{background:#fbfaf6;border:1px solid #d8dad7;border-radius:8px;padding:16px;margin:16px 0}
.stats{display:flex;gap:32px;flex-wrap:wrap}.num{font-variant-numeric:tabular-nums;font-size:24px}
.layout{display:grid;grid-template-columns:1.3fr 1fr;gap:16px}svg{width:100%;height:620px;background:#f1f0e9}
text{fill:#59636a;font-size:12px}circle{fill:#d9342b;fill-opacity:.55;stroke:#a8241e;stroke-width:.5}
button{font:inherit;min-height:40px;border:1px solid #d8dad7;border-radius:6px;background:#fbfaf6;padding:8px 16px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #d8dad7}
th{position:sticky;top:0;background:#fbfaf6}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.scroll{max-height:620px;overflow:auto}.tag{color:#b86e13}.note{font-size:13px;color:#59636a}
@media(max-width:850px){.layout{grid-template-columns:1fr}body{padding:12px}svg{height:440px}}
</style>
<h1>TramFlow · прогноз посадок по остановкам</h1>
<p>Оценка количества посадок за час. Направление берётся из рейса расписания;
остановка — гипотеза привязки оплаты. Это <strong>не заполненность салона</strong> и не подтверждённые GPS-координаты посадки.
Неопределённые остановки включены в общий поток и показаны отдельно.</p>
<div id="controls"><label>Месяц<select id="month"></select></label>
<label>Маршрут<select id="route"></select></label><label>Час, Москва<select id="hour"></select></label>
<label>Направление<select id="direction"><option value="all">Оба направления</option><option value="0">0 по GTFS</option><option value="1">1 по GTFS</option></select></label>
<button id="download">Скачать выбранный срез CSV</button></div>
<p class="note">Показано среднее по датам прогноза выбранного месяца. Размер круга соответствует прогнозу;
те же числа доступны в таблице. География отображается по координатам справочника, без онлайн-подложки.</p>
<div class="panel stats" aria-live="polite"><div>Всего посадок / день<br><span id="total" class="num"></span></div>
<div>Остановка определена<br><span id="mapped" class="num"></span></div>
<div>Остановка / координаты неизвестны<br><span id="unknown" class="num"></span></div></div>
<div class="layout"><div class="panel"><svg id="map" viewBox="0 0 900 650" role="img" aria-label="Географическая карта оценочных посадок"></svg></div>
<div class="panel scroll"><table><thead><tr><th>Остановка</th><th>Маршрут / направление</th><th>Посадки / день</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<p class="note" id="meta"></p><script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),q=id=>document.getElementById(id),fmt=x=>x.toLocaleString('ru-RU',{maximumFractionDigits:1});
function options(id,values,all){q(id).replaceChildren();if(all)q(id).add(new Option('Все','all'));for(const v of values)q(id).add(new Option(v,v))}
options('month',data.months,false);options('route',data.routes,true);options('hour',Array.from({length:24},(_,i)=>i),true);
const coords=data.points.filter(p=>p.lat!==null&&p.lon!==null),minLat=Math.min(...coords.map(p=>p.lat)),maxLat=Math.max(...coords.map(p=>p.lat)),minLon=Math.min(...coords.map(p=>p.lon)),maxLon=Math.max(...coords.map(p=>p.lon));
let visible=[];
function render(){
 const month=q('month').value,route=q('route').value,hour=q('hour').value,dir=q('direction').value,days=data.days[month];
 const sums=new Map();for(const r of data.values){if(r[0]!==month||(route!=='all'&&r[1]!==Number(route))||(hour!=='all'&&r[2]!==Number(hour)))continue;const p=data.points[r[3]];if(dir!=='all'&&p.direction!==dir&&p.direction!=='-1')continue;sums.set(r[3],(sums.get(r[3])||0)+r[4]/days)}
 visible=[...sums].map(([i,n])=>({...data.points[i],n})).sort((a,b)=>b.n-a.n);
 const total=visible.reduce((s,p)=>s+p.n,0),mapped=visible.filter(p=>p.lat!==null&&p.lon!==null),known=mapped.reduce((s,p)=>s+p.n,0);
 q('total').textContent=fmt(total);q('mapped').textContent=fmt(known);q('unknown').textContent=fmt(total-known);
 q('rows').replaceChildren();for(const p of visible){const tr=document.createElement('tr');for(const value of [p.name,`${p.route} / ${p.direction==='-1'?'не определено':p.direction}`,fmt(p.n)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}q('rows').append(tr)}
 q('map').replaceChildren();const ns='http://www.w3.org/2000/svg',peak=Math.max(1,...mapped.map(p=>p.n));
 for(const p of mapped){const c=document.createElementNS(ns,'circle');c.setAttribute('cx',40+820*(p.lon-minLon)/Math.max(.001,maxLon-minLon));c.setAttribute('cy',610-560*(p.lat-minLat)/Math.max(.001,maxLat-minLat));c.setAttribute('r',2+18*Math.sqrt(p.n/peak));const title=document.createElementNS(ns,'title');title.textContent=`${p.name}, маршрут ${p.route}, направление ${p.direction}: ${fmt(p.n)}`;c.append(title);q('map').append(c)}
 if(mapped.length===0){const text=document.createElementNS(ns,'text');text.setAttribute('x','40');text.setAttribute('y','60');text.textContent='Нет остановок с координатами для выбранного среза';q('map').append(text)}
 q('meta').textContent=`Модель: ${data.model}. Период: ${data.range.join(' — ')}. Часовой пояс Europe/Moscow. Прогнозные интервалы неопределённости не откалиброваны. При выборе направления неопределённый поток остаётся видимым.`;
}
for(const id of ['month','route','hour','direction'])q(id).addEventListener('change',render);
q('download').addEventListener('click',()=>{const safe=x=>{let value=String(x);if(typeof x==='string'&&/^[=+@\-\t\r]/.test(value))value='text:'+value;return `"${value.replaceAll('"','""')}"`};const csv=[['stop_id','name','route','direction','mean_daily_boardings'],...visible.map(p=>[p.id,p.name,p.route,p.direction,p.n])].map(r=>r.map(safe).join(';')).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));a.download='stop-demand.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});render();
</script></html>'''  # noqa: E501


def render_map(predictions: pd.DataFrame, catalog: pd.DataFrame, model: str) -> str:
    required = {*STOP_KEYS, "date", "hour", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError("complete stop forecast keys required")
    frame = predictions.copy()
    if frame.empty or frame[[*STOP_KEYS, "date", "hour"]].isna().any().any():
        raise ValueError("nonempty forecast with complete identities required")
    calendar = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    if calendar.isna().any() or not frame.hour.between(0, 23).all() or (frame.hour % 1).any():
        raise ValueError("integer forecast hours 0..23 required")
    if frame.duplicated([*STOP_KEYS, "date", "hour"]).any():
        raise ValueError("duplicate stop forecast key")
    if not np.isfinite(frame.prediction).all() or (frame.prediction < 0).any():
        raise ValueError("finite nonnegative forecasts required")
    geometry = catalog_features(catalog)
    names = catalog.groupby(STOP_KEYS, as_index=False).agg(name=("name", "first"))
    geometry = geometry.merge(names, on=STOP_KEYS, validate="one_to_one")
    identities = frame[STOP_KEYS].drop_duplicates().sort_values(STOP_KEYS)
    expected = len(identities) * frame.date.nunique() * frame.hour.nunique()
    if len(frame) != expected:
        raise ValueError("complete identity/date/hour grid required for daily means")
    identities = identities.merge(geometry, on=STOP_KEYS, how="left", validate="one_to_one")
    identities["point"] = range(len(identities))
    frame = frame.merge(identities[[*STOP_KEYS, "point"]], on=STOP_KEYS, validate="many_to_one")
    frame["month"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m")
    days = frame[["month", "date"]].drop_duplicates().groupby("month").size().to_dict()
    values = frame.groupby(["month", "route", "hour", "point"]).prediction.sum().reset_index()
    points = []
    for row in identities.itertuples():
        points.append({
            "id": str(row.stop_id), "route": int(row.route), "direction": str(row.direction),
            "name": str(row.name) if pd.notna(row.name) else "Остановка не определена",
            "lat": float(row.lat) if pd.notna(row.lat) else None,
            "lon": float(row.lon) if pd.notna(row.lon) else None,
        })
    data = {
        "model": model, "range": [str(min(frame.date)), str(max(frame.date))],
        "months": sorted(days), "days": {k: int(v) for k, v in days.items()},
        "routes": sorted(int(v) for v in frame.route.unique()), "points": points,
        "values": values.values.tolist(),
    }
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return TEMPLATE.replace("__DATA__", serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    forecast = pd.read_csv(args.predictions, dtype={"stop_id": str, "direction": str})
    if "day" in forecast and "date" not in forecast:
        forecast = forecast.rename(columns={"day": "date"})
    catalog = pd.read_csv(args.catalog, dtype={"stop_id": str, "direction": str})
    args.out.write_text(render_map(forecast, catalog, args.model))
    print(json.dumps({"output": str(args.out), "sha256": digest(args.out)}))


if __name__ == "__main__":
    main()

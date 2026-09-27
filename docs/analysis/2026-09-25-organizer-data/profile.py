import io
import json
import zipfile
import sys
import time
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import openpyxl

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(src)
report = {
    "archive_bytes": src.stat().st_size,
    "files": [
        {"name": n.filename, "bytes": n.file_size, "crc": n.CRC} for n in z.infolist()
    ],
}
labels = {}
for part in ["train", "test"]:
    d = pd.read_csv(z.open(f"labels/labels_day_{part}.csv"), sep=";")
    labels[part] = d
    assert not d.duplicated(["route", "date", "hour"]).any()
    assert (
        d.notna().all().all()
        and d.hour.between(0, 23).all()
        and (d.boardings >= 0).all()
    )
report["labels"] = {
    p: {
        "rows": len(d),
        "date_min": d.date.min(),
        "date_max": d.date.max(),
        "sum": int(d.boardings.sum()),
        "zeros": int((d.boardings == 0).sum()),
        "routes": sorted(d.route.unique().tolist()),
    }
    for p, d in labels.items()
}
alllabels = pd.concat(labels.values(), ignore_index=True)
alllabels["month"] = alllabels.date.str[:7]
alllabels.groupby(["route", "month"]).boardings.agg(
    ["count", "sum", "mean", "max"]
).reset_index().to_json(out / "route_month.json", orient="records", indent=2)
alllabels.groupby("date").boardings.sum().to_json(out / "daily.json", indent=2)
report["route_summary"] = (
    alllabels.groupby("route")
    .agg(
        rows=("boardings", "size"),
        total=("boardings", "sum"),
        positive_hours=("boardings", lambda x: int((x > 0).sum())),
        first=("date", "min"),
        last=("date", "max"),
    )
    .reset_index()
    .to_dict("records")
)
sub = pd.read_csv(z.open("test_submission.csv"), sep=";")
expected = pd.MultiIndex.from_product(
    [
        [1, 5, 7, 11, 12, 17, 25, 26, 28, 50],
        pd.date_range("2025-11-01", "2025-12-31").strftime("%Y-%m-%d"),
        range(24),
    ],
    names=["route", "date", "hour"],
)
assert len(sub) == 14640 and not sub.duplicated(["route", "date", "hour"]).any()
assert (
    pd.MultiIndex.from_frame(sub[["route", "date", "hour"]]).difference(expected).empty
)
assert (
    sub.prediction.notna().all()
    and np.isfinite(sub.prediction).all()
    and (sub.prediction >= 0).all()
)
report["submission"] = {
    "rows": len(sub),
    "complete_grid": True,
    "total_prediction": float(sub.prediction.sum()),
}
report["references"] = []
for n in z.namelist():
    if n.endswith(".xlsx"):
        w = openpyxl.load_workbook(
            io.BytesIO(z.read(n)), read_only=True, data_only=True
        )
        for s in w:
            rows = list(s.values)
            report["references"].append(
                {
                    "file": n,
                    "sheet": s.title,
                    "rows": len(rows),
                    "columns": s.max_column,
                    "header": list(rows[0]),
                }
            )
(out / "overview.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=str)
)
print("LABELS", json.dumps(report["labels"]), flush=True)
raw = {}
for part in ["train", "test"]:
    start = time.monotonic()
    counts = Counter()
    results = Counter()
    routes = Counter()
    places = Counter()
    types = Counter()
    dates = Counter()
    nulls = Counter()
    n = bad = 0
    min_t = None
    max_t = None
    input_years = Counter()
    columns = [
        "tran_date_time",
        "input_date_time",
        "validation_result",
        "tran_type_id",
        "place_id",
        "ngpt_route",
    ]
    for ch in pd.read_csv(
        z.open(part + ".csv"), sep=";", usecols=columns, dtype=str, chunksize=250000
    ):
        n += len(ch)
        nulls.update({k: int(v) for k, v in ch.isna().sum().items()})
        results.update(ch.validation_result.fillna("<NULL>").value_counts().to_dict())
        routes.update(ch.ngpt_route.fillna("<NULL>").value_counts().to_dict())
        places.update(ch.place_id.fillna("<NULL>").value_counts().to_dict())
        types.update(ch.tran_type_id.fillna("<NULL>").value_counts().to_dict())
        input_years.update(
            ch.input_date_time.str[:4].fillna("<NULL>").value_counts().to_dict()
        )
        t = pd.to_datetime(
            ch.tran_date_time, format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        bad += int(t.isna().sum())
        lo = t.min()
        hi = t.max()
        min_t = lo if min_t is None else min(min_t, lo)
        max_t = hi if max_t is None else max(max_t, hi)
        route = ch.ngpt_route.str.extract(r"^(\d+) трамвай$", expand=False)
        dates.update(t.dt.strftime("%Y-%m-%d").value_counts().to_dict())
        ok = ch.validation_result.eq("1") & t.notna() & route.notna()
        a = pd.DataFrame(
            {
                "route": route[ok].astype(int),
                "date": t[ok].dt.strftime("%Y-%m-%d"),
                "hour": t[ok].dt.hour,
            }
        )
        counts.update(a.groupby(["route", "date", "hour"]).size().to_dict())
        if n % 5000000 == 0:
            print(part, n, round(time.monotonic() - start, 1), flush=True)
    d = labels[part]
    lc = {(r.route, r.date, r.hour): int(r.boardings) for r in d.itertuples()}
    diff = [
        {"key": list(k), "raw": counts.get(k, 0), "label": lc.get(k)}
        for k in lc
        if counts.get(k, 0) != lc[k]
    ]
    extras = [{"key": list(k), "raw": v} for k, v in counts.items() if k not in lc]
    raw[part] = {
        "rows": n,
        "results": dict(results),
        "routes": dict(routes),
        "places": dict(places),
        "types": dict(types),
        "nulls": dict(nulls),
        "bad_event_time": bad,
        "min_time": str(min_t),
        "max_time": str(max_t),
        "input_years": dict(input_years),
        "dates": dict(sorted(dates.items())),
        "successful_grouped": sum(counts.values()),
        "label_mismatch_count": len(diff),
        "label_mismatch_examples": diff[:20],
        "extra_positive_keys": len(extras),
        "extra_positive_total": sum(e["raw"] for e in extras),
        "extra_examples": extras[:30],
        "seconds": round(time.monotonic() - start, 2),
    }
    (out / f"raw_{part}.json").write_text(
        json.dumps(raw[part], ensure_ascii=False, indent=2)
    )
    (out / f"counts_{part}.json").write_text(
        json.dumps(
            [
                {"route": k[0], "date": k[1], "hour": k[2], "count": v}
                for k, v in sorted(counts.items())
            ],
            indent=2,
        )
    )
    print(
        "FINISHED",
        part,
        json.dumps(
            {
                k: v
                for k, v in raw[part].items()
                if k not in ["dates", "extra_examples", "label_mismatch_examples"]
            }
        ),
        flush=True,
    )

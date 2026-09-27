import zipfile
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

z = zipfile.ZipFile(sys.argv[1])
out = Path(sys.argv[2])
route_ids = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
a = pd.concat(
    [
        pd.read_csv(z.open("labels/labels_day_" + p + ".csv"), sep=";")
        for p in ["train", "test"]
    ]
)
a.date = pd.to_datetime(a.date)
grid = pd.MultiIndex.from_product(
    [route_ids, pd.date_range("2025-01-01", "2025-10-31"), range(24)],
    names=["route", "date", "hour"],
)
d = a.set_index(["route", "date", "hour"]).reindex(grid).reset_index()
d["present"] = d.boardings.notna()
d.boardings = d.boardings.fillna(0)
d["dow"] = d.date.dt.dayofweek
d["month"] = d.date.dt.strftime("%Y-%m")
results = []
for origin, end in [("2025-07-01", "2025-08-31"), ("2025-09-01", "2025-10-31")]:
    origin = pd.Timestamp(origin)
    test = d[d.date.between(origin, pd.Timestamp(end))].copy()
    for weeks, method in [
        (1, "mean"),
        (4, "mean"),
        (8, "mean"),
        (4, "median"),
        (8, "median"),
    ]:
        hist = d[(d.date < origin) & (d.date >= origin - pd.Timedelta(weeks=weeks))]
        assert hist.date.max() < origin
        profile = (
            hist.groupby(["route", "dow", "hour"])
            .boardings.agg(method)
            .rename("prediction")
        )
        pred = test.merge(
            profile, on=["route", "dow", "hour"], how="left", validate="many_to_one"
        )
        assert pred.prediction.notna().all()
        pred.prediction = np.rint(pred.prediction.clip(lower=0))
        pred["ae"] = (pred.boardings - pred.prediction).abs()
        cuts = {}
        for col in ["route", "month"]:
            cuts[col] = {
                str(k): {
                    "target": float(g.boardings.sum()),
                    "mae": float(g.ae.mean()),
                    "wape": float(g.ae.sum() / g.boardings.sum())
                    if g.boardings.sum() > 0
                    else None,
                }
                for k, g in pred.groupby(col)
            }
        results.append(
            {
                "origin": str(origin.date()),
                "end": end,
                "model": f"{weeks}w_{method}_route_dow_hour",
                "n": len(pred),
                "wape": float(pred.ae.sum() / pred.boardings.sum()),
                "score": float(max(0, 1 - pred.ae.sum() / pred.boardings.sum())),
                "slices": cuts,
            }
        )
summary = {
    "zero_fill_policy": "EDA sensitivity: absent sparse-label keys treated as zero successful rows in supplied extract, NOT certified service/ingestion coverage; route5 has no label support",
    "grid_rows": len(d),
    "missing_label_keys": int((~d.present).sum()),
    "experiments": results,
    "monthly": d.groupby("month").boardings.sum().to_dict(),
    "route_month_daily_mean": d.groupby(["route", "month"])
    .boardings.sum()
    .div(d.groupby(["route", "month"]).date.nunique())
    .unstack()
    .round(2)
    .to_dict("index"),
    "zero_days": d.groupby(["route", "date"])
    .boardings.sum()
    .loc[lambda s: s == 0]
    .reset_index()
    .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d"))
    .to_dict("records"),
    "hour_totals": d.groupby("hour").boardings.sum().to_dict(),
    "dow_mean_daily": d.groupby(["date", "dow"])
    .boardings.sum()
    .reset_index()
    .groupby("dow")
    .boardings.mean()
    .to_dict(),
}
(out / "baselines.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
for r in results:
    print(r["origin"], r["model"], round(r["score"], 5))
print("MONTHLY", summary["monthly"])
print("ZERO DAYS", len(summary["zero_days"]))
print("ROUTE MONTH", summary["route_month_daily_mean"])

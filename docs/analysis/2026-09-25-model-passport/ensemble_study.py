import json
from pathlib import Path
from datetime import date
import numpy as np
from tramflow_ml.route_models.models import CANDIDATES, fit
from tramflow_ml.route_models.data import read_labels, day_index
from tramflow_ml.route_models.experiment import metric, slices
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
h = read_labels(args.archive)
rows = []
for stamp in ["2025-04-01", "2025-06-01", "2025-08-01"]:
    o = day_index(date.fromisoformat(stamp))
    truth = h.values[o : o + 61]
    preds = {
        n: fit(h, o, next(c for c in CANDIDATES if c.name == n)).predict()
        for n in ["cat_native", "cat_native_rmse", "type4", "week4"]
    }
    for other in ["cat_native_rmse", "type4", "week4"]:
        for w in [0.25, 0.5, 0.75]:
            p = np.rint((1 - w) * preds["cat_native"] + w * preds[other])
            rows.append(
                {
                    "origin": stamp,
                    "other": other,
                    "weight_other": w,
                    **metric(truth, p),
                    "slices": slices(truth, p),
                }
            )
    rows.append(
        {
            "origin": stamp,
            "other": "cat_native",
            "weight_other": 0,
            **metric(truth, preds["cat_native"]),
            "slices": slices(truth, preds["cat_native"]),
        }
    )
    print(stamp, flush=True)
ranks = []
for other, w in dict.fromkeys((r["other"], r["weight_other"]) for r in rows):
    rr = [r for r in rows if (r["other"], r["weight_other"]) == (other, w)]
    ranks.append(
        {
            "other": other,
            "weight_other": w,
            "pooled_score": 1
            - sum(r["absolute_error"] for r in rr) / sum(r["actual"] for r in rr),
            "scores": [r["score"] for r in rr],
        }
    )
ranks.sort(key=lambda r: r["pooled_score"], reverse=True)
args.output.write_text(
    json.dumps(
        {
            "source_hash": h.source_hash,
            "protocol": "9 fixed rounded convex blends on the same 3 reused development origins; no new holdout",
            "results": rows,
            "ranking": ranks,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(ranks, indent=2))

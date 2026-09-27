import json
import zipfile
import hashlib
import io
import sys
from pathlib import Path
import pandas as pd
import openpyxl

root = Path(sys.argv[2])
z = zipfile.ZipFile(sys.argv[1])
a = (
    pd.concat(
        [
            pd.DataFrame(json.loads((root / f"counts_{p}.json").read_text()))
            for p in ["train", "test"]
        ]
    )
    .groupby(["route", "date", "hour"])["count"]
    .sum()
)
b = (
    pd.concat(
        [
            pd.read_csv(z.open(f"labels/labels_day_{p}.csv"), sep=";")
            for p in ["train", "test"]
        ]
    )
    .set_index(["route", "date", "hour"])
    .boardings
)
inside = a[a.index.get_level_values("date") < "2025-11-01"]
assert inside.index.difference(b.index).empty and b.index.difference(inside.index).empty
assert (inside.sort_index() == b.sort_index()).all()
r = {
    "union_all_labels_exact_match": True,
    "keys": len(b),
    "target_sum": int(b.sum()),
    "archive_sha256": hashlib.file_digest(
        open(sys.argv[1], "rb"), "sha256"
    ).hexdigest(),
}
r["references"] = []
for n in z.namelist():
    if "справочники" not in n:
        continue
    w = openpyxl.load_workbook(io.BytesIO(z.read(n)), read_only=True, data_only=True)
    for s in w:
        rows = list(s.values)
        header_i = 0 if s.title == "Порядок_с_координатами" else 1
        df = pd.DataFrame(rows[header_i + 1 :], columns=rows[header_i])
        x = {"sheet": s.title, "data_rows": len(df), "fields": list(df.columns)}
        for col in [
            "route_short_name",
            "date",
            "actual_date",
            "start_date",
            "route_date_start",
            "direction_id",
            "capacity",
            "vehicle_capacity",
        ]:
            if col in df:
                x[col] = sorted(df[col].dropna().astype(str).unique().tolist())
        if "stop_id" in df:
            x["unique_stops"] = df.stop_id.nunique()
        if "stop_name" in df:
            x["duplicate_name_rows"] = int(df.stop_name.duplicated(keep=False).sum())
        if "stop_lat" in df:
            x["coordinate_null_rows"] = int(
                df[["stop_lat", "stop_lon"]].isna().any(axis=1).sum()
            )
        r["references"].append(x)
r["route5"] = []
for c in pd.read_csv(
    z.open("test.csv"),
    sep=";",
    usecols=["ngpt_route", "validation_result", "tran_date_time"],
    dtype=str,
    chunksize=500000,
):
    rows = c[c.ngpt_route == "5 трамвай"]
    r["route5"].extend(rows.to_dict("records"))
(root / "reconciliation.json").write_text(
    json.dumps(r, ensure_ascii=False, indent=2, default=int)
)
print(json.dumps(r, ensure_ascii=False, indent=2, default=int))

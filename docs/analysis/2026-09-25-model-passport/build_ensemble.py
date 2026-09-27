"""Build an experimental equal-weight MAE/RMSE ensemble fixed after development."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
from tramflow_ml.route_models.artifact import MOSCOW, RouteForecast
from tramflow_ml.route_models.data import read_labels, day_index
from tramflow_ml.route_models.experiment import build_artifact, metric, slices
from tramflow_ml.route_models.models import CANDIDATES, Trained, fit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
history = read_labels(args.archive)
configs = [
    next(c for c in CANDIDATES if c.name == name)
    for name in ("cat_native", "cat_native_rmse")
]
args.output.mkdir(parents=True, exist_ok=False)
selection = {
    "weights": [0.5, 0.5],
    "configs": [asdict(c) for c in configs],
    "rule": "best pooled score among nine predeclared convex blends; development reused",
    "source_hash": history.source_hash,
}
(args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
origin = day_index(date(2025, 9, 1))
predictions = [fit(history, origin, c).predict() for c in configs]
blend = np.rint((predictions[0] + predictions[1]) / 2)
truth = history.values[origin : origin + 61]
report = {
    "status": "experimental-challenger; not production-promoted",
    "validation": "September/October already inspected; diagnostic only, not an independent holdout",
    "overall": metric(truth, blend),
    "slices": slices(truth, blend),
    "single_mae_slices": slices(truth, predictions[0]),
    "october": metric(truth[30:], blend[30:]),
    "single_mae_october": metric(truth[30:], predictions[0][30:]),
}
models = [fit(history, 304, c) for c in configs]
for model in models:
    model.save(args.output / model.candidate.name)
restored = [Trained.load(args.output / c.name) for c in configs]
expected = np.rint((models[0].predict() + models[1].predict()) / 2)
actual = np.rint((restored[0].predict() + restored[1].predict()) / 2)
if not np.array_equal(expected, actual):
    raise ValueError("ensemble reload changed predictions")
artifact_data = build_artifact(restored[0], datetime.now(tz=MOSCOW)).model_dump(
    mode="json"
)
digest = hashlib.sha256(actual.tobytes()).hexdigest()[:12]
artifact_data["model_version"] = "cat-mae-rmse-50-50-" + digest
artifact_data["run_id"] = "route-ensemble-" + digest
for point, value in zip(artifact_data["points"], actual.reshape(-1), strict=True):
    point["predicted"] = float(value)
artifact = RouteForecast.model_validate(artifact_data)
(args.output / "forecast.json").write_text(artifact.model_dump_json(indent=2) + "\n")
artifact.write_submission(args.output / "submission.csv")
report.update(
    selection=selection,
    model_version=artifact.model_version,
    rows=len(artifact.points),
    reload_predictions_equal=True,
)
report["hashes"] = {
    str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(args.output.rglob("*"))
    if p.is_file()
}
(args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

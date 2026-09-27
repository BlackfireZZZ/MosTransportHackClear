"""Submit a 50%-ML ensemble of uniform and recency-weighted CatBoost corrections."""

import json
from datetime import date

import numpy as np
import overnight_catboost_daily50 as uniform
import overnight_catboost_recency as recency

from tramflow_ml.competition import (
    SUBMISSION_START,
    complete_labels,
    load_labels,
    rounded,
    validate_submission,
    write_submission,
)
from tramflow_ml.competition_cli import _verified_proof

CAP = 0.05
RECENCY_HALF_LIFE_DAYS = 60
OUTPUT = uniform.ROOT / "submission-catboost50-uniform-recency60-routeshape.csv"


def main() -> None:
    source_sha = _verified_proof(uniform.ARCHIVE, uniform.ROOT / "reconciliation.json")
    labels = complete_labels(
        load_labels(uniform.ARCHIVE),
        date(2025, 1, 1),
        date(2025, 10, 31),
        missing_as_zero=True,
    )
    future, anchor, uniform_correction, uniform_training = uniform.forecast(
        labels, SUBMISSION_START
    )
    recency_future, recency_anchor, recency_correction, recency_origins = recency.forecast(
        labels, SUBMISSION_START, RECENCY_HALF_LIFE_DAYS
    )
    if not future[["route", "date", "hour"]].equals(
        recency_future[["route", "date", "hour"]]
    ):
        raise ValueError("ensemble components have different forecast grids")
    if not np.allclose(anchor, recency_anchor):
        raise ValueError("ensemble components have different forecast anchors")
    uniform_ml = anchor * (1 + np.clip(uniform_correction, -CAP, CAP))
    recency_ml = anchor * (1 + np.clip(recency_correction, -CAP, CAP))
    prediction = 0.5 * anchor + 0.25 * uniform_ml + 0.25 * recency_ml
    output = future[["route", "date", "hour"]].copy()
    output["route"] = output.route.astype(int)
    output["hour"] = output.hour.astype(int)
    output["prediction"] = rounded(prediction)
    validate_submission(output)
    write_submission(output, OUTPUT)
    report = {
        "source_archive_sha256": source_sha,
        "cutoff": "2025-10-31",
        "target": "successful_validation_boardings_route_date_hour",
        "timezone_assumption": "Europe/Moscow civil route-date-hour keys",
        "hourly_anchor": "calendar_robust_28_routeshape",
        "mix": {"anchor": 0.5, "catboost_uniform": 0.25, "catboost_recency60": 0.25},
        "total_ml_weight": 0.5,
        "relative_correction_clip": [-CAP, CAP],
        "catboost_config": uniform.CONFIG,
        "uniform_training": uniform_training,
        "recency_training_origins": recency_origins,
        "output": str(OUTPUT),
        "output_sha256": uniform.digest(OUTPUT),
        "rows": len(output),
        "changed_rows_vs_lb_anchor": int(
            np.count_nonzero(output.prediction.to_numpy() != rounded(anchor))
        ),
    }
    (uniform.ROOT / "overnight_catboost_recency_ensemble_submission.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

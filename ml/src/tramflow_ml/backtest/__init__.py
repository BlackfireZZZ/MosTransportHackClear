"""History-aware rolling-origin backtesting over the leakage-safe feature layer."""

from tramflow_ml.backtest.config import (
    BACKTEST_CONFIG_VERSION,
    DEFAULT_EMBARGO_BUCKETS,
    DEFAULT_MINIMUM_LABEL_UNIT_RATIO,
    DEFAULT_MINIMUM_ORIGINS,
    DEFAULT_ORIGIN_STRIDE_PERIODS,
    DEFAULT_VALIDATION_PERIODS,
    BacktestConfig,
    FoldRules,
    policy_lookback_buckets,
)
from tramflow_ml.backtest.data import BacktestData
from tramflow_ml.backtest.folds import (
    DataSpan,
    FoldSet,
    build_fold_set,
    candidate_origins,
    data_span,
    fold_id,
)
from tramflow_ml.backtest.horizons import (
    bucket_ceiling,
    bucket_span,
    horizon_end,
    period_ceiling,
    period_start_of,
    period_step,
)
from tramflow_ml.backtest.manifest import (
    MANIFEST_VERSION,
    ExperimentManifest,
    ModelVersion,
    build_manifest,
)
from tramflow_ml.backtest.metrics import FoldMetrics, combine, fold_metrics
from tramflow_ml.backtest.records import (
    BACKTEST_VERSION,
    BacktestError,
    Fold,
    FoldView,
    Ineligible,
    IneligibleReason,
    Window,
)
from tramflow_ml.backtest.results import (
    PASSING_STATUS,
    BacktestOutcome,
    BacktestStatus,
    FoldResult,
    HorizonOutcome,
)
from tramflow_ml.backtest.runner import FoldData, FoldModel, fold_data_for, run_backtest

__all__ = [
    "BACKTEST_CONFIG_VERSION",
    "BACKTEST_VERSION",
    "DEFAULT_EMBARGO_BUCKETS",
    "DEFAULT_MINIMUM_LABEL_UNIT_RATIO",
    "DEFAULT_MINIMUM_ORIGINS",
    "DEFAULT_ORIGIN_STRIDE_PERIODS",
    "DEFAULT_VALIDATION_PERIODS",
    "MANIFEST_VERSION",
    "PASSING_STATUS",
    "BacktestConfig",
    "BacktestData",
    "BacktestError",
    "BacktestOutcome",
    "BacktestStatus",
    "DataSpan",
    "ExperimentManifest",
    "Fold",
    "FoldData",
    "FoldMetrics",
    "FoldModel",
    "FoldResult",
    "FoldRules",
    "FoldView",
    "FoldSet",
    "HorizonOutcome",
    "Ineligible",
    "IneligibleReason",
    "ModelVersion",
    "Window",
    "bucket_ceiling",
    "bucket_span",
    "build_fold_set",
    "build_manifest",
    "candidate_origins",
    "combine",
    "data_span",
    "fold_data_for",
    "fold_id",
    "fold_metrics",
    "horizon_end",
    "period_ceiling",
    "period_start_of",
    "period_step",
    "policy_lookback_buckets",
    "run_backtest",
]

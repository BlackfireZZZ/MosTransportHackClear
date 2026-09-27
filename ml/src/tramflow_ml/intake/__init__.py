"""Read-only profiling of a first data sample: describe it, never change it."""

from tramflow_ml.intake.columns import ColumnRef, printable
from tramflow_ml.intake.discover import StreamSchema, discover, resolve_profile
from tramflow_ml.intake.profile import (
    CANONICAL_FIELDS,
    IntakeProfile,
    builtin_profile,
    load_profile,
    profile_skeleton,
)
from tramflow_ml.intake.records import (
    FIELD_CONTRACTS,
    MIN_COVERAGE_RATIO,
    PROFILE_SCHEMA,
    REPORT_SCHEMA,
    IntakeError,
    IntakeReport,
    InvariantError,
    SchemaError,
    classify,
)
from tramflow_ml.intake.report import IntakeRequest, profile_sample, render
from tramflow_ml.intake.supportability import ObservedSpan, horizon_support, target_support

__all__ = [
    "CANONICAL_FIELDS",
    "ColumnRef",
    "FIELD_CONTRACTS",
    "MIN_COVERAGE_RATIO",
    "PROFILE_SCHEMA",
    "REPORT_SCHEMA",
    "IntakeError",
    "IntakeProfile",
    "IntakeReport",
    "IntakeRequest",
    "InvariantError",
    "ObservedSpan",
    "SchemaError",
    "StreamSchema",
    "builtin_profile",
    "classify",
    "printable",
    "discover",
    "horizon_support",
    "load_profile",
    "profile_sample",
    "profile_skeleton",
    "render",
    "resolve_profile",
    "target_support",
]

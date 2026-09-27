"""Explicit identity rules: versioned crosswalks, configurable clocks, honest outcomes."""

from tramflow_ml.identity.align import AlignmentError, align_event, align_events
from tramflow_ml.identity.catalog import (
    CanonicalCatalog,
    CanonicalPattern,
    CanonicalStop,
    CatalogError,
)
from tramflow_ml.identity.clock import LocalTimeError, adjust, align_time, localize
from tramflow_ml.identity.config import MOSCOW, AlignmentConfig, ConfigError, SourceClock
from tramflow_ml.identity.crosswalk import (
    Crosswalk,
    CrosswalkError,
    VehicleAssignment,
    identity_crosswalk,
    load_crosswalk,
)
from tramflow_ml.identity.geo import EARTH_RADIUS_METRES, GeoPoint, haversine_metres, stops_within
from tramflow_ml.identity.matching import match_event
from tramflow_ml.identity.quality import QualityReport, StreamQuality, quality_report
from tramflow_ml.identity.types import (
    AlignedEvent,
    AlignedTime,
    Ambiguous,
    IdentityError,
    LocalTimeReason,
    Matched,
    MatchKind,
    MatchResult,
    SourceEvent,
    Stale,
    Unmatched,
    UnmatchedReason,
    UnresolvedLocalTime,
)

__all__ = [
    "EARTH_RADIUS_METRES",
    "MOSCOW",
    "AlignedEvent",
    "AlignedTime",
    "AlignmentConfig",
    "AlignmentError",
    "Ambiguous",
    "CanonicalCatalog",
    "CanonicalPattern",
    "CanonicalStop",
    "CatalogError",
    "ConfigError",
    "Crosswalk",
    "CrosswalkError",
    "GeoPoint",
    "IdentityError",
    "LocalTimeError",
    "LocalTimeReason",
    "MatchKind",
    "MatchResult",
    "Matched",
    "QualityReport",
    "SourceClock",
    "SourceEvent",
    "Stale",
    "StreamQuality",
    "Unmatched",
    "UnmatchedReason",
    "UnresolvedLocalTime",
    "VehicleAssignment",
    "adjust",
    "align_event",
    "align_events",
    "align_time",
    "haversine_metres",
    "identity_crosswalk",
    "load_crosswalk",
    "localize",
    "match_event",
    "quality_report",
    "stops_within",
]

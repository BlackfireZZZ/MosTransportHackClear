"""Calendar attributes of a bucket. Pure arithmetic: no holiday, weather or event data.

``is_weekend`` is the civil weekday and is not a holiday proxy; a Russian holiday
calendar is deliberately absent until a certified one exists.
"""

from calendar import isleap, monthrange
from collections.abc import Callable, Mapping
from types import MappingProxyType

from tramflow_ml.features.periods import HOURS_PER_DAY, bucket_dates
from tramflow_ml.features.records import Bucket, FeatureError, FeatureValue, Granularity

MONTHS_PER_QUARTER = 3
FIRST_WEEKEND_WEEKDAY = 5

HOURLY_FEATURES = (
    "hour_of_day",
    "hour_of_week",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "day_of_year",
    "month_of_year",
    "bucket_hours",
)
DAILY_FEATURES = (
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "day_of_year",
    "days_in_month",
    "month_of_year",
    "is_month_start",
    "is_month_end",
    "bucket_hours",
)
MONTHLY_FEATURES = (
    "month_of_year",
    "quarter_of_year",
    "days_in_month",
    "is_leap_year",
    "bucket_days",
    "bucket_hours",
)
FEATURES_FOR_GRANULARITY: Mapping[Granularity, tuple[str, ...]] = MappingProxyType(
    {
        "hourly": HOURLY_FEATURES,
        "daily": DAILY_FEATURES,
        "monthly": MONTHLY_FEATURES,
    }
)


def calendar_feature_names(granularity: Granularity) -> tuple[str, ...]:
    try:
        return FEATURES_FOR_GRANULARITY[granularity]
    except KeyError as error:
        raise FeatureError(f"unsupported granularity {granularity!r}") from error


def calendar_features(bucket: Bucket, granularity: Granularity) -> dict[str, FeatureValue]:
    """Never missing: a calendar attribute exists for every bucket, at any cutoff."""
    try:
        build = _BUILDERS[granularity]
    except KeyError as error:
        raise FeatureError(f"unsupported granularity {granularity!r}") from error
    return build(bucket)


def _hourly(bucket: Bucket) -> dict[str, FeatureValue]:
    local = bucket.start
    return {
        "hour_of_day": float(local.hour),
        "hour_of_week": float(local.weekday() * HOURS_PER_DAY + local.hour),
        "day_of_week": float(local.weekday()),
        "is_weekend": float(local.weekday() >= FIRST_WEEKEND_WEEKDAY),
        "day_of_month": float(local.day),
        "day_of_year": float(local.timetuple().tm_yday),
        "month_of_year": float(local.month),
        "bucket_hours": bucket.elapsed_hours,
    }


def _daily(bucket: Bucket) -> dict[str, FeatureValue]:
    local = bucket.start
    days_in_month = monthrange(local.year, local.month)[1]
    return {
        "day_of_week": float(local.weekday()),
        "is_weekend": float(local.weekday() >= FIRST_WEEKEND_WEEKDAY),
        "day_of_month": float(local.day),
        "day_of_year": float(local.timetuple().tm_yday),
        "days_in_month": float(days_in_month),
        "month_of_year": float(local.month),
        "is_month_start": float(local.day == 1),
        "is_month_end": float(local.day == days_in_month),
        "bucket_hours": bucket.elapsed_hours,
    }


def _monthly(bucket: Bucket) -> dict[str, FeatureValue]:
    local = bucket.start
    return {
        "month_of_year": float(local.month),
        "quarter_of_year": float((local.month - 1) // MONTHS_PER_QUARTER + 1),
        "days_in_month": float(monthrange(local.year, local.month)[1]),
        "is_leap_year": float(isleap(local.year)),
        "bucket_days": float(len(bucket_dates(bucket))),
        "bucket_hours": bucket.elapsed_hours,
    }


_BUILDERS: Mapping[Granularity, Callable[[Bucket], dict[str, FeatureValue]]] = MappingProxyType(
    {"hourly": _hourly, "daily": _daily, "monthly": _monthly}
)

"""Weather data, target and feature engineering utilities for the Sydney Weather Intelligence models."""

from .data import HOURLY_VARIABLES, SYDNEY_LATITUDE, SYDNEY_LONGITUDE, aggregate_daily, fetch_hourly_weather
from .features import (
    CCI_FEATURES,
    MIN_HISTORY_DAYS,
    WHC_FEATURES,
    build_cci_features,
    build_whc_features,
)
from .targets import CLASS_NAMES, compute_cci, compute_whi, whi_to_class

__version__ = "2.0.0"

__all__ = [
    "HOURLY_VARIABLES",
    "SYDNEY_LATITUDE",
    "SYDNEY_LONGITUDE",
    "fetch_hourly_weather",
    "aggregate_daily",
    "compute_cci",
    "compute_whi",
    "whi_to_class",
    "CLASS_NAMES",
    "build_cci_features",
    "build_whc_features",
    "CCI_FEATURES",
    "WHC_FEATURES",
    "MIN_HISTORY_DAYS",
]
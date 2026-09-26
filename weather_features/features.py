"""Feature engineering that exactly reproduces the features used to train the CCI and WHC models."""

import numpy as np
import pandas as pd

from .targets import compute_cci, compute_whi, whi_to_class

MIN_HISTORY_DAYS = 30

CCI_FEATURES = [
    "cci", "temperature_2m", "temperature_max", "temperature_min", "relative_humidity_2m",
    "dew_point_2m", "wind_speed_10m", "cloud_cover", "precipitation", "pressure_msl",
    "shortwave_radiation_sum",
    "doy_sin", "doy_cos", "month",
    "cci_lag1", "cci_lag2", "temperature_2m_lag1", "temperature_2m_lag2",
    "precipitation_lag1", "precipitation_lag2", "pressure_msl_lag1", "pressure_msl_lag2",
    "pressure_change_1d", "temperature_change_1d",
    "cci_roll3_mean", "cci_roll7_mean", "temperature_2m_roll3_mean", "temperature_2m_roll7_mean",
    "temperature_2m_roll30_mean", "precipitation_roll3_sum", "precipitation_roll7_sum",
    "relative_humidity_2m_roll3_mean", "shortwave_radiation_sum_roll7_mean", "temperature_anomaly_30d",
]
CCI_LOG_COLUMNS = [
    "precipitation", "precipitation_lag1", "precipitation_lag2",
    "precipitation_roll3_sum", "precipitation_roll7_sum",
]

WHC_FEATURES = [
    "whi", "precipitation", "wind_gusts_max", "cloud_cover", "temperature_2m", "temperature_max",
    "relative_humidity_2m", "dew_point_2m", "pressure_msl", "shortwave_radiation_sum",
    "doy_sin", "doy_cos", "month",
    "whi_lag1", "whi_lag2", "pressure_msl_lag1", "pressure_msl_lag2",
    "wind_gusts_max_lag1", "wind_gusts_max_lag2", "cloud_cover_lag1", "cloud_cover_lag2",
    "pressure_change_1d", "pressure_change_3d", "temperature_change_1d",
    "whi_roll7_mean", "whi_roll14_mean", "whi_roll30_mean",
    "wind_gusts_max_roll7_mean", "wind_gusts_max_roll14_mean",
    "cloud_cover_roll7_mean", "cloud_cover_roll14_mean",
    "precipitation_roll7_sum", "precipitation_roll14_sum",
    "pressure_msl_roll7_mean", "relative_humidity_2m_roll7_mean",
    "shortwave_radiation_sum_roll14_mean", "temperature_2m_roll30_mean",
    "share_moderate_plus_14d", "share_moderate_plus_30d",
]
WHC_LOG_COLUMNS = ["precipitation", "precipitation_roll7_sum", "precipitation_roll14_sum"]


def _check_continuous(daily_df):
    """Raise ValueError if the daily index has gaps or duplicates, which would corrupt lag features."""
    index = pd.DatetimeIndex(daily_df.index)
    if index.has_duplicates:
        raise ValueError("Daily data contains duplicate dates.")
    expected = pd.date_range(index.min(), index.max(), freq="D")
    if len(expected) != len(index):
        raise ValueError("Daily data is missing dates; lag and rolling features need consecutive days.")


def _add_calendar(df):
    """Add cyclical day-of-year encodings and month."""
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df.index.month


def _add_rolling(df, specs):
    """Add rolling window features ending on each day, from (column, windows, aggregation) specs."""
    for col, windows, how in specs:
        for window in windows:
            df[f"{col}_roll{window}_{how}"] = df[col].rolling(window).agg(how)


def build_cci_features(daily_df):
    """Build the 34 features used by the Climate Comfort Index model.

    The first 29 days lack enough history for 30-day windows and are dropped. Rainfall features are
    log1p-transformed, matching the training pipeline.

    Args:
        daily_df (pandas.DataFrame): Consecutive daily data from aggregate_daily.

    Returns:
        pandas.DataFrame: One row per day with columns in CCI_FEATURES order.

    Raises:
        ValueError: If dates are not consecutive.
    """
    df = daily_df.sort_index().copy()
    _check_continuous(df)
    df["cci"] = compute_cci(df)
    _add_calendar(df)
    for col in ["cci", "temperature_2m", "precipitation", "pressure_msl"]:
        for lag in [1, 2]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df["pressure_change_1d"] = df["pressure_msl"].diff(1)
    df["temperature_change_1d"] = df["temperature_2m"].diff(1)
    _add_rolling(df, [
        ("cci", [3, 7], "mean"),
        ("temperature_2m", [3, 7, 30], "mean"),
        ("precipitation", [3, 7], "sum"),
        ("relative_humidity_2m", [3], "mean"),
        ("shortwave_radiation_sum", [7], "mean"),
    ])
    df["temperature_anomaly_30d"] = df["temperature_2m"] - df["temperature_2m_roll30_mean"]

    features = df[CCI_FEATURES].dropna().copy()
    features[CCI_LOG_COLUMNS] = np.log1p(features[CCI_LOG_COLUMNS])
    return features


def build_whc_features(daily_df):
    """Build the 39 features used by the Weather Hazard Category model.

    The first 29 days lack enough history for 30-day windows and are dropped. Rainfall features are
    log1p-transformed, matching the training pipeline.

    Args:
        daily_df (pandas.DataFrame): Consecutive daily data from aggregate_daily.

    Returns:
        pandas.DataFrame: One row per day with columns in WHC_FEATURES order.

    Raises:
        ValueError: If dates are not consecutive.
    """
    df = daily_df.sort_index().copy()
    _check_continuous(df)
    df["whi"] = compute_whi(df)
    df["whc"] = whi_to_class(df["whi"])
    _add_calendar(df)
    for col in ["whi", "pressure_msl", "wind_gusts_max", "cloud_cover"]:
        for lag in [1, 2]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df["pressure_change_1d"] = df["pressure_msl"].diff(1)
    df["pressure_change_3d"] = df["pressure_msl"].diff(3)
    df["temperature_change_1d"] = df["temperature_2m"].diff(1)
    _add_rolling(df, [
        ("whi", [7, 14, 30], "mean"),
        ("wind_gusts_max", [7, 14], "mean"),
        ("cloud_cover", [7, 14], "mean"),
        ("precipitation", [7, 14], "sum"),
        ("pressure_msl", [7], "mean"),
        ("relative_humidity_2m", [7], "mean"),
        ("shortwave_radiation_sum", [14], "mean"),
        ("temperature_2m", [30], "mean"),
    ])
    for window in [14, 30]:
        df[f"share_moderate_plus_{window}d"] = (df["whc"] >= 1).astype(int).rolling(window).mean()

    features = df[WHC_FEATURES].dropna().copy()
    features[WHC_LOG_COLUMNS] = np.log1p(features[WHC_LOG_COLUMNS])
    return features
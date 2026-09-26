"""Target variable formulas from the project brief: Climate Comfort Index and Weather Hazard Index/Category."""

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Extreme Risk"}


def compute_cci(df):
    """Calculate the Climate Comfort Index (0-100) for each daily row.

    Args:
        df (pandas.DataFrame): Daily data with temperature_2m, relative_humidity_2m, wind_speed_10m,
            cloud_cover and precipitation columns.

    Returns:
        pandas.Series: CCI values, where higher means more comfortable.
    """
    temp_score = (1 - (df["temperature_2m"] - 22).abs() / 20).clip(lower=0)
    humidity_score = (1 - (df["relative_humidity_2m"] - 50).abs() / 50).clip(lower=0)
    wind_score = (1 - (df["wind_speed_10m"] - 10).abs() / 40).clip(lower=0)
    cloud_score = (1 - df["cloud_cover"] / 100).clip(lower=0)
    rain_score = (1 - df["precipitation"] / 20).clip(lower=0)
    return 100 * (
        0.35 * temp_score
        + 0.20 * humidity_score
        + 0.15 * wind_score
        + 0.15 * cloud_score
        + 0.15 * rain_score
    )


def compute_whi(df):
    """Calculate the Weather Hazard Index (0-100) for each daily row.

    Args:
        df (pandas.DataFrame): Daily data with precipitation, wind_gusts_max, cloud_cover, snowfall and
            temperature_2m columns.

    Returns:
        pandas.Series: WHI values, where higher means more hazardous.
    """
    rain_hazard = (df["precipitation"] / 30).clip(upper=1)
    wind_hazard = (df["wind_gusts_max"] / 100).clip(upper=1)
    cloud_hazard = (df["cloud_cover"] / 100).clip(upper=1)
    snow_hazard = (df["snowfall"] / 15).clip(upper=1)
    temp_hazard = ((df["temperature_2m"] - 22).abs() / 25).clip(upper=1)
    return 100 * (
        0.30 * rain_hazard
        + 0.30 * wind_hazard
        + 0.20 * cloud_hazard
        + 0.10 * snow_hazard
        + 0.10 * temp_hazard
    )


def whi_to_class(whi):
    """Convert Weather Hazard Index values into Weather Hazard Category numbers.

    Args:
        whi (pandas.Series): WHI values.

    Returns:
        pandas.Series: 0 = Low Risk (<25), 1 = Moderate (25-50), 2 = High (50-75), 3 = Extreme (>=75).
    """
    return pd.cut(whi, bins=[-np.inf, 25, 50, 75, np.inf], labels=[0, 1, 2, 3], right=False).astype(int)
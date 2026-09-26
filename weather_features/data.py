"""Download hourly weather from the Open-Meteo archive API and aggregate it to daily values."""

import time

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SYDNEY_LATITUDE = -33.8688
SYDNEY_LONGITUDE = 151.2093
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snowfall",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]


def fetch_hourly_weather(start_date, end_date, latitude=SYDNEY_LATITUDE, longitude=SYDNEY_LONGITUDE,
                         retries=3, timeout=60):
    """Download hourly weather observations from the Open-Meteo historical archive.

    Args:
        start_date (str): First date to download, formatted YYYY-MM-DD.
        end_date (str): Last date to download, formatted YYYY-MM-DD.
        latitude (float): Location latitude. Defaults to Sydney.
        longitude (float): Location longitude. Defaults to Sydney.
        retries (int): Number of attempts before giving up.
        timeout (int): Seconds to wait for each request.

    Returns:
        pandas.DataFrame: One row per hour with a 'time' column and one column per weather variable.

    Raises:
        RuntimeError: If the API cannot be reached after all retries.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Australia/Sydney",
    }
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            hourly = pd.DataFrame(response.json()["hourly"])
            hourly["time"] = pd.to_datetime(hourly["time"])
            return hourly
        except (requests.RequestException, KeyError, ValueError) as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download weather data from Open-Meteo: {last_error}")


def aggregate_daily(hourly_df):
    """Aggregate hourly observations into the daily values used by both models.

    Uses the aggregations required by the project brief: daily mean for temperature, humidity, wind speed
    and cloud cover; daily total for precipitation and snowfall; daily maximum for wind gusts. A
    'hours_recorded' column counts non-missing hourly readings so incomplete days can be detected.

    Args:
        hourly_df (pandas.DataFrame): Hourly data with a 'time' column, as returned by fetch_hourly_weather.

    Returns:
        pandas.DataFrame: One row per day, indexed by date.
    """
    df = hourly_df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    daily = df.groupby("date").agg(
        temperature_2m=("temperature_2m", "mean"),
        temperature_max=("temperature_2m", "max"),
        temperature_min=("temperature_2m", "min"),
        relative_humidity_2m=("relative_humidity_2m", "mean"),
        dew_point_2m=("dew_point_2m", "mean"),
        wind_speed_10m=("wind_speed_10m", "mean"),
        wind_gusts_max=("wind_gusts_10m", "max"),
        cloud_cover=("cloud_cover", "mean"),
        precipitation=("precipitation", "sum"),
        snowfall=("snowfall", "sum"),
        pressure_msl=("pressure_msl", "mean"),
        shortwave_radiation_sum=("shortwave_radiation", "sum"),
        hours_recorded=("temperature_2m", "count"),
    )
    daily.index.name = "date"
    return daily
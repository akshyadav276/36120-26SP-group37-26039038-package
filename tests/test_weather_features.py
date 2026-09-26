"""Unit tests for the weather_features module."""

import numpy as np
import pandas as pd
import pytest

from weather_features import (
    CCI_FEATURES,
    WHC_FEATURES,
    aggregate_daily,
    build_cci_features,
    build_whc_features,
    compute_cci,
    compute_whi,
    whi_to_class,
)


def make_daily(n_days=40, seed=0):
    """Create a synthetic, consecutive daily weather table for testing."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days, freq="D", name="date")
    return pd.DataFrame({
        "temperature_2m": rng.uniform(10, 30, n_days),
        "temperature_max": rng.uniform(20, 35, n_days),
        "temperature_min": rng.uniform(5, 15, n_days),
        "relative_humidity_2m": rng.uniform(40, 90, n_days),
        "dew_point_2m": rng.uniform(5, 20, n_days),
        "wind_speed_10m": rng.uniform(5, 25, n_days),
        "wind_gusts_max": rng.uniform(20, 80, n_days),
        "cloud_cover": rng.uniform(0, 100, n_days),
        "precipitation": rng.exponential(2, n_days),
        "snowfall": np.zeros(n_days),
        "pressure_msl": rng.uniform(1005, 1030, n_days),
        "shortwave_radiation_sum": rng.uniform(1000, 8000, n_days),
    }, index=index)


def make_hourly(n_hours=48):
    """Create a synthetic hourly weather table for testing."""
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n_hours, freq="h"),
        "temperature_2m": np.full(n_hours, 20.0),
        "relative_humidity_2m": np.full(n_hours, 60.0),
        "dew_point_2m": np.full(n_hours, 12.0),
        "precipitation": np.full(n_hours, 1.0),
        "snowfall": np.zeros(n_hours),
        "cloud_cover": np.full(n_hours, 50.0),
        "pressure_msl": np.full(n_hours, 1015.0),
        "wind_speed_10m": np.full(n_hours, 10.0),
        "wind_gusts_10m": np.arange(n_hours, dtype=float),
        "shortwave_radiation": np.full(n_hours, 100.0),
    })


def test_cci_is_100_on_ideal_day():
    day = pd.DataFrame({"temperature_2m": [22.0], "relative_humidity_2m": [50.0],
                        "wind_speed_10m": [10.0], "cloud_cover": [0.0], "precipitation": [0.0]})
    assert compute_cci(day).iloc[0] == pytest.approx(100.0)


def test_cci_stays_between_0_and_100():
    extreme = pd.DataFrame({"temperature_2m": [-20.0, 50.0], "relative_humidity_2m": [0.0, 100.0],
                            "wind_speed_10m": [0.0, 150.0], "cloud_cover": [100.0, 100.0],
                            "precipitation": [200.0, 50.0]})
    cci = compute_cci(extreme)
    assert ((cci >= 0) & (cci <= 100)).all()


def test_whi_matches_hand_calculation():
    day = pd.DataFrame({"precipitation": [0.0], "wind_gusts_max": [46.8], "cloud_cover": [97.5],
                        "snowfall": [0.0], "temperature_2m": [17.391667]})
    assert compute_whi(day).iloc[0] == pytest.approx(35.3833, abs=1e-3)


def test_whi_to_class_boundaries():
    whi = pd.Series([0.0, 24.99, 25.0, 49.99, 50.0, 74.99, 75.0, 100.0])
    assert whi_to_class(whi).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_aggregate_daily_uses_brief_aggregations():
    daily = aggregate_daily(make_hourly(48))
    assert len(daily) == 2
    assert daily["precipitation"].iloc[0] == pytest.approx(24.0)
    assert daily["wind_gusts_max"].iloc[0] == pytest.approx(23.0)
    assert daily["wind_gusts_max"].iloc[1] == pytest.approx(47.0)
    assert daily["temperature_2m"].iloc[0] == pytest.approx(20.0)
    assert (daily["hours_recorded"] == 24).all()


def test_build_cci_features_columns_and_rows():
    features = build_cci_features(make_daily(40))
    assert list(features.columns) == CCI_FEATURES
    assert len(features) == 40 - 29
    assert not features.isna().any().any()


def test_build_whc_features_columns_and_rows():
    features = build_whc_features(make_daily(40))
    assert list(features.columns) == WHC_FEATURES
    assert len(features) == 40 - 29
    assert not features.isna().any().any()


def test_rainfall_features_are_log_transformed():
    daily = make_daily(40)
    features = build_cci_features(daily)
    last_day = features.index[-1]
    assert features.loc[last_day, "precipitation"] == pytest.approx(np.log1p(daily.loc[last_day, "precipitation"]))


def test_missing_dates_raise_error():
    daily = make_daily(40).drop(pd.Timestamp("2024-01-15"))
    with pytest.raises(ValueError):
        build_cci_features(daily)
    with pytest.raises(ValueError):
        build_whc_features(daily)
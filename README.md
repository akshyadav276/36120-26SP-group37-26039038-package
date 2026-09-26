# akshita-aml-26039038

Custom Python package for UTS 36120 Advanced Machine Learning Application (Spring 2026).

- **Author:** Akshita Yadav
- **Student ID:** 26039038

## Version history

| Version | Assignment | Changes |
|---|---|---|
| 1.0.0 | AT1 | `nba_draft_model`: tuned Random Forest predicting 3+ NBA seasons |
| 2.0.0 | AT2 | New `weather_features` module for the Sydney Weather Intelligence API |

## Installation

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple akshita-aml-26039038
```

## weather_features (AT2)

Reusable functions shared by the experimentation notebooks and the FastAPI deployment, so training and
production use identical logic.

| Function | Purpose |
|---|---|
| `fetch_hourly_weather(start_date, end_date)` | Download hourly Sydney weather from the Open-Meteo archive API |
| `aggregate_daily(hourly_df)` | Aggregate hourly data to daily values using the brief's rules |
| `compute_cci(df)` | Climate Comfort Index (0-100) |
| `compute_whi(df)` | Weather Hazard Index (0-100) |
| `whi_to_class(whi)` | Convert WHI into Weather Hazard Category (0-3) |
| `build_cci_features(daily_df)` | Build the 34 features used by the CCI model |
| `build_whc_features(daily_df)` | Build the 39 features used by the WHC model |

### Example

```python
from weather_features import fetch_hourly_weather, aggregate_daily, build_cci_features

hourly = fetch_hourly_weather("2025-01-01", "2025-02-15")
daily = aggregate_daily(hourly)
features = build_cci_features(daily)
print(features.tail())
```

## nba_draft_model (AT1)

```python
from nba_draft_model.model import NBADraftPredictor

predictor = NBADraftPredictor()
```

## Running tests

```
pip install pytest
python -m pytest tests -v
```
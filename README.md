# NBA Draft Career Longevity Prediction Model

A machine learning model to predict whether college basketball players will achieve sustained NBA careers (3+ seasons).

## Features

- Tuned Random Forest classifier
- AUPRC Score: 0.5470
- 97% accuracy on validation set

## Installation

```bash
pip install -e .
```

## Usage

```python
from nba_draft_model.model import NBADraftPredictor

predictor = NBADraftPredictor()
predictions = predictor.predict(X_test)
```

## Model Performance

- AUPRC: 0.5470
- ROC AUC: 0.8930
- Accuracy: 96.93%

## Author

Akshita Yadav
Student ID: 26039038
UTS 36120-26SP, Group 37
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class NBADraftPredictor:
    """
    NBA Draft Career Longevity Prediction Model
    Predicts whether college basketball players will achieve sustained NBA careers (3+ seasons)
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
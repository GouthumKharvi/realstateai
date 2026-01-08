"""
ML Engine for Procurement AI
Handles model training, prediction, anomaly detection, and time-series forecasting
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# ML Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


class MLEngine:
    """
    Machine Learning engine for procurement automation
    Supports classification, regression, anomaly detection, and forecasting
    """
    
    def __init__(self):
        """Initialize MLEngine"""
        self.model = None
        self.model_type = None
        self.feature_columns = None
        self.target_column = None
        self.is_trained = False
        self.model_metrics = {}
    
    
    def train_model(self, 
                   data: pd.DataFrame, 
                   target: str,
                   model_type: str = 'random_forest',
                   task_type: str = 'classification',
                   test_size: float = 0.2,
                   **kwargs) -> Dict:
        """
        Train ML model on provided data
        
        Args:
            data: DataFrame with features and target
            target: Target column name
            model_type: Type of model ('random_forest', 'logistic_regression', 'linear_regression')
            task_type: 'classification' or 'regression'
            test_size: Proportion of test data (default: 0.2)
            **kwargs: Additional model parameters
            
        Returns:
            Dictionary with training metrics
        """
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        X = data.drop(columns=[target])
        y = data[target]
        
        self.feature_columns = X.columns.tolist()
        self.target_column = target
        self.model_type = model_type
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if task_type == 'classification':
            metrics = self._train_classifier(X_train, X_test, y_train, y_test, model_type, **kwargs)
        else:
            metrics = self._train_regressor(X_train, X_test, y_train, y_test, model_type, **kwargs)
        
        self.is_trained = True
        self.model_metrics = metrics
        
        return metrics
    
    
    def _train_classifier(self, X_train, X_test, y_train, y_test, model_type, **kwargs):
        """Train classification model"""
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'model_type': model_type,
            'task_type': 'classification',
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        return metrics
    
    
    def _train_regressor(self, X_train, X_test, y_train, y_test, model_type, **kwargs):
        """Train regression model"""
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'model_type': model_type,
            'task_type': 'regression',
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        return metrics
    
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            data: DataFrame with features (must match training features)
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if set(data.columns) != set(self.feature_columns):
            raise ValueError(f"Features mismatch. Expected: {self.feature_columns}")
        
        data = data[self.feature_columns]
        predictions = self.model.predict(data)
        
        return predictions
    
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classification only)
        
        Args:
            data: DataFrame with features
            
        Returns:
            Array of probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        data = data[self.feature_columns]
        probabilities = self.model.predict_proba(data)
        
        return probabilities
    
    
    def detect_anomalies(self, 
                        data: pd.DataFrame,
                        contamination: float = 0.1,
                        **kwargs) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            data: DataFrame with features
            contamination: Expected proportion of outliers (default: 0.1)
            **kwargs: Additional IsolationForest parameters
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=kwargs.get('n_estimators', 100),
            max_samples=kwargs.get('max_samples', 'auto'),
            random_state=42
        )
        
        predictions = iso_forest.fit_predict(data)
        anomaly_scores = iso_forest.score_samples(data)
        
        result = data.copy()
        result['anomaly_score'] = anomaly_scores
        result['is_anomaly'] = predictions == -1
        
        result['anomaly_severity'] = 'normal'
        result.loc[result['is_anomaly'], 'anomaly_severity'] = 'medium'
        
        score_threshold = np.percentile(anomaly_scores, 5)
        result.loc[result['anomaly_score'] < score_threshold, 'anomaly_severity'] = 'high'
        
        return result
    
    
    def forecast_timeseries(self, 
                           data: pd.Series,
                           periods: int = 12,
                           method: str = 'moving_average') -> Dict:
        """
        Forecast time-series data
        
        Args:
            data: Time-series data (pandas Series with datetime index)
            periods: Number of periods to forecast
            method: Forecasting method ('moving_average', 'linear_trend', 'exponential_smoothing')
            
        Returns:
            Dictionary with historical data, forecasts, and metadata
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        if len(data) < 3:
            raise ValueError("Need at least 3 data points for forecasting")
        
        if method == 'moving_average':
            forecast = self._forecast_moving_average(data, periods)
        elif method == 'linear_trend':
            forecast = self._forecast_linear_trend(data, periods)
        elif method == 'exponential_smoothing':
            forecast = self._forecast_exponential_smoothing(data, periods)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        last_date = data.index[-1]
        freq = pd.infer_freq(data.index) or 'M'
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        
        forecast_series = pd.Series(forecast, index=future_dates)
        
        return {
            'historical': data,
            'forecast': forecast_series,
            'method': method,
            'periods': periods,
            'last_historical_value': data.iloc[-1],
            'first_forecast_value': forecast[0],
            'last_forecast_value': forecast[-1]
        }
    
    
    def _forecast_moving_average(self, data: pd.Series, periods: int, window: int = 3) -> np.ndarray:
        """Forecast using moving average"""
        window = min(window, len(data))
        last_values = data.tail(window).values
        avg = np.mean(last_values)
        forecast = np.full(periods, avg)
        return forecast
    
    
    def _forecast_linear_trend(self, data: pd.Series, periods: int) -> np.ndarray:
        """Forecast using linear trend"""
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        return forecast
    
    
    def _forecast_exponential_smoothing(self, data: pd.Series, periods: int, alpha: float = 0.3) -> np.ndarray:
        """Forecast using exponential smoothing"""
        smoothed = data.ewm(alpha=alpha, adjust=False).mean()
        last_value = smoothed.iloc[-1]
        
        if len(data) >= 2:
            trend = smoothed.iloc[-1] - smoothed.iloc[-2]
        else:
            trend = 0
        
        forecast = []
        for i in range(periods):
            forecast_value = last_value + (trend * (i + 1))
            forecast.append(forecast_value)
        
        return np.array(forecast)
    
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'metrics': self.model_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.model_metrics = model_data['metrics']
        self.is_trained = True
    
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models)
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

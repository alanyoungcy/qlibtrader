"""
Sklearn-based analysis and modeling for trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLAnalyzer:
    """
    Sklearn-based analysis and modeling for trading strategies.
    """
    
    def __init__(self):
        """Initialize ML analyzer."""
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        
        # Available model types
        self.classifier_models = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC
        }
        
        self.regressor_models = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'linear_regression': LinearRegression,
            'svr': SVR
        }
    
    def feature_engineering(self, data: pd.DataFrame, 
                          target_col: str = 'returns',
                          lookback_window: int = 20) -> pd.DataFrame:
        """
        Create features from raw OHLCV data.
        
        Args:
            data: Raw OHLCV data
            target_col: Target column for prediction
            lookback_window: Window for rolling calculations
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")
            
            self.logger.info(f"Engineering features with {lookback_window} day lookback")
            
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            df['volume_price_ratio'] = df['volume'] / df['close']
            
            # Moving averages and ratios
            for window in [5, 10, 20, 50]:
                if window <= lookback_window:
                    df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                    df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
                    df[f'ma_slope_{window}'] = df[f'ma_{window}'].diff()
            
            # Volatility features
            df['volatility_5'] = df['returns'].rolling(window=5).std()
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            
            # Volume features
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_spike'] = (df['volume'] > df['volume_ma_20'] * 1.5).astype(int)
            
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'volatility_lag_{lag}'] = df['volatility_20'].shift(lag)
            
            # Target variable (future returns)
            if target_col == 'returns':
                df['target'] = df['returns'].shift(-1)  # Next day return
            else:
                df['target'] = df[target_col].shift(-1)
            
            # Create classification target (direction)
            df['target_direction'] = np.where(df['target'] > 0, 1, 0)
            df['target_strong'] = np.where(df['target'] > df['target'].std(), 1,
                                         np.where(df['target'] < -df['target'].std(), -1, 0))
            
            # Remove rows with NaN values
            initial_len = len(df)
            df = df.dropna()
            final_len = len(df)
            
            self.logger.info(f"Feature engineering complete: {initial_len} -> {final_len} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            raise
    
    def train_classifier(self, X: pd.DataFrame, y: pd.Series, 
                        model_type: str = 'random_forest',
                        test_size: float = 0.2,
                        random_state: int = 42) -> Dict[str, Any]:
        """
        Train classification models for direction prediction.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of classifier to use
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            if model_type not in self.classifier_models:
                raise ValueError(f"Unknown classifier: {model_type}")
            
            self.logger.info(f"Training {model_type} classifier")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            model_class = self.classifier_models[model_type]
            
            # Set default parameters
            if model_type == 'random_forest':
                model = model_class(n_estimators=100, random_state=random_state, n_jobs=-1)
            elif model_type == 'logistic_regression':
                model = model_class(random_state=random_state, max_iter=1000)
            elif model_type == 'svm':
                model = model_class(random_state=random_state, probability=True)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            accuracy = model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Store model and scaler
            self.models[f'{model_type}_classifier'] = model
            self.scalers[f'{model_type}_classifier'] = scaler
            
            results = {
                'model_type': model_type,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            self.logger.info(f"Classifier training complete. Accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training classifier: {e}")
            raise
    
    def train_regressor(self, X: pd.DataFrame, y: pd.Series,
                       model_type: str = 'gradient_boosting',
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Train regression models for return prediction.
        
        Args:
            X: Feature matrix
            y: Target values
            model_type: Type of regressor to use
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            if model_type not in self.regressor_models:
                raise ValueError(f"Unknown regressor: {model_type}")
            
            self.logger.info(f"Training {model_type} regressor")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            model_class = self.regressor_models[model_type]
            
            # Set default parameters
            if model_type == 'random_forest':
                model = model_class(n_estimators=100, random_state=random_state, n_jobs=-1)
            elif model_type == 'gradient_boosting':
                model = model_class(n_estimators=100, random_state=random_state)
            elif model_type == 'linear_regression':
                model = model_class()
            elif model_type == 'svr':
                model = model_class(kernel='rbf')
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            
            # Store model and scaler
            self.models[f'{model_type}_regressor'] = model
            self.scalers[f'{model_type}_regressor'] = scaler
            
            results = {
                'model_type': model_type,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'actual': y_test.values
            }
            
            self.logger.info(f"Regressor training complete. R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training regressor: {e}")
            raise
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      model_type: str = 'random_forest',
                      cv: int = 5,
                      task: str = 'classification') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            X: Feature matrix
            y: Target values
            model_type: Type of model to use
            cv: Number of cross-validation folds
            task: 'classification' or 'regression'
            
        Returns:
            Dictionary with CV results
        """
        try:
            self.logger.info(f"Performing {cv}-fold cross-validation for {model_type}")
            
            # Select appropriate model
            if task == 'classification':
                if model_type not in self.classifier_models:
                    raise ValueError(f"Unknown classifier: {model_type}")
                model = self.classifier_models[model_type]()
                scoring = 'accuracy'
            else:
                if model_type not in self.regressor_models:
                    raise ValueError(f"Unknown regressor: {model_type}")
                model = self.regressor_models[model_type]()
                scoring = 'neg_mean_squared_error'
            
            # Perform cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            results = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores.tolist()
            }
            
            if task == 'regression':
                results['mean_score'] = -results['mean_score']  # Convert back from negative MSE
            
            self.logger.info(f"CV results: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            raise
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Generate predictions using a trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of the trained model
            
        Returns:
            Predictions array
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            self.logger.info(f"Generated {len(predictions)} predictions using {model_name}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            raise
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name: Name of the trained model
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            if not hasattr(model, 'feature_importances_'):
                self.logger.warning(f"Model {model_name} does not support feature importance")
                return {}
            
            # Get feature names
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
            
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], reverse=True))
            
            self.logger.info(f"Retrieved feature importance for {model_name}")
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df

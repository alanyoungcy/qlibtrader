"""
Qlib analysis wrapper for generating trading signals and predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib.data.dataset import Dataset
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.model.trainer import task_train
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.contrib.data.handler import Alpha158
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class QlibAnalyzer:
    """
    Wrapper for Qlib analysis and predictions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Qlib analyzer.
        
        Args:
            config: Configuration dictionary for Qlib
        """
        self.config = config or {}
        self.model = None
        self.feature_handler = None
        self.dataset = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize Qlib if available
        if QLIB_AVAILABLE:
            self._initialize_qlib()
        else:
            self.logger.warning("Qlib not available. Install qlib for full functionality.")
    
    def _initialize_qlib(self):
        """Initialize Qlib with configuration."""
        try:
            # Set up Qlib
            qlib.init(provider_uri=self.config.get('provider_uri', '~/.qlib/qlib_data/cn_data'),
                     region=REG_CN,
                     auto_mount=self.config.get('auto_mount', True))
            self.logger.info("Qlib initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Qlib: {e}")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Qlib analysis.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Processed DataFrame ready for Qlib
        """
        try:
            if not QLIB_AVAILABLE:
                self.logger.warning("Qlib not available, returning processed data without Qlib features")
                return self._basic_feature_engineering(data)
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")
            
            # Create Qlib-compatible data structure
            processed_data = data.copy()
            
            # Add symbol column if not present
            if 'symbol' not in processed_data.columns:
                # Assume single symbol if not specified
                processed_data['symbol'] = 'STOCK'
            
            # Ensure datetime index
            if not isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data.index = pd.to_datetime(processed_data.index)
            
            # Add basic features for Qlib
            processed_data = self._add_qlib_features(processed_data)
            
            self.logger.info(f"Data prepared for Qlib: {len(processed_data)} records")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return data
    
    def train_model(self, train_data: pd.DataFrame, 
                   model_type: str = 'lightgbm',
                   target_col: str = 'target') -> None:
        """
        Train Qlib model.
        
        Args:
            train_data: Training data
            model_type: Type of model to use
            target_col: Name of target column
        """
        try:
            if not QLIB_AVAILABLE:
                self.logger.warning("Qlib not available, skipping model training")
                return
            
            self.logger.info(f"Training {model_type} model with {len(train_data)} records")
            
            # Create feature handler
            self.feature_handler = Alpha158()
            
            # Prepare dataset
            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": self.feature_handler,
                    "segments": {
                        "train": ("2020-01-01", "2022-01-01"),
                        "valid": ("2022-01-01", "2023-01-01"),
                        "test": ("2023-01-01", "2024-01-01"),
                    },
                },
            }
            
            self.dataset = Dataset(dataset_config)
            
            # Initialize model
            if model_type == 'lightgbm':
                self.model = LGBModel(
                    loss="mse",
                    colsample_bytree=0.8879,
                    learning_rate=0.0421,
                    subsample=0.8789,
                    lambda_l1=205.6999,
                    lambda_l2=580.9768,
                    max_depth=8,
                    num_leaves=210,
                    num_threads=20
                )
            
            # Train model
            task_train(
                model=self.model,
                dataset=self.dataset,
                recorder_name="task",
                recorder_path="./results/",
                save_path="./results/"
            )
            
            self.logger.info("Model training completed")
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
    
    def generate_signals(self, data: pd.DataFrame, 
                        prediction_horizon: int = 5,
                        threshold: float = 0.02) -> pd.DataFrame:
        """
        Generate trading signals from data.
        
        Args:
            data: Input data
            prediction_horizon: Days ahead to predict
            threshold: Signal threshold for buy/sell decisions
            
        Returns:
            DataFrame with signals
        """
        try:
            signals_df = data.copy()
            
            if not QLIB_AVAILABLE or self.model is None:
                # Fallback to basic signal generation
                self.logger.info("Using basic signal generation")
                signals_df = self._generate_basic_signals(signals_df, threshold)
            else:
                # Use trained Qlib model for predictions
                self.logger.info("Using Qlib model for signal generation")
                signals_df = self._generate_qlib_signals(signals_df, prediction_horizon, threshold)
            
            # Add signal summary
            signals_df['signal_strength'] = abs(signals_df['signal'])
            signals_df['position'] = np.where(signals_df['signal'] > threshold, 1,
                                            np.where(signals_df['signal'] < -threshold, -1, 0))
            
            self.logger.info(f"Generated signals for {len(signals_df)} records")
            return signals_df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from trained model.
        
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if self.model is None or not hasattr(self.model, 'feature_importance_'):
                self.logger.warning("No trained model available for feature importance")
                return {}
            
            importance = self.model.feature_importance_
            feature_names = self.model.feature_name_
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _add_qlib_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Qlib-specific features to data."""
        df = data.copy()
        
        # Technical indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price-based features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
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
        
        return df
    
    def _generate_basic_signals(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Generate basic trading signals without Qlib."""
        df = data.copy()
        
        # Simple momentum signal
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Mean reversion signal
        df['bb_position'] = (df['close'] - df['ma_20']) / (df['volatility_20'] * 2)
        
        # Combine signals
        df['signal'] = (
            0.5 * np.tanh(df['momentum_5'] * 10) +  # Momentum component
            0.3 * np.tanh(-df['bb_position']) +     # Mean reversion component
            0.2 * np.tanh(df['rsi'] / 50 - 1)       # RSI component
        )
        
        return df
    
    def _generate_qlib_signals(self, data: pd.DataFrame, 
                              prediction_horizon: int, 
                              threshold: float) -> pd.DataFrame:
        """Generate signals using trained Qlib model."""
        try:
            # This would use the trained model to make predictions
            # For now, return basic signals as placeholder
            self.logger.info("Qlib signal generation not fully implemented, using basic signals")
            return self._generate_basic_signals(data, threshold)
            
        except Exception as e:
            self.logger.error(f"Error in Qlib signal generation: {e}")
            return self._generate_basic_signals(data, threshold)
    
    def _basic_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic feature engineering when Qlib is not available."""
        df = data.copy()
        
        # Add basic technical indicators
        df['returns'] = df['close'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['price_ma'] = df['close'].rolling(window=20).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Price momentum
        df['momentum'] = df['close'] / df['close'].shift(20) - 1
        
        return df

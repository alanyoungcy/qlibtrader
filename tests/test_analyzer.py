"""
Tests for the analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analyzer import QlibAnalyzer


class TestQlibAnalyzer:
    """Test cases for QlibAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = QlibAnalyzer()
        
        # Create sample OHLCV data
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2023-01-01', periods=5))
    
    def test_init(self):
        """Test analyzer initialization."""
        assert self.analyzer.config == {}
        assert self.analyzer.model is None
        assert self.analyzer.feature_handler is None
    
    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {'provider_uri': 'test_uri'}
        analyzer = QlibAnalyzer(config)
        assert analyzer.config == config
    
    def test_prepare_data(self):
        """Test data preparation."""
        prepared_data = self.analyzer.prepare_data(self.sample_data)
        
        assert not prepared_data.empty
        assert len(prepared_data) == len(self.sample_data)
        
        # Should have additional features
        assert 'returns' in prepared_data.columns
        assert 'log_returns' in prepared_data.columns
    
    def test_prepare_data_empty(self):
        """Test data preparation with empty data."""
        empty_data = pd.DataFrame()
        prepared_data = self.analyzer.prepare_data(empty_data)
        
        assert prepared_data.empty
    
    def test_prepare_data_missing_columns(self):
        """Test data preparation with missing columns."""
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102]
        })
        
        with pytest.raises(ValueError, match="Data must contain columns"):
            self.analyzer.prepare_data(incomplete_data)
    
    def test_generate_signals(self):
        """Test signal generation."""
        prepared_data = self.analyzer.prepare_data(self.sample_data)
        signals = self.analyzer.generate_signals(prepared_data)
        
        assert not signals.empty
        assert 'signal' in signals.columns
        assert 'signal_strength' in signals.columns
        assert 'position' in signals.columns
    
    def test_generate_signals_with_threshold(self):
        """Test signal generation with custom threshold."""
        prepared_data = self.analyzer.prepare_data(self.sample_data)
        signals = self.analyzer.generate_signals(prepared_data, threshold=0.1)
        
        assert not signals.empty
        # Check that position values are within expected range
        assert signals['position'].isin([-1, 0, 1]).all()
    
    def test_get_feature_importance_no_model(self):
        """Test getting feature importance without trained model."""
        importance = self.analyzer.get_feature_importance()
        assert importance == {}
    
    def test_get_feature_importance_with_model(self):
        """Test getting feature importance with trained model."""
        # Mock a model with feature importance
        mock_model = Mock()
        mock_model.feature_importance_ = np.array([0.5, 0.3, 0.2])
        mock_model.feature_name_ = ['feature1', 'feature2', 'feature3']
        
        self.analyzer.model = mock_model
        
        importance = self.analyzer.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert 'feature1' in importance
        assert importance['feature1'] == 0.5
    
    def test_train_model_no_qlib(self):
        """Test model training when Qlib is not available."""
        train_data = self.analyzer.prepare_data(self.sample_data)
        
        # Should not raise an exception even without Qlib
        self.analyzer.train_model(train_data)
        
        # Model should still be None
        assert self.analyzer.model is None
    
    def test_basic_feature_engineering(self):
        """Test basic feature engineering."""
        engineered_data = self.analyzer._basic_feature_engineering(self.sample_data)
        
        assert 'returns' in engineered_data.columns
        assert 'volume_ma' in engineered_data.columns
        assert 'price_ma' in engineered_data.columns
        assert 'volatility' in engineered_data.columns
        assert 'momentum' in engineered_data.columns
    
    def test_add_qlib_features(self):
        """Test adding Qlib-specific features."""
        features_data = self.analyzer._add_qlib_features(self.sample_data)
        
        # Should have technical indicators
        assert 'rsi' in features_data.columns
        assert 'macd' in features_data.columns
        assert 'macd_signal' in features_data.columns
        assert 'macd_histogram' in features_data.columns
        
        # Should have moving averages
        assert 'ma_5' in features_data.columns
        assert 'ma_10' in features_data.columns
        assert 'ma_20' in features_data.columns
    
    def test_generate_basic_signals(self):
        """Test basic signal generation."""
        signals_data = self.analyzer._generate_basic_signals(self.sample_data, threshold=0.02)
        
        assert 'signal' in signals_data.columns
        assert 'momentum_5' in signals_data.columns
        assert 'momentum_20' in signals_data.columns
        assert 'bb_position' in signals_data.columns
        
        # Check signal values are reasonable
        assert signals_data['signal'].min() >= -2.0
        assert signals_data['signal'].max() <= 2.0
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        rsi = self.analyzer._calculate_rsi(prices)
        
        assert len(rsi) == len(prices)
        assert rsi.iloc[0] == 50.0  # First value should be 50 (neutral)
        assert 0 <= rsi.min() <= 100
        assert 0 <= rsi.max() <= 100
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        macd, signal, histogram = self.analyzer._calculate_macd(prices)
        
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)
        
        # MACD should be difference between EMAs
        assert histogram.equals(macd - signal)
    
    def test_validate_data_format(self):
        """Test data validation."""
        # Valid data should pass
        assert self.analyzer.validate_data(self.sample_data) is True
        
        # Empty data should fail
        assert self.analyzer.validate_data(pd.DataFrame()) is False
        
        # Data with missing columns should fail
        incomplete_data = pd.DataFrame({'close': [100, 101]})
        assert self.analyzer.validate_data(incomplete_data) is False
    
    def test_error_handling(self):
        """Test error handling in various methods."""
        # Test with None input
        with pytest.raises((AttributeError, TypeError)):
            self.analyzer.prepare_data(None)
        
        # Test with invalid data types
        invalid_data = pd.DataFrame({'close': ['invalid', 'data']})
        with pytest.raises((ValueError, TypeError)):
            self.analyzer.prepare_data(invalid_data)


if __name__ == "__main__":
    pytest.main([__file__])

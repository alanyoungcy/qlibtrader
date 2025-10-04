"""
Tests for the data fetcher module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DatabentoFetcher


class TestDatabentoFetcher:
    """Test cases for DatabentoFetcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = DatabentoFetcher(api_key="test_key")
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        fetcher = DatabentoFetcher(api_key="test_key")
        assert fetcher.client is not None
    
    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {'DATABENTO_API_KEY': 'env_key'}):
            fetcher = DatabentoFetcher()
            assert fetcher.client is not None
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                DatabentoFetcher()
    
    def test_validate_symbols(self):
        """Test symbol validation."""
        symbols = ["AAPL", "MSFT", "INVALID_SYMBOL"]
        valid_symbols = self.fetcher.validate_symbols(symbols)
        
        # Should return valid symbols only
        assert "AAPL" in valid_symbols
        assert "MSFT" in valid_symbols
        assert "INVALID_SYMBOL" not in valid_symbols
    
    def test_standardize_columns(self):
        """Test column standardization."""
        # Create test data with Databento column names
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000000, 1100000, 1200000],
            'ts_event': pd.date_range('2023-01-01', periods=3)
        })
        
        standardized = self.fetcher._standardize_columns(data)
        
        # Check that columns are renamed
        assert 'timestamp' in standardized.columns or standardized.index.name == 'timestamp'
        assert 'open' in standardized.columns
        assert 'close' in standardized.columns
    
    @patch('core.data_fetcher.db.Historical')
    def test_fetch_historical_success(self, mock_historical):
        """Test successful historical data fetching."""
        # Mock the client and its methods
        mock_client = Mock()
        mock_client.timeseries.get_range.return_value = Mock()
        mock_client.timeseries.get_range.return_value.to_df.return_value = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [103, 104],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        mock_historical.return_value = mock_client
        self.fetcher.client = mock_client
        
        # Test fetch
        result = self.fetcher.fetch_historical(
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-02"
        )
        
        assert not result.empty
        assert len(result) == 2
        assert 'close' in result.columns
    
    def test_fetch_historical_invalid_dates(self):
        """Test fetch with invalid date format."""
        with pytest.raises(Exception):
            self.fetcher.fetch_historical(
                symbols=["AAPL"],
                start_date="invalid-date",
                end_date="2023-01-02"
            )
    
    def test_cache_data(self):
        """Test data caching functionality."""
        data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000000, 1100000, 1200000]
        })
        
        with patch('builtins.open', Mock()), \
             patch('pandas.DataFrame.to_parquet', Mock()), \
             patch('os.makedirs', Mock()):
            
            # Should not raise an exception
            self.fetcher.cache_data(data, "test_path.parquet")
    
    def test_load_cached_data(self):
        """Test loading cached data."""
        with patch('os.path.exists', return_value=True), \
             patch('pandas.read_parquet', return_value=pd.DataFrame({'close': [100]})):
            
            result = self.fetcher.load_cached_data("test_path.parquet")
            assert not result.empty
    
    def test_load_cached_data_not_found(self):
        """Test loading non-existent cached data."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                self.fetcher.load_cached_data("nonexistent_path.parquet")
    
    def test_get_available_symbols(self):
        """Test getting available symbols."""
        symbols = self.fetcher.get_available_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols  # Should include common symbols


if __name__ == "__main__":
    pytest.main([__file__])

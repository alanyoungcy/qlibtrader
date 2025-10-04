"""
Tests for the backtester module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtester import BacktestEngine


class TestBacktestEngine:
    """Test cases for BacktestEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backtester = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # Create sample signals
        self.sample_signals = pd.DataFrame({
            'signal': [0, 1, 0, -1, 0],
            'signal_strength': [0, 0.5, 0, 0.3, 0],
            'position': [0, 1, 0, -1, 0]
        }, index=self.sample_data.index)
    
    def test_init(self):
        """Test backtester initialization."""
        assert self.backtester.initial_capital == 100000
        assert self.backtester.commission == 0.001
        assert self.backtester.slippage == 0.0005
        assert self.backtester.results == {}
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        backtester = BacktestEngine()
        assert backtester.initial_capital == 100000.0
        assert backtester.commission == 0.001
        assert backtester.slippage == 0.0005
    
    def test_run_simple_backtest(self):
        """Test simple backtest execution."""
        strategy_params = {
            'max_position_size': 1.0,
            'position_size_method': 'fixed'
        }
        
        results = self.backtester._run_simple_backtest(
            self.sample_data,
            self.sample_signals,
            strategy_params
        )
        
        assert isinstance(results, dict)
        assert 'portfolio_values' in results
        assert 'trades' in results
        assert 'metrics' in results
        
        # Check portfolio values
        portfolio_df = results['portfolio_values']
        assert not portfolio_df.empty
        assert 'portfolio_value' in portfolio_df.columns
        assert 'cash' in portfolio_df.columns
        assert 'position' in portfolio_df.columns
    
    def test_run_simple_backtest_empty_data(self):
        """Test backtest with empty data."""
        empty_data = pd.DataFrame()
        empty_signals = pd.DataFrame()
        
        with pytest.raises((KeyError, IndexError)):
            self.backtester._run_simple_backtest(empty_data, empty_signals, {})
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Create sample portfolio values
        portfolio_values = pd.DataFrame({
            'portfolio_value': [100000, 101000, 102000, 101500, 103000]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # Mock results
        self.backtester.results = {
            'portfolio_values': portfolio_values,
            'trades': pd.DataFrame({
                'value': [1000, -500, 1500]
            })
        }
        
        metrics = self.backtester.calculate_metrics(portfolio_values)
        
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # Check that total return is calculated correctly
        expected_return = (103000 - 100000) / 100000
        assert abs(metrics['total_return'] - expected_return) < 0.001
    
    def test_calculate_metrics_empty_data(self):
        """Test metrics calculation with empty data."""
        empty_portfolio = pd.DataFrame()
        metrics = self.backtester.calculate_metrics(empty_portfolio)
        assert metrics == {}
    
    def test_generate_report(self):
        """Test report generation."""
        # Mock results
        self.backtester.results = {
            'portfolio_values': pd.DataFrame({
                'portfolio_value': [100000, 101000, 102000]
            }),
            'trades': pd.DataFrame({
                'value': [1000, -500]
            }),
            'metrics': {
                'total_return': 0.02,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.05,
                'final_portfolio_value': 102000
            }
        }
        
        report = self.backtester.generate_report()
        
        assert isinstance(report, pd.DataFrame)
        assert not report.empty
        assert 'Metric' in report.columns
        assert 'Value' in report.columns
        
        # Check that key metrics are included
        report_dict = dict(zip(report['Metric'], report['Value']))
        assert 'Total Return' in report_dict
        assert 'Sharpe Ratio' in report_dict
    
    def test_generate_report_no_results(self):
        """Test report generation with no results."""
        self.backtester.results = {}
        report = self.backtester.generate_report()
        assert report.empty
    
    def test_plot_results(self):
        """Test plot generation."""
        # Mock results with portfolio values
        portfolio_values = pd.DataFrame({
            'portfolio_value': [100000, 101000, 102000, 101500, 103000]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        self.backtester.results = {
            'portfolio_values': portfolio_values,
            'trades': pd.DataFrame()
        }
        
        # Mock plotly
        with patch('core.backtester.go') as mock_go, \
             patch('core.backtester.make_subplots') as mock_subplots:
            
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            
            fig = self.backtester.plot_results()
            
            # Should return the mocked figure
            assert fig == mock_fig
    
    def test_plot_results_empty(self):
        """Test plot generation with empty results."""
        self.backtester.results = {}
        
        with patch('core.backtester.go') as mock_go:
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig
            
            fig = self.backtester.plot_results()
            
            # Should return empty figure
            assert fig == mock_fig
    
    def test_position_sizing_fixed(self):
        """Test fixed position sizing."""
        strategy_params = {
            'max_position_size': 0.5,
            'position_size_method': 'fixed'
        }
        
        results = self.backtester._run_simple_backtest(
            self.sample_data,
            self.sample_signals,
            strategy_params
        )
        
        # Check that positions are limited by max_position_size
        portfolio_df = results['portfolio_values']
        assert abs(portfolio_df['position']).max() <= 0.5
    
    def test_position_sizing_volatility_adjusted(self):
        """Test volatility-adjusted position sizing."""
        # Add volatility to data
        data_with_vol = self.sample_data.copy()
        data_with_vol['volatility_20'] = 0.02
        
        strategy_params = {
            'max_position_size': 1.0,
            'position_size_method': 'volatility_adjusted'
        }
        
        results = self.backtester._run_simple_backtest(
            data_with_vol,
            self.sample_signals,
            strategy_params
        )
        
        assert isinstance(results, dict)
        assert 'portfolio_values' in results
    
    def test_trade_execution(self):
        """Test trade execution logic."""
        # Create signals that should trigger trades
        trade_signals = pd.DataFrame({
            'signal': [0, 1, 0, -1, 0],
            'signal_strength': [0, 0.5, 0, 0.3, 0],
            'position': [0, 1, 0, -1, 0]
        }, index=self.sample_data.index)
        
        strategy_params = {
            'max_position_size': 1.0,
            'position_size_method': 'fixed'
        }
        
        results = self.backtester._run_simple_backtest(
            self.sample_data,
            trade_signals,
            strategy_params
        )
        
        trades_df = results['trades']
        
        # Should have some trades
        if not trades_df.empty:
            assert 'shares' in trades_df.columns
            assert 'price' in trades_df.columns
            assert 'commission' in trades_df.columns
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        # Create a signal that triggers a trade
        trade_signals = pd.DataFrame({
            'signal': [0, 1, 0, 0, 0],
            'signal_strength': [0, 0.5, 0, 0, 0],
            'position': [0, 1, 0, 0, 0]
        }, index=self.sample_data.index)
        
        strategy_params = {
            'max_position_size': 1.0,
            'position_size_method': 'fixed'
        }
        
        results = self.backtester._run_simple_backtest(
            self.sample_data,
            trade_signals,
            strategy_params
        )
        
        trades_df = results['trades']
        
        # Check commission is applied
        if not trades_df.empty and 'commission' in trades_df.columns:
            assert (trades_df['commission'] >= 0).all()
    
    def test_portfolio_value_tracking(self):
        """Test portfolio value tracking."""
        strategy_params = {
            'max_position_size': 1.0,
            'position_size_method': 'fixed'
        }
        
        results = self.backtester._run_simple_backtest(
            self.sample_data,
            self.sample_signals,
            strategy_params
        )
        
        portfolio_df = results['portfolio_values']
        
        # Check portfolio value components
        assert 'portfolio_value' in portfolio_df.columns
        assert 'cash' in portfolio_df.columns
        assert 'position' in portfolio_df.columns
        assert 'price' in portfolio_df.columns
        
        # Portfolio value should be positive
        assert (portfolio_df['portfolio_value'] > 0).all()
    
    def test_error_handling(self):
        """Test error handling in backtest."""
        # Test with invalid data types
        invalid_data = "not a dataframe"
        invalid_signals = "not a dataframe"
        
        with pytest.raises((AttributeError, TypeError)):
            self.backtester._run_simple_backtest(invalid_data, invalid_signals, {})
    
    def test_results_storage(self):
        """Test that results are stored properly."""
        strategy_params = {'max_position_size': 1.0}
        
        self.backtester.run_backtest(
            self.sample_data,
            self.sample_signals,
            strategy_params
        )
        
        assert self.backtester.results != {}
        assert 'portfolio_values' in self.backtester.results
        assert 'metrics' in self.backtester.results


if __name__ == "__main__":
    pytest.main([__file__])

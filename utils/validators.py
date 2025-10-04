"""
Data validation utilities for the trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, date
import re

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data for trading system components.
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_ohlcv_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLCV data format and content.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check if data is empty
            if data.empty:
                results['errors'].append("Data is empty")
                results['is_valid'] = False
                return results
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                results['errors'].append(f"Missing required columns: {missing_columns}")
                results['is_valid'] = False
            
            # Check data types
            numeric_columns = required_columns
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        results['errors'].append(f"Column '{col}' must be numeric")
                        results['is_valid'] = False
            
            # Check for negative values
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_count = (data[col] <= 0).sum()
                    if negative_count > 0:
                        results['errors'].append(f"Column '{col}' has {negative_count} non-positive values")
                        results['is_valid'] = False
            
            # Check volume
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    results['errors'].append(f"Volume has {negative_volume} negative values")
                    results['is_valid'] = False
            
            # Check OHLC relationships
            if all(col in data.columns for col in price_columns):
                # High should be >= Low
                invalid_hl = (data['high'] < data['low']).sum()
                if invalid_hl > 0:
                    results['errors'].append(f"High < Low in {invalid_hl} rows")
                    results['is_valid'] = False
                
                # High should be >= Open and Close
                invalid_ho = (data['high'] < data['open']).sum()
                invalid_hc = (data['high'] < data['close']).sum()
                if invalid_ho > 0:
                    results['errors'].append(f"High < Open in {invalid_ho} rows")
                    results['is_valid'] = False
                if invalid_hc > 0:
                    results['errors'].append(f"High < Close in {invalid_hc} rows")
                    results['is_valid'] = False
                
                # Low should be <= Open and Close
                invalid_lo = (data['low'] > data['open']).sum()
                invalid_lc = (data['low'] > data['close']).sum()
                if invalid_lo > 0:
                    results['errors'].append(f"Low > Open in {invalid_lo} rows")
                    results['is_valid'] = False
                if invalid_lc > 0:
                    results['errors'].append(f"Low > Close in {invalid_lc} rows")
                    results['is_valid'] = False
            
            # Check for missing values
            missing_values = data.isnull().sum()
            for col, count in missing_values.items():
                if count > 0:
                    results['warnings'].append(f"Column '{col}' has {count} missing values")
            
            # Check for duplicate timestamps
            if isinstance(data.index, pd.DatetimeIndex):
                duplicate_timestamps = data.index.duplicated().sum()
                if duplicate_timestamps > 0:
                    results['warnings'].append(f"Found {duplicate_timestamps} duplicate timestamps")
            
            # Calculate statistics
            results['stats'] = {
                'total_rows': len(data),
                'date_range': {
                    'start': str(data.index.min()) if len(data) > 0 else None,
                    'end': str(data.index.max()) if len(data) > 0 else None
                },
                'price_range': {
                    'min_close': float(data['close'].min()) if 'close' in data.columns else None,
                    'max_close': float(data['close'].max()) if 'close' in data.columns else None,
                    'avg_volume': float(data['volume'].mean()) if 'volume' in data.columns else None
                }
            }
            
            self.logger.info(f"OHLCV validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
            
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
            self.logger.error(f"Error validating OHLCV data: {e}")
        
        return results
    
    def validate_signals(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate trading signals.
        
        Args:
            signals: DataFrame with trading signals
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            if signals.empty:
                results['errors'].append("Signals data is empty")
                results['is_valid'] = False
                return results
            
            # Check for signal column
            if 'signal' not in signals.columns:
                results['errors'].append("Missing 'signal' column")
                results['is_valid'] = False
                return results
            
            signal_values = signals['signal']
            
            # Check signal values are numeric
            if not pd.api.types.is_numeric_dtype(signal_values):
                results['errors'].append("Signal values must be numeric")
                results['is_valid'] = False
            
            # Check for NaN values in signals
            nan_signals = signal_values.isnull().sum()
            if nan_signals > 0:
                results['warnings'].append(f"Found {nan_signals} NaN signal values")
            
            # Check for infinite values
            inf_signals = np.isinf(signal_values).sum()
            if inf_signals > 0:
                results['errors'].append(f"Found {inf_signals} infinite signal values")
                results['is_valid'] = False
            
            # Check signal range (optional warning)
            signal_min = signal_values.min()
            signal_max = signal_values.max()
            if abs(signal_min) > 10 or abs(signal_max) > 10:
                results['warnings'].append(f"Signal values outside typical range [-1, 1]: min={signal_min}, max={signal_max}")
            
            # Calculate signal statistics
            non_zero_signals = (signal_values != 0).sum()
            buy_signals = (signal_values > 0).sum()
            sell_signals = (signal_values < 0).sum()
            
            results['stats'] = {
                'total_signals': len(signals),
                'non_zero_signals': int(non_zero_signals),
                'buy_signals': int(buy_signals),
                'sell_signals': int(sell_signals),
                'signal_frequency': float(non_zero_signals / len(signals)) if len(signals) > 0 else 0,
                'signal_range': {
                    'min': float(signal_min),
                    'max': float(signal_max),
                    'mean': float(signal_values.mean()),
                    'std': float(signal_values.std())
                }
            }
            
            self.logger.info(f"Signal validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
            
        except Exception as e:
            results['errors'].append(f"Signal validation error: {str(e)}")
            results['is_valid'] = False
            self.logger.error(f"Error validating signals: {e}")
        
        return results
    
    def validate_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Validate trading symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            if not symbols:
                results['errors'].append("Symbol list is empty")
                results['is_valid'] = False
                return results
            
            # Check for duplicate symbols
            unique_symbols = list(set(symbols))
            duplicates = len(symbols) - len(unique_symbols)
            if duplicates > 0:
                results['warnings'].append(f"Found {duplicates} duplicate symbols")
            
            # Check symbol format
            invalid_symbols = []
            for symbol in unique_symbols:
                if not isinstance(symbol, str):
                    invalid_symbols.append(f"Non-string symbol: {symbol}")
                elif not re.match(r'^[A-Za-z0-9.-]+$', symbol):
                    invalid_symbols.append(f"Invalid symbol format: {symbol}")
                elif len(symbol) > 10:
                    results['warnings'].append(f"Long symbol name: {symbol}")
            
            if invalid_symbols:
                results['errors'].extend(invalid_symbols)
                results['is_valid'] = False
            
            # Check for empty symbols
            empty_symbols = [s for s in unique_symbols if not s.strip()]
            if empty_symbols:
                results['errors'].append(f"Found {len(empty_symbols)} empty symbols")
                results['is_valid'] = False
            
            results['stats'] = {
                'total_symbols': len(symbols),
                'unique_symbols': len(unique_symbols),
                'duplicate_count': duplicates
            }
            
            self.logger.info(f"Symbol validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
            
        except Exception as e:
            results['errors'].append(f"Symbol validation error: {str(e)}")
            results['is_valid'] = False
            self.logger.error(f"Error validating symbols: {e}")
        
        return results
    
    def validate_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Validate date range parameters.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Parse dates
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
            except Exception as e:
                results['errors'].append(f"Invalid date format: {e}")
                results['is_valid'] = False
                return results
            
            # Check date order
            if start_dt >= end_dt:
                results['errors'].append("Start date must be before end date")
                results['is_valid'] = False
            
            # Check date range
            date_diff = (end_dt - start_dt).days
            if date_diff > 365 * 10:  # More than 10 years
                results['warnings'].append(f"Large date range: {date_diff} days")
            elif date_diff < 1:
                results['errors'].append("Date range must be at least 1 day")
                results['is_valid'] = False
            
            # Check if dates are in the future
            today = pd.Timestamp.now()
            if start_dt > today:
                results['warnings'].append("Start date is in the future")
            if end_dt > today:
                results['warnings'].append("End date is in the future")
            
            results['stats'] = {
                'start_date': str(start_dt.date()),
                'end_date': str(end_dt.date()),
                'duration_days': date_diff
            }
            
            self.logger.info(f"Date range validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
            
        except Exception as e:
            results['errors'].append(f"Date validation error: {str(e)}")
            results['is_valid'] = False
            self.logger.error(f"Error validating date range: {e}")
        
        return results
    
    def validate_strategy_params(self, params: Dict[str, Any], strategy_type: str) -> Dict[str, Any]:
        """
        Validate strategy parameters.
        
        Args:
            params: Strategy parameters
            strategy_type: Type of strategy
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            if not isinstance(params, dict):
                results['errors'].append("Parameters must be a dictionary")
                results['is_valid'] = False
                return results
            
            # Define parameter schemas for different strategies
            schemas = {
                'momentum': {
                    'lookback_period': {'type': int, 'min': 1, 'max': 252},
                    'momentum_threshold': {'type': float, 'min': 0.0, 'max': 1.0},
                    'volume_threshold': {'type': float, 'min': 0.0, 'max': 10.0},
                    'rsi_oversold': {'type': int, 'min': 0, 'max': 50},
                    'rsi_overbought': {'type': int, 'min': 50, 'max': 100}
                },
                'mean_reversion': {
                    'lookback_window': {'type': int, 'min': 1, 'max': 252},
                    'std_dev_multiplier': {'type': float, 'min': 0.1, 'max': 5.0},
                    'volume_confirmation': {'type': bool},
                    'volume_threshold': {'type': float, 'min': 0.0, 'max': 10.0}
                }
            }
            
            if strategy_type not in schemas:
                results['warnings'].append(f"Unknown strategy type: {strategy_type}")
                return results
            
            schema = schemas[strategy_type]
            
            # Validate each parameter
            for param_name, param_value in params.items():
                if param_name not in schema:
                    results['warnings'].append(f"Unknown parameter: {param_name}")
                    continue
                
                param_schema = schema[param_name]
                
                # Type validation
                expected_type = param_schema['type']
                if not isinstance(param_value, expected_type):
                    results['errors'].append(f"Parameter '{param_name}' must be {expected_type.__name__}")
                    results['is_valid'] = False
                    continue
                
                # Range validation for numeric types
                if expected_type in [int, float]:
                    if 'min' in param_schema and param_value < param_schema['min']:
                        results['errors'].append(f"Parameter '{param_name}' must be >= {param_schema['min']}")
                        results['is_valid'] = False
                    if 'max' in param_schema and param_value > param_schema['max']:
                        results['errors'].append(f"Parameter '{param_name}' must be <= {param_schema['max']}")
                        results['is_valid'] = False
            
            # Check for missing required parameters
            required_params = ['lookback_period', 'lookback_window']  # Common required params
            missing_params = [p for p in required_params if p in schema and p not in params]
            if missing_params:
                results['warnings'].append(f"Missing parameters: {missing_params}")
            
            results['stats'] = {
                'strategy_type': strategy_type,
                'parameter_count': len(params),
                'validated_parameters': list(params.keys())
            }
            
            self.logger.info(f"Strategy parameter validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
            
        except Exception as e:
            results['errors'].append(f"Parameter validation error: {str(e)}")
            results['is_valid'] = False
            self.logger.error(f"Error validating strategy parameters: {e}")
        
        return results
    
    def validate_backtest_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate backtest results.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            required_keys = ['portfolio_values', 'metrics']
            missing_keys = [key for key in required_keys if key not in results]
            
            if missing_keys:
                validation_results['errors'].append(f"Missing required keys: {missing_keys}")
                validation_results['is_valid'] = False
            
            # Validate portfolio values
            if 'portfolio_values' in results:
                portfolio_values = results['portfolio_values']
                if isinstance(portfolio_values, pd.DataFrame):
                    if portfolio_values.empty:
                        validation_results['errors'].append("Portfolio values is empty")
                        validation_results['is_valid'] = False
                    else:
                        # Check for required columns
                        required_cols = ['portfolio_value']
                        missing_cols = [col for col in required_cols if col not in portfolio_values.columns]
                        if missing_cols:
                            validation_results['errors'].append(f"Portfolio values missing columns: {missing_cols}")
                            validation_results['is_valid'] = False
                else:
                    validation_results['errors'].append("Portfolio values must be a DataFrame")
                    validation_results['is_valid'] = False
            
            # Validate metrics
            if 'metrics' in results:
                metrics = results['metrics']
                if isinstance(metrics, dict):
                    # Check for negative portfolio values
                    if 'final_portfolio_value' in metrics:
                        if metrics['final_portfolio_value'] < 0:
                            validation_results['warnings'].append("Final portfolio value is negative")
                    
                    # Check for unrealistic returns
                    if 'total_return' in metrics:
                        return_val = metrics['total_return']
                        if abs(return_val) > 10:  # 1000% return
                            validation_results['warnings'].append(f"Unrealistic total return: {return_val:.2%}")
                else:
                    validation_results['errors'].append("Metrics must be a dictionary")
                    validation_results['is_valid'] = False
            
            validation_results['stats'] = {
                'has_portfolio_values': 'portfolio_values' in results,
                'has_metrics': 'metrics' in results,
                'has_trades': 'trades' in results
            }
            
            self.logger.info(f"Backtest results validation completed: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings")
            
        except Exception as e:
            validation_results['errors'].append(f"Backtest validation error: {str(e)}")
            validation_results['is_valid'] = False
            self.logger.error(f"Error validating backtest results: {e}")
        
        return validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.
        
        Returns:
            Dictionary with validation summary
        """
        return {
            'total_validations': len(self.validation_results),
            'failed_validations': len([r for r in self.validation_results.values() if not r.get('is_valid', False)]),
            'validation_results': self.validation_results
        }

"""
Abstract base strategy class for trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Strategy state
        self.positions = {}
        self.signals = pd.DataFrame()
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with signals
        """
        pass
    
    @abstractmethod
    def get_strategy_params(self) -> Dict:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input data
            
        Returns:
            True if data is valid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Data missing required columns: {required_columns}")
            return False
        
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        if data.isnull().any().any():
            self.logger.warning("Data contains null values")
        
        return True
    
    def calculate_position_size(self, 
                              signal_strength: float,
                              portfolio_value: float,
                              volatility: float,
                              max_position_pct: float = 0.1) -> float:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal_strength: Signal strength (-1 to 1)
            portfolio_value: Current portfolio value
            volatility: Asset volatility
            max_position_pct: Maximum position as percentage of portfolio
            
        Returns:
            Position size
        """
        # Base position size
        base_size = abs(signal_strength) * max_position_pct
        
        # Adjust for volatility (reduce position size for high volatility)
        vol_adjustment = min(0.02 / max(volatility, 0.01), 1.0)
        
        # Final position size
        position_size = base_size * vol_adjustment * np.sign(signal_strength)
        
        return position_size
    
    def apply_risk_management(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk management rules to signals.
        
        Args:
            signals: Raw signals
            
        Returns:
            Signals with risk management applied
        """
        df = signals.copy()
        
        # Stop loss and take profit levels
        stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        take_profit_pct = self.params.get('take_profit_pct', 0.10)
        
        # Maximum number of concurrent positions
        max_positions = self.params.get('max_positions', 5)
        
        # Position sizing limits
        max_position_size = self.params.get('max_position_size', 1.0)
        
        # Apply position size limits
        df['signal'] = np.clip(df['signal'], -max_position_size, max_position_size)
        
        # Add risk management flags
        df['stop_loss'] = df['close'] * (1 - stop_loss_pct)
        df['take_profit'] = df['close'] * (1 + take_profit_pct)
        
        return df
    
    def get_performance_summary(self) -> Dict:
        """
        Get strategy performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.signals.empty:
            return {}
        
        # Calculate basic performance metrics
        total_signals = len(self.signals)
        buy_signals = len(self.signals[self.signals['signal'] > 0])
        sell_signals = len(self.signals[self.signals['signal'] < 0])
        
        return {
            'strategy_name': self.name,
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_frequency': total_signals / len(self.signals) if len(self.signals) > 0 else 0,
            'parameters': self.params
        }
    
    def reset(self):
        """Reset strategy state."""
        self.positions = {}
        self.signals = pd.DataFrame()
        self.performance_metrics = {}
        self.logger.info(f"Strategy {self.name} reset")
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name}({self.params})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"

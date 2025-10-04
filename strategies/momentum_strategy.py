"""
Momentum-based trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy based on price and volume momentum.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize momentum strategy.
        
        Args:
            params: Strategy parameters
        """
        default_params = {
            'lookback_period': 20,
            'momentum_threshold': 0.02,
            'volume_threshold': 1.5,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_position_size': 1.0
        }
        
        # Merge with provided params
        if params:
            default_params.update(params)
        
        super().__init__(name="MomentumStrategy", params=default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum-based trading signals.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with signals and indicators
        """
        if not self.validate_data(data):
            return pd.DataFrame()
        
        try:
            df = data.copy()
            self.logger.info(f"Generating momentum signals for {len(df)} data points")
            
            # Calculate momentum indicators
            df = self._calculate_momentum_indicators(df)
            
            # Generate signals
            df = self._generate_momentum_signals(df)
            
            # Apply risk management
            df = self.apply_risk_management(df)
            
            # Store signals
            self.signals = df
            
            self.logger.info(f"Generated {len(df[df['signal'] != 0])} momentum signals")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {e}")
            return pd.DataFrame()
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        
        lookback = self.params['lookback_period']
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(periods=lookback)
        df['price_momentum_short'] = df['close'].pct_change(periods=5)
        
        # Moving averages
        df['ma_short'] = df['close'].rolling(window=10).mean()
        df['ma_long'] = df['close'].rolling(window=lookback).mean()
        df['ma_ratio'] = df['close'] / df['ma_long']
        
        # Volume momentum
        df['volume_ma'] = df['volume'].rolling(window=lookback).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_momentum'] = df['volume'].pct_change(periods=5)
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=lookback).std()
        
        # Rate of change
        df['roc'] = df['close'].pct_change(periods=10) * 100
        
        return df
    
    def _generate_momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum trading signals."""
        
        # Initialize signal column
        df['signal'] = 0.0
        
        # Momentum conditions
        momentum_threshold = self.params['momentum_threshold']
        volume_threshold = self.params['volume_threshold']
        rsi_oversold = self.params['rsi_oversold']
        rsi_overbought = self.params['rsi_overbought']
        
        # Buy conditions (positive momentum)
        buy_conditions = (
            (df['price_momentum'] > momentum_threshold) &
            (df['price_momentum_short'] > 0) &
            (df['volume_ratio'] > volume_threshold) &
            (df['rsi'] > rsi_oversold) &
            (df['rsi'] < rsi_overbought) &
            (df['macd'] > df['macd_signal']) &
            (df['ma_ratio'] > 1.01)  # Price above long MA
        )
        
        # Sell conditions (negative momentum)
        sell_conditions = (
            (df['price_momentum'] < -momentum_threshold) &
            (df['price_momentum_short'] < 0) &
            (df['volume_ratio'] > volume_threshold) &
            (df['rsi'] > rsi_oversold) &
            (df['rsi'] < rsi_overbought) &
            (df['macd'] < df['macd_signal']) &
            (df['ma_ratio'] < 0.99)  # Price below long MA
        )
        
        # Generate signals
        df.loc[buy_conditions, 'signal'] = 1.0
        df.loc[sell_conditions, 'signal'] = -1.0
        
        # Adjust signal strength based on momentum strength
        df.loc[buy_conditions, 'signal'] *= np.minimum(
            abs(df.loc[buy_conditions, 'price_momentum']) / momentum_threshold,
            2.0  # Cap at 2x
        )
        
        df.loc[sell_conditions, 'signal'] *= np.minimum(
            abs(df.loc[sell_conditions, 'price_momentum']) / momentum_threshold,
            2.0  # Cap at 2x
        )
        
        # Add signal metadata
        df['signal_strength'] = abs(df['signal'])
        df['signal_direction'] = np.sign(df['signal'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def get_strategy_params(self) -> Dict:
        """Get strategy parameters."""
        return self.params
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for momentum strategy.
        
        Returns:
            Dictionary with feature importance scores
        """
        return {
            'price_momentum': 0.35,
            'volume_ratio': 0.20,
            'rsi': 0.15,
            'macd': 0.15,
            'ma_ratio': 0.10,
            'volatility': 0.05
        }
    
    def get_strategy_description(self) -> str:
        """Get strategy description."""
        return """
        Momentum Strategy:
        - Buys when price shows strong positive momentum
        - Requires volume confirmation (volume > 1.5x average)
        - Uses RSI to avoid overbought/oversold conditions
        - MACD confirmation for trend direction
        - Position sizing based on momentum strength
        - Stop loss and take profit levels applied
        """

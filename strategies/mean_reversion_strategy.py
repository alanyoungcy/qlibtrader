"""
Mean reversion trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy based on statistical mean reversion.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize mean reversion strategy.
        
        Args:
            params: Strategy parameters
        """
        default_params = {
            'lookback_window': 50,
            'std_dev_multiplier': 2.0,
            'min_reversion_period': 5,
            'max_reversion_period': 20,
            'volume_confirmation': True,
            'volume_threshold': 1.2,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06,
            'max_position_size': 1.0
        }
        
        # Merge with provided params
        if params:
            default_params.update(params)
        
        super().__init__(name="MeanReversionStrategy", params=default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion trading signals.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with signals and indicators
        """
        if not self.validate_data(data):
            return pd.DataFrame()
        
        try:
            df = data.copy()
            self.logger.info(f"Generating mean reversion signals for {len(df)} data points")
            
            # Calculate mean reversion indicators
            df = self._calculate_mean_reversion_indicators(df)
            
            # Generate signals
            df = self._generate_mean_reversion_signals(df)
            
            # Apply risk management
            df = self.apply_risk_management(df)
            
            # Store signals
            self.signals = df
            
            self.logger.info(f"Generated {len(df[df['signal'] != 0])} mean reversion signals")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {e}")
            return pd.DataFrame()
    
    def _calculate_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        
        window = self.params['lookback_window']
        std_mult = self.params['std_dev_multiplier']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        bb_std = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_mult)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_mult)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Z-score (standardized price deviation)
        df['z_score'] = (df['close'] - df['bb_middle']) / bb_std
        
        # Price distance from mean
        df['price_distance'] = (df['close'] - df['bb_middle']) / df['bb_middle']
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # Mean reversion strength
        df['reversion_strength'] = self._calculate_reversion_strength(df)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=window).std()
        
        return df
    
    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion trading signals."""
        
        # Initialize signal column
        df['signal'] = 0.0
        
        # Get parameters
        std_mult = self.params['std_dev_multiplier']
        volume_threshold = self.params['volume_threshold']
        rsi_oversold = self.params['rsi_oversold']
        rsi_overbought = self.params['rsi_overbought']
        volume_confirmation = self.params['volume_confirmation']
        
        # Oversold conditions (buy signal)
        oversold_conditions = (
            (df['close'] <= df['bb_lower']) &  # Price below lower Bollinger Band
            (df['z_score'] <= -std_mult) &     # Z-score below threshold
            (df['rsi'] <= rsi_oversold) &      # RSI oversold
            (df['stoch_k'] <= 20) &           # Stochastic oversold
            (df['williams_r'] <= -80) &       # Williams %R oversold
            (df['reversion_strength'] > 0.5)  # Strong reversion potential
        )
        
        # Overbought conditions (sell signal)
        overbought_conditions = (
            (df['close'] >= df['bb_upper']) &  # Price above upper Bollinger Band
            (df['z_score'] >= std_mult) &      # Z-score above threshold
            (df['rsi'] >= rsi_overbought) &    # RSI overbought
            (df['stoch_k'] >= 80) &           # Stochastic overbought
            (df['williams_r'] >= -20) &       # Williams %R overbought
            (df['reversion_strength'] > 0.5)  # Strong reversion potential
        )
        
        # Add volume confirmation if required
        if volume_confirmation:
            oversold_conditions &= (df['volume_ratio'] >= volume_threshold)
            overbought_conditions &= (df['volume_ratio'] >= volume_threshold)
        
        # Generate signals
        df.loc[oversold_conditions, 'signal'] = 1.0
        df.loc[overbought_conditions, 'signal'] = -1.0
        
        # Adjust signal strength based on deviation from mean
        df.loc[oversold_conditions, 'signal'] *= np.minimum(
            abs(df.loc[oversold_conditions, 'z_score']) / std_mult,
            2.0  # Cap at 2x
        )
        
        df.loc[overbought_conditions, 'signal'] *= np.minimum(
            abs(df.loc[overbought_conditions, 'z_score']) / std_mult,
            2.0  # Cap at 2x
        )
        
        # Add signal metadata
        df['signal_strength'] = abs(df['signal'])
        df['signal_direction'] = np.sign(df['signal'])
        
        return df
    
    def _calculate_reversion_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate mean reversion strength indicator."""
        
        min_period = self.params['min_reversion_period']
        max_period = self.params['max_reversion_period']
        
        # Calculate autocorrelation for different lags
        reversion_scores = []
        
        for lag in range(min_period, max_period + 1):
            # Calculate autocorrelation
            returns = df['close'].pct_change()
            autocorr = returns.autocorr(lag=lag)
            
            # Negative autocorrelation indicates mean reversion
            reversion_scores.append(-autocorr if not np.isnan(autocorr) else 0)
        
        # Average reversion strength
        reversion_strength = pd.Series(reversion_scores).mean()
        
        # Create series with same length as df
        return pd.Series([reversion_strength] * len(df), index=df.index)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame, 
                            k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_window).min()
        high_max = df['high'].rolling(window=k_window).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=d_window).mean()
        return stoch_k, stoch_d
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R indicator."""
        high_max = df['high'].rolling(window=window).max()
        low_min = df['low'].rolling(window=window).min()
        williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
        return williams_r
    
    def get_strategy_params(self) -> Dict:
        """Get strategy parameters."""
        return self.params
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for mean reversion strategy.
        
        Returns:
            Dictionary with feature importance scores
        """
        return {
            'z_score': 0.30,
            'bb_position': 0.25,
            'rsi': 0.15,
            'stochastic': 0.10,
            'williams_r': 0.10,
            'reversion_strength': 0.05,
            'volume_ratio': 0.05
        }
    
    def get_strategy_description(self) -> str:
        """Get strategy description."""
        return """
        Mean Reversion Strategy:
        - Buys when price is significantly below mean (oversold)
        - Sells when price is significantly above mean (overbought)
        - Uses Bollinger Bands and Z-score for mean deviation
        - Confirms with RSI, Stochastic, and Williams %R
        - Volume confirmation to avoid false signals
        - Position sizing based on deviation strength
        - Tight stop loss and take profit for quick reversions
        """

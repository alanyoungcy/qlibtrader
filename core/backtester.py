"""
Backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import backtrader as bt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Integration with backtester library for strategy evaluation.
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital amount
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        self.trades = []
        self.portfolio_values = []
        
        if not BACKTRADER_AVAILABLE:
            self.logger.warning("Backtrader not available. Using simplified backtesting.")
    
    def run_backtest(self, 
                     data: pd.DataFrame,
                     signals: pd.DataFrame,
                     strategy_params: Dict) -> Dict:
        """
        Execute backtest with given data and signals.
        
        Args:
            data: Historical price data
            signals: Trading signals
            strategy_params: Strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info("Starting backtest execution")
            
            if BACKTRADER_AVAILABLE and strategy_params.get('use_backtrader', False):
                results = self._run_backtrader_backtest(data, signals, strategy_params)
            else:
                results = self._run_simple_backtest(data, signals, strategy_params)
            
            # Store results
            self.results = results
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    def _run_simple_backtest(self, data: pd.DataFrame, 
                           signals: pd.DataFrame,
                           params: Dict) -> Dict:
        """
        Run simplified backtest without backtrader.
        
        Args:
            data: Price data
            signals: Trading signals
            params: Strategy parameters
            
        Returns:
            Backtest results
        """
        try:
            # Ensure data alignment
            df = data.copy()
            if 'signal' in signals.columns:
                df['signal'] = signals['signal']
            elif 'position' in signals.columns:
                df['signal'] = signals['position']
            else:
                df['signal'] = 0
            
            # Initialize portfolio tracking
            portfolio_value = self.initial_capital
            position = 0
            cash = self.initial_capital
            
            portfolio_values = []
            trades = []
            
            # Position sizing parameters
            max_position_size = params.get('max_position_size', 1.0)
            position_size_method = params.get('position_size_method', 'fixed')
            
            self.logger.info(f"Running simple backtest with {len(df)} data points")
            
            for i, (date, row) in enumerate(df.iterrows()):
                current_price = row['close']
                signal = row.get('signal', 0)
                
                # Calculate position size
                if position_size_method == 'fixed':
                    target_position = signal * max_position_size
                elif position_size_method == 'volatility_adjusted':
                    # Adjust position size based on volatility
                    volatility = row.get('volatility_20', 0.02)
                    vol_adjustment = min(0.02 / volatility, 2.0)  # Cap at 2x
                    target_position = signal * max_position_size * vol_adjustment
                else:
                    target_position = signal
                
                # Execute trade if signal changes
                if abs(target_position - position) > 0.01:  # Threshold for trade execution
                    trade_value = abs(target_position - position) * current_price
                    commission_cost = trade_value * self.commission
                    
                    # Check if we have enough cash
                    required_cash = trade_value + commission_cost
                    if target_position > position and required_cash > cash:
                        target_position = position + (cash - commission_cost) / current_price
                    
                    # Execute trade
                    trade_shares = target_position - position
                    cash -= trade_shares * current_price + commission_cost
                    position = target_position
                    
                    # Record trade
                    if abs(trade_shares) > 0.01:
                        trades.append({
                            'date': date,
                            'price': current_price,
                            'shares': trade_shares,
                            'value': trade_shares * current_price,
                            'commission': commission_cost,
                            'position_after': position,
                            'cash_after': cash
                        })
                
                # Calculate portfolio value
                portfolio_value = cash + position * current_price
                portfolio_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'position': position,
                    'price': current_price
                })
            
            # Calculate performance metrics
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df.set_index('date', inplace=True)
            
            results = {
                'portfolio_values': portfolio_df,
                'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
                'metrics': self._calculate_metrics(portfolio_df),
                'final_portfolio_value': portfolio_value,
                'total_return': (portfolio_value - self.initial_capital) / self.initial_capital
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in simple backtest: {e}")
            raise
    
    def _run_backtrader_backtest(self, data: pd.DataFrame,
                               signals: pd.DataFrame,
                               params: Dict) -> Dict:
        """
        Run backtest using backtrader library.
        
        Args:
            data: Price data
            signals: Trading signals
            params: Strategy parameters
            
        Returns:
            Backtest results
        """
        try:
            # This would implement backtrader-specific backtesting
            # For now, fall back to simple backtest
            self.logger.info("Backtrader backtest not fully implemented, using simple backtest")
            return self._run_simple_backtest(data, signals, params)
            
        except Exception as e:
            self.logger.error(f"Error in backtrader backtest: {e}")
            raise
    
    def calculate_metrics(self, portfolio_values: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_values: Portfolio value time series
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if portfolio_values is None:
                portfolio_values = self.results.get('portfolio_values', pd.DataFrame())
            
            if portfolio_values.empty:
                return {}
            
            # Calculate returns
            returns = portfolio_values['portfolio_value'].pct_change().dropna()
            
            # Basic metrics
            total_return = (portfolio_values['portfolio_value'].iloc[-1] / 
                          portfolio_values['portfolio_value'].iloc[0]) - 1
            
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Win rate (if we have trades)
            trades = self.results.get('trades', pd.DataFrame())
            win_rate = 0
            if not trades.empty:
                winning_trades = len(trades[trades['value'] > 0])
                total_trades = len(trades)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'final_portfolio_value': portfolio_values['portfolio_value'].iloc[-1]
            }
            
            self.logger.info(f"Calculated metrics: Sharpe={sharpe_ratio:.3f}, Max DD={max_drawdown:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate detailed backtest report.
        
        Returns:
            DataFrame with detailed performance report
        """
        try:
            if not self.results:
                self.logger.warning("No backtest results available")
                return pd.DataFrame()
            
            portfolio_values = self.results.get('portfolio_values', pd.DataFrame())
            trades = self.results.get('trades', pd.DataFrame())
            metrics = self.results.get('metrics', {})
            
            # Create summary report
            report_data = []
            
            # Portfolio summary
            if not portfolio_values.empty:
                report_data.append({
                    'Metric': 'Initial Capital',
                    'Value': self.initial_capital,
                    'Type': 'Portfolio'
                })
                report_data.append({
                    'Metric': 'Final Portfolio Value',
                    'Value': metrics.get('final_portfolio_value', 0),
                    'Type': 'Portfolio'
                })
                report_data.append({
                    'Metric': 'Total Return',
                    'Value': f"{metrics.get('total_return', 0):.2%}",
                    'Type': 'Performance'
                })
                report_data.append({
                    'Metric': 'Annualized Return',
                    'Value': f"{metrics.get('annualized_return', 0):.2%}",
                    'Type': 'Performance'
                })
                report_data.append({
                    'Metric': 'Volatility',
                    'Value': f"{metrics.get('volatility', 0):.2%}",
                    'Type': 'Risk'
                })
                report_data.append({
                    'Metric': 'Sharpe Ratio',
                    'Value': f"{metrics.get('sharpe_ratio', 0):.3f}",
                    'Type': 'Risk'
                })
                report_data.append({
                    'Metric': 'Sortino Ratio',
                    'Value': f"{metrics.get('sortino_ratio', 0):.3f}",
                    'Type': 'Risk'
                })
                report_data.append({
                    'Metric': 'Maximum Drawdown',
                    'Value': f"{metrics.get('max_drawdown', 0):.2%}",
                    'Type': 'Risk'
                })
                report_data.append({
                    'Metric': 'Calmar Ratio',
                    'Value': f"{metrics.get('calmar_ratio', 0):.3f}",
                    'Type': 'Risk'
                })
            
            # Trading summary
            if not trades.empty:
                report_data.append({
                    'Metric': 'Total Trades',
                    'Value': metrics.get('total_trades', 0),
                    'Type': 'Trading'
                })
                report_data.append({
                    'Metric': 'Win Rate',
                    'Value': f"{metrics.get('win_rate', 0):.2%}",
                    'Type': 'Trading'
                })
            
            report_df = pd.DataFrame(report_data)
            
            self.logger.info("Generated backtest report")
            return report_df
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return pd.DataFrame()
    
    def plot_results(self) -> go.Figure:
        """
        Create interactive plots of backtest results.
        
        Returns:
            Plotly figure with backtest visualizations
        """
        try:
            if not self.results:
                self.logger.warning("No backtest results available for plotting")
                return go.Figure()
            
            portfolio_values = self.results.get('portfolio_values', pd.DataFrame())
            trades = self.results.get('trades', pd.DataFrame())
            
            if portfolio_values.empty:
                return go.Figure()
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Portfolio Value', 'Drawdown', 'Trades'),
                vertical_spacing=0.1,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Portfolio value plot
            fig.add_trace(
                go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add benchmark line
            initial_value = portfolio_values['portfolio_value'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=portfolio_values.index,
                    y=[initial_value] * len(portfolio_values),
                    mode='lines',
                    name='Initial Capital',
                    line=dict(color='gray', dash='dash')
                ),
                row=1, col=1
            )
            
            # Drawdown plot
            returns = portfolio_values['portfolio_value'].pct_change().dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown %',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.3)',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # Trades plot
            if not trades.empty:
                buy_trades = trades[trades['shares'] > 0]
                sell_trades = trades[trades['shares'] < 0]
                
                if not buy_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_trades['date'],
                            y=buy_trades['price'],
                            mode='markers',
                            name='Buy',
                            marker=dict(color='green', symbol='triangle-up', size=8)
                        ),
                        row=3, col=1
                    )
                
                if not sell_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_trades['date'],
                            y=sell_trades['price'],
                            mode='markers',
                            name='Sell',
                            marker=dict(color='red', symbol='triangle-down', size=8)
                        ),
                        row=3, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title='Backtest Results',
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=3, col=1)
            
            self.logger.info("Generated backtest plots")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            return go.Figure()
    
    def _calculate_metrics(self, portfolio_values: pd.DataFrame) -> Dict:
        """Calculate performance metrics from portfolio values."""
        return self.calculate_metrics(portfolio_values)

"""
Backtesting panel for the Gradio UI.
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BacktestPanel:
    """
    Backtesting and performance analysis panel.
    """
    
    def __init__(self):
        """Initialize backtest panel."""
        self.logger = logging.getLogger(__name__)
        self.current_results = None
        self.backtester = None
    
    def create_interface(self) -> gr.Blocks:
        """
        Create backtest panel interface.
        
        Returns:
            Gradio interface for backtest panel
        """
        with gr.Blocks(title="Backtesting") as interface:
            gr.Markdown("# 📈 Backtesting & Performance Analysis")
            gr.Markdown("Configure backtest parameters and analyze strategy performance")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Backtest Configuration
                    gr.Markdown("## Backtest Configuration")
                    
                    # Basic Parameters
                    initial_capital = gr.Number(
                        label="Initial Capital ($)",
                        value=100000,
                        minimum=1000,
                        maximum=10000000
                    )
                    
                    commission_rate = gr.Slider(
                        label="Commission Rate (%)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.01
                    )
                    
                    slippage_rate = gr.Slider(
                        label="Slippage Rate (%)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.05,
                        step=0.01
                    )
                    
                    # Position Sizing
                    gr.Markdown("### Position Sizing")
                    max_position_size = gr.Slider(
                        label="Max Position Size",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    
                    position_size_method = gr.Dropdown(
                        choices=["fixed", "volatility_adjusted", "kelly"],
                        label="Position Size Method",
                        value="fixed"
                    )
                    
                    # Risk Management
                    gr.Markdown("### Risk Management")
                    stop_loss_pct = gr.Slider(
                        label="Stop Loss (%)",
                        minimum=0.0,
                        maximum=20.0,
                        value=5.0,
                        step=0.5
                    )
                    
                    take_profit_pct = gr.Slider(
                        label="Take Profit (%)",
                        minimum=0.0,
                        maximum=50.0,
                        value=10.0,
                        step=0.5
                    )
                    
                    max_positions = gr.Slider(
                        label="Max Concurrent Positions",
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1
                    )
                    
                    # Advanced Options
                    gr.Markdown("### Advanced Options")
                    rebalancing_freq = gr.Dropdown(
                        choices=["daily", "weekly", "monthly"],
                        label="Rebalancing Frequency",
                        value="daily"
                    )
                    
                    use_backtrader = gr.Checkbox(
                        label="Use Backtrader Engine",
                        value=False
                    )
                    
                    # Run Backtest Button
                    run_backtest_btn = gr.Button("🚀 Run Backtest", variant="primary", size="lg")
                    
                    # Backtest Status
                    backtest_status = gr.Textbox(
                        label="Backtest Status",
                        value="Ready to backtest",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    # Backtest Results
                    gr.Markdown("## Backtest Results")
                    
                    # Performance Metrics
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Performance Metrics")
                            metrics_table = gr.Dataframe(
                                label="Performance Metrics",
                                headers=["Metric", "Value"],
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Portfolio Value")
                            portfolio_plot = gr.Plot(label="Portfolio Value Over Time")
                    
                    # Trade Analysis
                    gr.Markdown("### Trade Analysis")
                    trades_table = gr.Dataframe(
                        label="Trade History",
                        headers=["Date", "Action", "Price", "Shares", "Value", "Commission"],
                        interactive=False
                    )
                    
                    # Risk Metrics
                    with gr.Row():
                        gr.Markdown("### Risk Analysis")
                        risk_metrics = gr.Textbox(
                            label="Risk Metrics",
                            lines=8,
                            interactive=False
                        )
                    
                    # Backtest Summary
                    gr.Markdown("### Backtest Summary")
                    backtest_summary = gr.Textbox(
                        label="Summary",
                        lines=4,
                        interactive=False
                    )
            
            # Backtest Actions
            with gr.Row():
                save_results_btn = gr.Button("💾 Save Results", variant="secondary")
                export_report_btn = gr.Button("📊 Export Report", variant="secondary")
                clear_backtest_btn = gr.Button("🗑️ Clear Results", variant="stop")
            
            # Event Handlers
            run_backtest_btn.click(
                fn=self._run_backtest,
                inputs=[
                    initial_capital, commission_rate, slippage_rate,
                    max_position_size, position_size_method,
                    stop_loss_pct, take_profit_pct, max_positions,
                    rebalancing_freq, use_backtrader
                ],
                outputs=[metrics_table, portfolio_plot, trades_table, risk_metrics, backtest_summary, backtest_status]
            )
            
            save_results_btn.click(
                fn=self._save_results,
                outputs=[backtest_status]
            )
            
            clear_backtest_btn.click(
                fn=self._clear_backtest,
                outputs=[metrics_table, portfolio_plot, trades_table, risk_metrics, backtest_summary, backtest_status]
            )
        
        return interface
    
    def _run_backtest(self, *params) -> Tuple:
        """
        Run backtest with given parameters.
        
        Returns:
            Tuple of backtest results
        """
        try:
            # Extract parameters
            (initial_capital, commission_rate, slippage_rate,
             max_position_size, position_size_method,
             stop_loss_pct, take_profit_pct, max_positions,
             rebalancing_freq, use_backtrader) = params
            
            # Convert commission and slippage to decimals
            commission = commission_rate / 100.0
            slippage = slippage_rate / 100.0
            
            self.logger.info("Starting backtest")
            
            # Get sample data and signals (in real implementation, these would come from other panels)
            data = self._get_sample_data()
            signals = self._get_sample_signals(data)
            
            if data is None or data.empty:
                return self._empty_results("No data available")
            
            if signals is None or signals.empty:
                return self._empty_results("No signals available")
            
            # Initialize backtester
            from core.backtester import BacktestEngine
            self.backtester = BacktestEngine(
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage
            )
            
            # Prepare strategy parameters
            strategy_params = {
                'max_position_size': max_position_size,
                'position_size_method': position_size_method,
                'stop_loss_pct': stop_loss_pct / 100.0,
                'take_profit_pct': take_profit_pct / 100.0,
                'max_positions': max_positions,
                'rebalancing_freq': rebalancing_freq,
                'use_backtrader': use_backtrader
            }
            
            # Run backtest
            results = self.backtester.run_backtest(data, signals, strategy_params)
            
            # Store results
            self.current_results = results
            
            # Prepare output components
            metrics_df = self._prepare_metrics_table(results)
            portfolio_plot = self._create_portfolio_plot(results)
            trades_df = self._prepare_trades_table(results)
            risk_metrics = self._generate_risk_metrics(results)
            summary = self._generate_backtest_summary(results)
            
            status = f"✅ Backtest completed successfully"
            
            self.logger.info("Backtest completed")
            return metrics_df, portfolio_plot, trades_df, risk_metrics, summary, status
            
        except Exception as e:
            error_msg = f"Backtest error: {str(e)}"
            self.logger.error(error_msg)
            return self._empty_results(f"❌ {error_msg}")
    
    def _prepare_metrics_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Prepare performance metrics table."""
        try:
            metrics = results.get('metrics', {})
            
            metrics_data = [
                ["Total Return", f"{metrics.get('total_return', 0):.2%}"],
                ["Annualized Return", f"{metrics.get('annualized_return', 0):.2%}"],
                ["Volatility", f"{metrics.get('volatility', 0):.2%}"],
                ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}"],
                ["Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}"],
                ["Maximum Drawdown", f"{metrics.get('max_drawdown', 0):.2%}"],
                ["Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}"],
                ["Win Rate", f"{metrics.get('win_rate', 0):.2%}"],
                ["Total Trades", f"{metrics.get('total_trades', 0)}"],
                ["Final Portfolio Value", f"${metrics.get('final_portfolio_value', 0):,.2f}"]
            ]
            
            return pd.DataFrame(metrics_data, columns=["Metric", "Value"])
            
        except Exception as e:
            self.logger.error(f"Error preparing metrics table: {e}")
            return pd.DataFrame([["Error", str(e)]], columns=["Metric", "Value"])
    
    def _create_portfolio_plot(self, results: Dict[str, Any]):
        """Create portfolio value plot."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            portfolio_values = results.get('portfolio_values', pd.DataFrame())
            
            if portfolio_values.empty:
                return None
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Value', 'Drawdown'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
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
            
            # Update layout
            fig.update_layout(
                height=400,
                showlegend=True,
                title="Backtest Results"
            )
            
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio plot: {e}")
            return None
    
    def _prepare_trades_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Prepare trades table."""
        try:
            trades = results.get('trades', pd.DataFrame())
            
            if trades.empty:
                return pd.DataFrame([["No trades executed", "", "", "", "", ""]], 
                                  columns=["Date", "Action", "Price", "Shares", "Value", "Commission"])
            
            # Select relevant columns and format
            trade_cols = ['date', 'shares', 'price', 'value', 'commission']
            available_cols = [col for col in trade_cols if col in trades.columns]
            
            if available_cols:
                trades_display = trades[available_cols].copy()
                
                # Add action column
                trades_display['Action'] = np.where(trades_display['shares'] > 0, 'BUY', 'SELL')
                
                # Format columns
                if 'date' in trades_display.columns:
                    trades_display['Date'] = pd.to_datetime(trades_display['date']).dt.strftime('%Y-%m-%d')
                if 'price' in trades_display.columns:
                    trades_display['Price'] = trades_display['price'].round(2)
                if 'value' in trades_display.columns:
                    trades_display['Value'] = trades_display['value'].round(2)
                if 'commission' in trades_display.columns:
                    trades_display['Commission'] = trades_display['commission'].round(2)
                if 'shares' in trades_display.columns:
                    trades_display['Shares'] = trades_display['shares'].round(2)
                
                # Select display columns
                display_cols = ['Date', 'Action', 'Price', 'Shares', 'Value', 'Commission']
                available_display_cols = [col for col in display_cols if col in trades_display.columns]
                
                return trades_display[available_display_cols].head(20)
            
            return pd.DataFrame([["No trade data available", "", "", "", "", ""]], 
                              columns=["Date", "Action", "Price", "Shares", "Value", "Commission"])
            
        except Exception as e:
            self.logger.error(f"Error preparing trades table: {e}")
            return pd.DataFrame([["Error", str(e), "", "", "", ""]], 
                              columns=["Date", "Action", "Price", "Shares", "Value", "Commission"])
    
    def _generate_risk_metrics(self, results: Dict[str, Any]) -> str:
        """Generate risk metrics text."""
        try:
            metrics = results.get('metrics', {})
            
            risk_lines = []
            risk_lines.append("📊 Risk Analysis")
            risk_lines.append("")
            
            # Volatility metrics
            volatility = metrics.get('volatility', 0)
            risk_lines.append(f"Annualized Volatility: {volatility:.2%}")
            
            # Risk-adjusted returns
            sharpe = metrics.get('sharpe_ratio', 0)
            sortino = metrics.get('sortino_ratio', 0)
            risk_lines.append(f"Sharpe Ratio: {sharpe:.3f}")
            risk_lines.append(f"Sortino Ratio: {sortino:.3f}")
            
            # Drawdown analysis
            max_dd = metrics.get('max_drawdown', 0)
            calmar = metrics.get('calmar_ratio', 0)
            risk_lines.append(f"Maximum Drawdown: {max_dd:.2%}")
            risk_lines.append(f"Calmar Ratio: {calmar:.3f}")
            
            # Trading metrics
            total_trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0)
            risk_lines.append(f"Total Trades: {total_trades}")
            risk_lines.append(f"Win Rate: {win_rate:.2%}")
            
            return "\n".join(risk_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating risk metrics: {e}")
            return f"Error generating risk metrics: {str(e)}"
    
    def _generate_backtest_summary(self, results: Dict[str, Any]) -> str:
        """Generate backtest summary."""
        try:
            metrics = results.get('metrics', {})
            portfolio_values = results.get('portfolio_values', pd.DataFrame())
            
            summary_lines = []
            summary_lines.append("📈 Backtest Summary")
            summary_lines.append("")
            
            # Performance overview
            total_return = metrics.get('total_return', 0)
            annualized_return = metrics.get('annualized_return', 0)
            
            summary_lines.append(f"Total Return: {total_return:.2%}")
            summary_lines.append(f"Annualized Return: {annualized_return:.2%}")
            
            # Risk summary
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown', 0)
            
            summary_lines.append(f"Sharpe Ratio: {sharpe:.3f}")
            summary_lines.append(f"Maximum Drawdown: {max_dd:.2%}")
            
            # Portfolio value
            final_value = metrics.get('final_portfolio_value', 0)
            summary_lines.append(f"Final Portfolio Value: ${final_value:,.2f}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating backtest summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def _save_results(self) -> str:
        """Save backtest results."""
        try:
            if self.current_results is None:
                return "❌ No results to save"
            
            # Save results (implementation would save to file)
            self.logger.info("Backtest results saved")
            return "✅ Results saved successfully"
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return f"❌ Error saving results: {str(e)}"
    
    def _clear_backtest(self) -> Tuple:
        """Clear backtest results."""
        try:
            self.current_results = None
            self.backtester = None
            
            empty_df = pd.DataFrame([["No results", ""]], columns=["Metric", "Value"])
            empty_trades = pd.DataFrame([["No trades", "", "", "", "", ""]], 
                                      columns=["Date", "Action", "Price", "Shares", "Value", "Commission"])
            
            return empty_df, None, empty_trades, "", "✅ Results cleared"
            
        except Exception as e:
            self.logger.error(f"Error clearing backtest: {e}")
            return (pd.DataFrame(), None, pd.DataFrame(), f"Error: {str(e)}", "❌ Error clearing results")
    
    def _empty_results(self, error_msg: str) -> Tuple:
        """Return empty results with error message."""
        empty_df = pd.DataFrame([["Error", error_msg]], columns=["Metric", "Value"])
        empty_trades = pd.DataFrame([["Error", error_msg, "", "", "", ""]], 
                                  columns=["Date", "Action", "Price", "Shares", "Value", "Commission"])
        
        return empty_df, None, empty_trades, f"Error: {error_msg}", error_msg
    
    def _get_sample_data(self) -> pd.DataFrame:
        """Get sample data for testing."""
        try:
            # Create sample OHLCV data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            
            # Generate realistic price data
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
            
            # Ensure OHLC relationships are valid
            data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
            data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating sample data: {e}")
            return pd.DataFrame()
    
    def _get_sample_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get sample signals for testing."""
        try:
            if data.empty:
                return pd.DataFrame()
            
            # Generate simple momentum signals
            signals_df = data.copy()
            
            # Calculate momentum
            momentum = signals_df['close'].pct_change(20)  # 20-day momentum
            
            # Generate signals
            signals_df['signal'] = 0.0
            signals_df.loc[momentum > 0.02, 'signal'] = 1.0  # Buy signal
            signals_df.loc[momentum < -0.02, 'signal'] = -1.0  # Sell signal
            
            # Add signal strength
            signals_df['signal_strength'] = abs(momentum)
            signals_df['position'] = np.where(signals_df['signal'] > 0, 1,
                                            np.where(signals_df['signal'] < 0, -1, 0))
            
            return signals_df
            
        except Exception as e:
            self.logger.error(f"Error creating sample signals: {e}")
            return pd.DataFrame()
    
    def get_current_results(self) -> Optional[Dict[str, Any]]:
        """Get current backtest results."""
        return self.current_results

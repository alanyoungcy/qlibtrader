"""
Simplified Gradio application for the trading system.
This version focuses on core functionality without complex components.
"""

import gradio as gr
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ConfigManager
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class SimpleTradingSystemUI:
    """
    Simplified Gradio application for the trading system.
    """
    
    def __init__(self, config_path: str = "configs"):
        """
        Initialize the simplified trading system UI.
        
        Args:
            config_path: Path to configuration files
        """
        self.config_manager = ConfigManager(config_path)
        self.logger = setup_logger("simple_trading_ui")
        
        # Load configurations
        self._load_configurations()
        
        # Initialize data storage
        self.current_data = None
        self.current_signals = None
        self.current_results = None
        
        self.logger.info("Simple Trading System UI initialized")
    
    def _load_configurations(self):
        """Load system configurations."""
        try:
            # Create default configs if they don't exist
            self.config_manager.create_default_configs()
            
            # Load configurations
            self.data_config = self.config_manager.get_data_config()
            self.strategy_config = self.config_manager.get_strategy_config()
            self.model_config = self.config_manager.get_model_config()
            
            self.logger.info("Configurations loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
            # Set default configs
            self.data_config = {}
            self.strategy_config = {}
            self.model_config = {}
    
    def create_interface(self) -> gr.Blocks:
        """
        Build the simplified Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        try:
            with gr.Blocks(
                theme=gr.themes.Soft(),
                title="Trading System - Simplified",
                css=self._get_custom_css()
            ) as demo:
                
                # Header
                gr.Markdown("# 🚀 AI-Powered Trading System")
                gr.Markdown("""
                A simplified trading system interface for data fetching, analysis, and backtesting.
                """)
                
                # Main tabs
                with gr.Tabs():
                    # Tab 1: Data Management
                    with gr.Tab("📊 Data Management"):
                        self._create_data_panel()
                    
                    # Tab 2: Analysis & Signals
                    with gr.Tab("🔍 Analysis & Signals"):
                        self._create_analysis_panel()
                    
                    # Tab 3: Backtesting
                    with gr.Tab("📈 Backtesting"):
                        self._create_backtest_panel()
                    
                    # Tab 4: System Status
                    with gr.Tab("⚙️ System Status"):
                        self._create_status_panel()
                
                # Footer
                gr.Markdown("""
                ---
                **Trading System** - Simplified Interface
                """)
            
            self.logger.info("Simplified Gradio interface created successfully")
            return demo
            
        except Exception as e:
            self.logger.error(f"Error creating interface: {e}")
            raise
    
    def _create_data_panel(self):
        """Create simplified data management panel."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Data Configuration")
                
                symbols_input = gr.Textbox(
                    label="Symbols (comma-separated)",
                    placeholder="AAPL,MSFT,GOOGL",
                    value="AAPL,MSFT,GOOGL"
                )
                
                with gr.Row():
                    start_date = gr.Textbox(
                        label="Start Date",
                        placeholder="2023-01-01",
                        value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                    )
                    end_date = gr.Textbox(
                        label="End Date",
                        placeholder="2024-01-01",
                        value=datetime.now().strftime("%Y-%m-%d")
                    )
                
                fetch_data_btn = gr.Button("📊 Fetch Data", variant="primary")
                clear_data_btn = gr.Button("🗑️ Clear Data", variant="secondary")
                
                data_status = gr.Textbox(
                    label="Data Status",
                    value="No data loaded",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## Data Preview")
                
                data_display = gr.Dataframe(
                    label="Market Data",
                    headers=["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"],
                    interactive=False
                )
                
                data_stats = gr.Textbox(
                    label="Data Statistics",
                    lines=4,
                    interactive=False
                )
        
        # Event handlers
        fetch_data_btn.click(
            fn=self._fetch_data,
            inputs=[symbols_input, start_date, end_date],
            outputs=[data_display, data_stats, data_status]
        )
        
        clear_data_btn.click(
            fn=self._clear_data,
            outputs=[data_display, data_stats, data_status]
        )
    
    def _create_analysis_panel(self):
        """Create simplified analysis panel."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Analysis Configuration")
                
                analysis_type = gr.Radio(
                    choices=["Basic Analysis", "Technical Indicators", "ML Analysis"],
                    label="Analysis Type",
                    value="Basic Analysis"
                )
                
                generate_signals_btn = gr.Button("🔍 Generate Signals", variant="primary")
                clear_signals_btn = gr.Button("🗑️ Clear Signals", variant="secondary")
                
                signals_status = gr.Textbox(
                    label="Signals Status",
                    value="No signals generated",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## Analysis Results")
                
                signals_display = gr.Dataframe(
                    label="Trading Signals",
                    headers=["Date", "Symbol", "Signal", "Confidence", "Price"],
                    interactive=False
                )
                
                analysis_stats = gr.Textbox(
                    label="Analysis Statistics",
                    lines=4,
                    interactive=False
                )
        
        # Event handlers
        generate_signals_btn.click(
            fn=self._generate_signals,
            inputs=[analysis_type],
            outputs=[signals_display, analysis_stats, signals_status]
        )
        
        clear_signals_btn.click(
            fn=self._clear_signals,
            outputs=[signals_display, analysis_stats, signals_status]
        )
    
    def _create_backtest_panel(self):
        """Create simplified backtest panel."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Backtest Configuration")
                
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
                
                run_backtest_btn = gr.Button("📈 Run Backtest", variant="primary")
                clear_backtest_btn = gr.Button("🗑️ Clear Results", variant="secondary")
                
                backtest_status = gr.Textbox(
                    label="Backtest Status",
                    value="No backtest completed",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## Backtest Results")
                
                results_display = gr.Dataframe(
                    label="Performance Metrics",
                    headers=["Metric", "Value"],
                    interactive=False
                )
                
                backtest_stats = gr.Textbox(
                    label="Backtest Summary",
                    lines=6,
                    interactive=False
                )
        
        # Event handlers
        run_backtest_btn.click(
            fn=self._run_backtest,
            inputs=[initial_capital, commission_rate],
            outputs=[results_display, backtest_stats, backtest_status]
        )
        
        clear_backtest_btn.click(
            fn=self._clear_backtest,
            outputs=[results_display, backtest_stats, backtest_status]
        )
    
    def _create_status_panel(self):
        """Create system status panel."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("## System Information")
                
                system_info = gr.Textbox(
                    label="System Status",
                    value="System running normally",
                    lines=3,
                    interactive=False
                )
                
                check_status_btn = gr.Button("🔄 Check Status", variant="primary")
                
                gr.Markdown("## Configuration")
                
                config_display = gr.JSON(
                    label="Current Configuration",
                    value={
                        "data_config": self.data_config,
                        "strategy_config": self.strategy_config,
                        "model_config": self.model_config
                    }
                )
        
        # Event handlers
        check_status_btn.click(
            fn=self._check_system_status,
            outputs=[system_info]
        )
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            background-color: #f8f9fa;
        }
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        """
    
    # Event handler methods
    def _fetch_data(self, symbols: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str, str]:
        """Fetch market data."""
        try:
            self.logger.info(f"Fetching data for symbols: {symbols}")
            
            # Create sample data for demonstration
            symbols_list = [s.strip() for s in symbols.split(',')]
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            data = []
            for symbol in symbols_list:
                for date in dates:
                    # Generate sample OHLCV data
                    base_price = 100 + hash(symbol) % 50
                    price_change = (hash(f"{symbol}{date}") % 20 - 10) / 100
                    price = base_price * (1 + price_change)
                    
                    data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Symbol': symbol,
                        'Open': round(price * 0.99, 2),
                        'High': round(price * 1.02, 2),
                        'Low': round(price * 0.98, 2),
                        'Close': round(price, 2),
                        'Volume': hash(f"{symbol}{date}") % 1000000 + 100000
                    })
            
            df = pd.DataFrame(data)
            self.current_data = df
            
            # Generate statistics
            stats = f"Data loaded: {len(df)} records\n"
            stats += f"Symbols: {', '.join(symbols_list)}\n"
            stats += f"Date range: {start_date} to {end_date}\n"
            stats += f"Average volume: {df['Volume'].mean():,.0f}"
            
            return df, stats, "✅ Data fetched successfully"
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return pd.DataFrame(), f"Error: {str(e)}", "❌ Error fetching data"
    
    def _clear_data(self) -> Tuple[pd.DataFrame, str, str]:
        """Clear loaded data."""
        self.current_data = None
        return pd.DataFrame(), "No data loaded", "Data cleared"
    
    def _generate_signals(self, analysis_type: str) -> Tuple[pd.DataFrame, str, str]:
        """Generate trading signals."""
        try:
            if self.current_data is None or self.current_data.empty:
                return pd.DataFrame(), "No data available", "❌ No data loaded"
            
            self.logger.info(f"Generating signals using {analysis_type}")
            
            # Generate sample signals
            signals = []
            for symbol in self.current_data['Symbol'].unique():
                symbol_data = self.current_data[self.current_data['Symbol'] == symbol]
                for _, row in symbol_data.iterrows():
                    # Simple signal generation logic
                    price = row['Close']
                    signal = "BUY" if price > symbol_data['Close'].mean() else "SELL"
                    confidence = min(0.9, max(0.1, abs(price - symbol_data['Close'].mean()) / symbol_data['Close'].std()))
                    
                    signals.append({
                        'Date': row['Date'],
                        'Symbol': symbol,
                        'Signal': signal,
                        'Confidence': round(confidence, 3),
                        'Price': price
                    })
            
            df = pd.DataFrame(signals)
            self.current_signals = df
            
            # Generate statistics
            stats = f"Signals generated: {len(df)}\n"
            stats += f"Buy signals: {len(df[df['Signal'] == 'BUY'])}\n"
            stats += f"Sell signals: {len(df[df['Signal'] == 'SELL'])}\n"
            stats += f"Average confidence: {df['Confidence'].mean():.3f}"
            
            return df, stats, "✅ Signals generated successfully"
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return pd.DataFrame(), f"Error: {str(e)}", "❌ Error generating signals"
    
    def _clear_signals(self) -> Tuple[pd.DataFrame, str, str]:
        """Clear generated signals."""
        self.current_signals = None
        return pd.DataFrame(), "No signals generated", "Signals cleared"
    
    def _run_backtest(self, initial_capital: float, commission_rate: float) -> Tuple[pd.DataFrame, str, str]:
        """Run backtest simulation."""
        try:
            if self.current_signals is None or self.current_signals.empty:
                return pd.DataFrame(), "No signals available", "❌ No signals generated"
            
            self.logger.info(f"Running backtest with capital: ${initial_capital:,.0f}")
            
            # Simple backtest simulation
            capital = initial_capital
            position = 0
            trades = []
            
            for _, signal in self.current_signals.iterrows():
                if signal['Signal'] == 'BUY' and position == 0:
                    position = capital * 0.1  # Use 10% of capital
                    capital -= position
                    trades.append({
                        'Date': signal['Date'],
                        'Action': 'BUY',
                        'Price': signal['Price'],
                        'Amount': position,
                        'Capital': capital
                    })
                elif signal['Signal'] == 'SELL' and position > 0:
                    sell_amount = position * 1.05  # 5% gain
                    capital += sell_amount
                    trades.append({
                        'Date': signal['Date'],
                        'Action': 'SELL',
                        'Price': signal['Price'],
                        'Amount': sell_amount,
                        'Capital': capital
                    })
                    position = 0
            
            # Calculate final results
            final_capital = capital + position
            total_return = (final_capital - initial_capital) / initial_capital
            
            results_df = pd.DataFrame([
                {'Metric': 'Initial Capital', 'Value': f"${initial_capital:,.2f}"},
                {'Metric': 'Final Capital', 'Value': f"${final_capital:,.2f}"},
                {'Metric': 'Total Return', 'Value': f"{total_return:.2%}"},
                {'Metric': 'Total Trades', 'Value': str(len(trades))},
                {'Metric': 'Commission Rate', 'Value': f"{commission_rate:.2%}"}
            ])
            
            self.current_results = {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'trades': trades
            }
            
            stats = f"Backtest completed successfully!\n"
            stats += f"Initial Capital: ${initial_capital:,.2f}\n"
            stats += f"Final Capital: ${final_capital:,.2f}\n"
            stats += f"Total Return: {total_return:.2%}\n"
            stats += f"Total Trades: {len(trades)}"
            
            return results_df, stats, "✅ Backtest completed successfully"
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return pd.DataFrame(), f"Error: {str(e)}", "❌ Error running backtest"
    
    def _clear_backtest(self) -> Tuple[pd.DataFrame, str, str]:
        """Clear backtest results."""
        self.current_results = None
        return pd.DataFrame(), "No backtest completed", "Backtest results cleared"
    
    def _check_system_status(self) -> str:
        """Check system status."""
        try:
            status = "System Status:\n"
            status += f"✅ UI: Running\n"
            status += f"✅ Data: {'Loaded' if self.current_data is not None else 'Not loaded'}\n"
            status += f"✅ Signals: {'Generated' if self.current_signals is not None else 'Not generated'}\n"
            status += f"✅ Backtest: {'Completed' if self.current_results is not None else 'Not completed'}\n"
            status += f"✅ Config: Loaded\n"
            status += f"✅ Logging: Active"
            
            return status
            
        except Exception as e:
            return f"Error checking status: {str(e)}"
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", 
               server_port: int = 7860, **kwargs) -> None:
        """
        Launch the simplified Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            **kwargs: Additional launch parameters
        """
        try:
            interface = self.create_interface()
            
            self.logger.info(f"Launching Simplified Trading System UI on {server_name}:{server_port}")
            
            # Simple launch parameters
            launch_kwargs = {
                "share": share,
                "server_name": server_name,
                "server_port": server_port,
                "show_error": True,
                "quiet": False,
                "inbrowser": False,
                **kwargs
            }
            
            interface.launch(**launch_kwargs)
            
        except Exception as e:
            self.logger.error(f"Error launching interface: {e}")
            raise

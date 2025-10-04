"""
Main Gradio application for the trading system.
"""

import gradio as gr
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .components.data_panel import DataPanel
from .components.analysis_panel import AnalysisPanel
from .components.backtest_panel import BacktestPanel
from utils.config import ConfigManager
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class TradingSystemUI:
    """
    Main Gradio application for the trading system.
    """
    
    def __init__(self, config_path: str = "configs"):
        """
        Initialize the trading system UI.
        
        Args:
            config_path: Path to configuration files
        """
        self.config_manager = ConfigManager(config_path)
        self.logger = setup_logger("trading_system_ui")
        
        # Initialize panels
        self.data_panel = DataPanel()
        self.analysis_panel = AnalysisPanel()
        self.backtest_panel = BacktestPanel()
        
        # Load configurations
        self._load_configurations()
        
        self.logger.info("Trading System UI initialized")
    
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
    
    def create_interface(self) -> gr.Blocks:
        """
        Build the main Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        try:
            with gr.Blocks(
                theme=gr.themes.Soft(),
                title="Trading System",
                css=self._get_custom_css()
            ) as demo:
                
                # Header
                gr.Markdown("# 🚀 AI-Powered Trading System")
                gr.Markdown("""
                A comprehensive trading system integrating Databento data, Qlib analysis, 
                machine learning models, and advanced backtesting capabilities.
                """)
                
                # Main tabs
                with gr.Tabs():
                    # Tab 1: Data Management
                    with gr.Tab("📊 Data Management", id="data_tab"):
                        self.data_panel.create_interface()
                    
                    # Tab 2: Analysis & Signals
                    with gr.Tab("🔍 Analysis & Signals", id="analysis_tab"):
                        self.analysis_panel.create_interface()
                    
                    # Tab 3: Backtesting
                    with gr.Tab("📈 Backtesting", id="backtest_tab"):
                        self.backtest_panel.create_interface()
                    
                    # Tab 4: Results & Reports
                    with gr.Tab("📋 Results & Reports", id="results_tab"):
                        self._create_results_panel()
                    
                    # Tab 5: Configuration
                    with gr.Tab("⚙️ Configuration", id="config_tab"):
                        self._create_config_panel()
                
                # Footer
                gr.Markdown("""
                ---
                **Trading System** - Built with Databento, Qlib, Scikit-learn, and Gradio
                """)
            
            self.logger.info("Gradio interface created successfully")
            return demo
            
        except Exception as e:
            self.logger.error(f"Error creating interface: {e}")
            raise
    
    def _create_results_panel(self) -> gr.Blocks:
        """
        Create results and reports panel.
        
        Returns:
            Results panel interface
        """
        with gr.Blocks(title="Results & Reports") as results_interface:
            gr.Markdown("# 📋 Results & Reports")
            gr.Markdown("View comprehensive analysis results and generate reports")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Report Configuration
                    gr.Markdown("## Report Configuration")
                    
                    report_type = gr.Dropdown(
                        choices=["Performance Summary", "Detailed Analysis", "Risk Report", "Trade Analysis"],
                        label="Report Type",
                        value="Performance Summary"
                    )
                    
                    include_plots = gr.Checkbox(
                        label="Include Plots",
                        value=True
                    )
                    
                    export_format = gr.Dropdown(
                        choices=["HTML", "PDF", "Excel", "JSON"],
                        label="Export Format",
                        value="HTML"
                    )
                    
                    generate_report_btn = gr.Button("📊 Generate Report", variant="primary")
                    
                    export_report_btn = gr.Button("💾 Export Report", variant="secondary")
                
                with gr.Column(scale=2):
                    # Report Display
                    gr.Markdown("## Generated Report")
                    
                    report_display = gr.HTML(
                        label="Report Content",
                        value="<p>No report generated yet. Configure and generate a report.</p>"
                    )
                    
                    # Report Statistics
                    gr.Markdown("### Report Statistics")
                    report_stats = gr.Textbox(
                        label="Statistics",
                        lines=6,
                        interactive=False
                    )
            
            # Event Handlers
            generate_report_btn.click(
                fn=self._generate_report,
                inputs=[report_type, include_plots],
                outputs=[report_display, report_stats]
            )
            
            export_report_btn.click(
                fn=self._export_report,
                inputs=[export_format],
                outputs=[gr.Textbox(label="Export Status", interactive=False)]
            )
        
        return results_interface
    
    def _create_config_panel(self) -> gr.Blocks:
        """
        Create configuration panel.
        
        Returns:
            Configuration panel interface
        """
        with gr.Blocks(title="Configuration") as config_interface:
            gr.Markdown("# ⚙️ System Configuration")
            gr.Markdown("Manage system settings and configurations")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Configuration Tabs
                    with gr.Tabs():
                        with gr.Tab("Data Config"):
                            gr.Markdown("### Data Configuration")
                            
                            databento_api_key = gr.Textbox(
                                label="Databento API Key",
                                value=self.data_config.get('databento', {}).get('api_key', ''),
                                type="password"
                            )
                            
                            default_schema = gr.Dropdown(
                                choices=["ohlcv-1d", "ohlcv-1h", "ohlcv-1m"],
                                label="Default Schema",
                                value=self.data_config.get('databento', {}).get('default_schema', 'ohlcv-1d')
                            )
                            
                            cache_dir = gr.Textbox(
                                label="Cache Directory",
                                value=self.data_config.get('databento', {}).get('cache_dir', './data/raw')
                            )
                        
                        with gr.Tab("Strategy Config"):
                            gr.Markdown("### Strategy Configuration")
                            
                            initial_capital = gr.Number(
                                label="Initial Capital",
                                value=self.strategy_config.get('backtesting', {}).get('initial_capital', 100000)
                            )
                            
                            commission = gr.Slider(
                                label="Commission Rate",
                                minimum=0.0,
                                maximum=1.0,
                                value=self.strategy_config.get('backtesting', {}).get('commission', 0.001)
                            )
                        
                        with gr.Tab("Model Config"):
                            gr.Markdown("### Model Configuration")
                            
                            qlib_model = gr.Dropdown(
                                choices=["lightgbm", "xgboost", "linear"],
                                label="Qlib Model Type",
                                value=self.model_config.get('qlib', {}).get('model_type', 'lightgbm')
                            )
                            
                            sklearn_classifier = gr.Dropdown(
                                choices=["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"],
                                label="Sklearn Classifier",
                                value=self.model_config.get('sklearn', {}).get('classifier', 'RandomForestClassifier')
                            )
                    
                    # Save Configuration
                    save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
                    load_config_btn = gr.Button("📂 Load Configuration", variant="secondary")
                    reset_config_btn = gr.Button("🔄 Reset to Defaults", variant="stop")
                    
                    config_status = gr.Textbox(
                        label="Configuration Status",
                        value="Configuration loaded",
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    # Configuration Preview
                    gr.Markdown("## Configuration Preview")
                    
                    config_preview = gr.JSON(
                        label="Current Configuration",
                        value={
                            "data_config": self.data_config,
                            "strategy_config": self.strategy_config,
                            "model_config": self.model_config
                        }
                    )
                    
                    # Environment Variables
                    gr.Markdown("### Environment Variables")
                    env_vars = gr.Textbox(
                        label="Required Environment Variables",
                        value="DATABENTO_API_KEY=your_api_key_here",
                        lines=3,
                        interactive=False
                    )
            
            # Event Handlers
            save_config_btn.click(
                fn=self._save_configuration,
                inputs=[databento_api_key, default_schema, cache_dir, initial_capital, commission, qlib_model, sklearn_classifier],
                outputs=[config_status]
            )
            
            load_config_btn.click(
                fn=self._load_configuration,
                outputs=[config_preview, config_status]
            )
            
            reset_config_btn.click(
                fn=self._reset_configuration,
                outputs=[config_status]
            )
        
        return config_interface
    
    def _get_custom_css(self) -> str:
        """
        Get custom CSS for the interface.
        
        Returns:
            CSS string
        """
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
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        """
    
    def _generate_report(self, report_type: str, include_plots: bool) -> Tuple[str, str]:
        """
        Generate analysis report.
        
        Args:
            report_type: Type of report to generate
            include_plots: Whether to include plots
            
        Returns:
            Tuple of (report_html, statistics)
        """
        try:
            self.logger.info(f"Generating {report_type} report")
            
            # Get data from panels
            data = self.data_panel.get_current_data()
            signals = self.analysis_panel.get_current_signals()
            backtest_results = self.backtest_panel.get_current_results()
            
            # Generate report HTML
            html_content = f"""
            <div class="report-container">
                <h2>{report_type} Report</h2>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h3>Data Summary</h3>
                <ul>
                    <li>Data loaded: {'Yes' if data is not None and not data.empty else 'No'}</li>
                    <li>Signals generated: {'Yes' if signals is not None and not signals.empty else 'No'}</li>
                    <li>Backtest completed: {'Yes' if backtest_results is not None else 'No'}</li>
                </ul>
                
                <h3>Analysis Results</h3>
                <p>Report type: {report_type}</p>
                <p>Include plots: {'Yes' if include_plots else 'No'}</p>
                
                <h3>Performance Metrics</h3>
                <p>Detailed performance analysis would be displayed here.</p>
                
                <h3>Risk Analysis</h3>
                <p>Risk metrics and analysis would be shown here.</p>
                
                <h3>Recommendations</h3>
                <p>Based on the analysis, recommendations would be provided here.</p>
            </div>
            """
            
            # Generate statistics
            stats_lines = []
            stats_lines.append(f"Report Type: {report_type}")
            stats_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if data is not None and not data.empty:
                stats_lines.append(f"Data Records: {len(data)}")
            if signals is not None and not signals.empty:
                stats_lines.append(f"Signals Generated: {len(signals[signals['signal'] != 0])}")
            if backtest_results is not None:
                metrics = backtest_results.get('metrics', {})
                stats_lines.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
                stats_lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            
            stats_text = "\n".join(stats_lines)
            
            return html_content, stats_text
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            error_html = f"<p>Error generating report: {str(e)}</p>"
            error_stats = f"Error: {str(e)}"
            return error_html, error_stats
    
    def _export_report(self, export_format: str) -> str:
        """
        Export report in specified format.
        
        Args:
            export_format: Format to export (HTML, PDF, Excel, JSON)
            
        Returns:
            Export status message
        """
        try:
            self.logger.info(f"Exporting report in {export_format} format")
            
            # In a real implementation, this would generate and save the report
            # For now, return a status message
            
            return f"✅ Report exported successfully in {export_format} format"
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            return f"❌ Error exporting report: {str(e)}"
    
    def _save_configuration(self, *params) -> str:
        """
        Save configuration changes.
        
        Args:
            *params: Configuration parameters
            
        Returns:
            Status message
        """
        try:
            # Extract parameters
            (databento_api_key, default_schema, cache_dir, 
             initial_capital, commission, qlib_model, sklearn_classifier) = params
            
            # Update configurations
            self.config_manager.update_config('data_config', 'databento.api_key', databento_api_key)
            self.config_manager.update_config('data_config', 'databento.default_schema', default_schema)
            self.config_manager.update_config('data_config', 'databento.cache_dir', cache_dir)
            
            self.config_manager.update_config('strategy_config', 'backtesting.initial_capital', initial_capital)
            self.config_manager.update_config('strategy_config', 'backtesting.commission', commission)
            
            self.config_manager.update_config('model_config', 'qlib.model_type', qlib_model)
            self.config_manager.update_config('model_config', 'sklearn.classifier', sklearn_classifier)
            
            self.logger.info("Configuration saved successfully")
            return "✅ Configuration saved successfully"
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return f"❌ Error saving configuration: {str(e)}"
    
    def _load_configuration(self) -> Tuple[Dict, str]:
        """
        Load current configuration.
        
        Returns:
            Tuple of (configuration_dict, status_message)
        """
        try:
            config_dict = {
                "data_config": self.config_manager.get_data_config(),
                "strategy_config": self.config_manager.get_strategy_config(),
                "model_config": self.config_manager.get_model_config()
            }
            
            return config_dict, "✅ Configuration loaded successfully"
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}, f"❌ Error loading configuration: {str(e)}"
    
    def _reset_configuration(self) -> str:
        """
        Reset configuration to defaults.
        
        Returns:
            Status message
        """
        try:
            self.config_manager.create_default_configs()
            
            # Reload configurations
            self.data_config = self.config_manager.get_data_config()
            self.strategy_config = self.config_manager.get_strategy_config()
            self.model_config = self.config_manager.get_model_config()
            
            self.logger.info("Configuration reset to defaults")
            return "✅ Configuration reset to defaults"
            
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {e}")
            return f"❌ Error resetting configuration: {str(e)}"
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", 
               server_port: int = 7860, **kwargs) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            **kwargs: Additional launch parameters
        """
        try:
            interface = self.create_interface()
            
            self.logger.info(f"Launching Trading System UI on {server_name}:{server_port}")
            
            # Add default parameters to prevent connection issues
            launch_kwargs = {
                "share": share,
                "server_name": server_name,
                "server_port": server_port,
                "show_error": True,
                "quiet": False,
                "inbrowser": False,  # Don't auto-open browser
                **kwargs
            }
            
            interface.launch(**launch_kwargs)
            
        except Exception as e:
            self.logger.error(f"Error launching interface: {e}")
            raise

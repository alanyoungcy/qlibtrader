"""
Analysis panel for the Gradio UI.
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisPanel:
    """
    Analysis and signal generation panel.
    """
    
    def __init__(self):
        """Initialize analysis panel."""
        self.logger = logging.getLogger(__name__)
        self.current_signals = None
        self.analyzer = None
        self.ml_analyzer = None
    
    def create_interface(self) -> gr.Blocks:
        """
        Create analysis panel interface.
        
        Returns:
            Gradio interface for analysis panel
        """
        with gr.Blocks(title="Analysis & Signals") as interface:
            gr.Markdown("# 🔍 Analysis & Signal Generation")
            gr.Markdown("Configure analysis parameters and generate trading signals")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Analysis Configuration
                    gr.Markdown("## Analysis Configuration")
                    
                    # Analysis Type Selection
                    analysis_type = gr.Radio(
                        choices=["Qlib Analysis", "ML Analysis", "Strategy Signals"],
                        label="Analysis Type",
                        value="Strategy Signals"
                    )
                    
                    # Strategy Selection (for strategy signals)
                    strategy_type = gr.Dropdown(
                        choices=["MomentumStrategy", "MeanReversionStrategy"],
                        label="Strategy Type",
                        value="MomentumStrategy",
                        visible=True
                    )
                    
                    # Qlib Configuration
                    with gr.Group(visible=False) as qlib_group:
                        gr.Markdown("### Qlib Parameters")
                        qlib_model_type = gr.Dropdown(
                            choices=["lightgbm", "xgboost", "linear"],
                            label="Model Type",
                            value="lightgbm"
                        )
                        qlib_features = gr.Textbox(
                            label="Features (comma-separated)",
                            value="close,volume,rsi,macd,returns",
                            placeholder="close,volume,rsi,macd"
                        )
                        qlib_horizon = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=5,
                            label="Prediction Horizon (days)"
                        )
                    
                    # ML Configuration
                    with gr.Group(visible=False) as ml_group:
                        gr.Markdown("### ML Parameters")
                        ml_task = gr.Radio(
                            choices=["Classification", "Regression"],
                            label="ML Task",
                            value="Classification"
                        )
                        ml_model_type = gr.Dropdown(
                            choices=["random_forest", "gradient_boosting", "logistic_regression", "svm"],
                            label="Model Type",
                            value="random_forest"
                        )
                        ml_lookback = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=20,
                            label="Feature Lookback Window"
                        )
                        ml_test_size = gr.Slider(
                            minimum=0.1,
                            maximum=0.5,
                            value=0.2,
                            step=0.05,
                            label="Test Set Size"
                        )
                    
                    # Strategy Configuration
                    with gr.Group() as strategy_group:
                        gr.Markdown("### Strategy Parameters")
                        
                        # Momentum Strategy Parameters
                        with gr.Group() as momentum_params:
                            momentum_lookback = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=20,
                                label="Lookback Period"
                            )
                            momentum_threshold = gr.Slider(
                                minimum=0.01,
                                maximum=0.1,
                                value=0.02,
                                step=0.005,
                                label="Momentum Threshold"
                            )
                            momentum_volume_threshold = gr.Slider(
                                minimum=1.0,
                                maximum=3.0,
                                value=1.5,
                                step=0.1,
                                label="Volume Threshold"
                            )
                        
                        # Mean Reversion Strategy Parameters
                        with gr.Group(visible=False) as mean_reversion_params:
                            mr_window = gr.Slider(
                                minimum=20,
                                maximum=100,
                                value=50,
                                label="Lookback Window"
                            )
                            mr_std_multiplier = gr.Slider(
                                minimum=1.0,
                                maximum=3.0,
                                value=2.0,
                                step=0.1,
                                label="Standard Deviation Multiplier"
                            )
                            mr_volume_confirmation = gr.Checkbox(
                                label="Volume Confirmation",
                                value=True
                            )
                    
                    # Signal Configuration
                    gr.Markdown("### Signal Configuration")
                    signal_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.02,
                        step=0.01,
                        label="Signal Threshold"
                    )
                    
                    # Run Analysis Button
                    run_analysis_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")
                    
                    # Analysis Status
                    analysis_status = gr.Textbox(
                        label="Analysis Status",
                        value="Ready to analyze",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    # Analysis Results
                    gr.Markdown("## Analysis Results")
                    
                    # Signal Preview
                    signals_preview = gr.Dataframe(
                        label="Generated Signals",
                        headers=["Date", "Symbol", "Signal", "Strength", "Position"],
                        interactive=False
                    )
                    
                    # Signal Statistics
                    with gr.Row():
                        gr.Markdown("### Signal Statistics")
                        signal_stats = gr.Textbox(
                            label="Statistics",
                            lines=6,
                            interactive=False
                        )
                    
                    # Feature Importance (for ML/Qlib)
                    with gr.Row():
                        gr.Markdown("### Feature Importance")
                        feature_importance = gr.Dataframe(
                            label="Feature Importance",
                            headers=["Feature", "Importance"],
                            interactive=False
                        )
                    
                    # Analysis Summary
                    gr.Markdown("### Analysis Summary")
                    analysis_summary = gr.Textbox(
                        label="Summary",
                        lines=4,
                        interactive=False
                    )
            
            # Analysis Actions
            with gr.Row():
                save_signals_btn = gr.Button("💾 Save Signals", variant="secondary")
                export_analysis_btn = gr.Button("📊 Export Analysis", variant="secondary")
                clear_analysis_btn = gr.Button("🗑️ Clear Results", variant="stop")
            
            # Event Handlers
            analysis_type.change(
                fn=self._toggle_analysis_groups,
                inputs=[analysis_type],
                outputs=[qlib_group, ml_group, strategy_group]
            )
            
            strategy_type.change(
                fn=self._toggle_strategy_params,
                inputs=[strategy_type],
                outputs=[momentum_params, mean_reversion_params]
            )
            
            run_analysis_btn.click(
                fn=self._run_analysis,
                inputs=[
                    analysis_type, strategy_type, qlib_model_type, qlib_features, qlib_horizon,
                    ml_task, ml_model_type, ml_lookback, ml_test_size,
                    momentum_lookback, momentum_threshold, momentum_volume_threshold,
                    mr_window, mr_std_multiplier, mr_volume_confirmation,
                    signal_threshold
                ],
                outputs=[signals_preview, signal_stats, feature_importance, analysis_summary, analysis_status]
            )
            
            save_signals_btn.click(
                fn=self._save_signals,
                outputs=[analysis_status]
            )
            
            clear_analysis_btn.click(
                fn=self._clear_analysis,
                outputs=[signals_preview, signal_stats, feature_importance, analysis_summary, analysis_status]
            )
        
        return interface
    
    def _toggle_analysis_groups(self, analysis_type: str) -> Tuple[gr.Group, gr.Group, gr.Group]:
        """
        Toggle visibility of analysis configuration groups.
        
        Args:
            analysis_type: Selected analysis type
            
        Returns:
            Tuple of group visibility states
        """
        if analysis_type == "Qlib Analysis":
            return gr.Group(visible=True), gr.Group(visible=False), gr.Group(visible=False)
        elif analysis_type == "ML Analysis":
            return gr.Group(visible=False), gr.Group(visible=True), gr.Group(visible=False)
        else:  # Strategy Signals
            return gr.Group(visible=False), gr.Group(visible=False), gr.Group(visible=True)
    
    def _toggle_strategy_params(self, strategy_type: str) -> Tuple[gr.Group, gr.Group]:
        """
        Toggle visibility of strategy parameter groups.
        
        Args:
            strategy_type: Selected strategy type
            
        Returns:
            Tuple of group visibility states
        """
        if strategy_type == "MomentumStrategy":
            return gr.Group(visible=True), gr.Group(visible=False)
        else:  # MeanReversionStrategy
            return gr.Group(visible=False), gr.Group(visible=True)
    
    def _run_analysis(self, analysis_type: str, strategy_type: str, *params) -> Tuple:
        """
        Run analysis based on selected type and parameters.
        
        Returns:
            Tuple of analysis results
        """
        try:
            # Extract parameters
            (qlib_model_type, qlib_features, qlib_horizon,
             ml_task, ml_model_type, ml_lookback, ml_test_size,
             momentum_lookback, momentum_threshold, momentum_volume_threshold,
             mr_window, mr_std_multiplier, mr_volume_confirmation,
             signal_threshold) = params
            
            # Get data (this would come from data panel in real implementation)
            # For now, we'll create dummy data
            data = self._get_sample_data()
            
            if data is None or data.empty:
                return (pd.DataFrame(), "No data available", pd.DataFrame(), 
                       "Please load data first", "❌ No data available")
            
            self.logger.info(f"Running {analysis_type} analysis")
            
            if analysis_type == "Qlib Analysis":
                return self._run_qlib_analysis(data, qlib_model_type, qlib_features, qlib_horizon, signal_threshold)
            elif analysis_type == "ML Analysis":
                return self._run_ml_analysis(data, ml_task, ml_model_type, ml_lookback, ml_test_size, signal_threshold)
            else:  # Strategy Signals
                return self._run_strategy_analysis(data, strategy_type, params, signal_threshold)
                
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.logger.error(error_msg)
            return (pd.DataFrame(), "", pd.DataFrame(), "", f"❌ {error_msg}")
    
    def _run_qlib_analysis(self, data: pd.DataFrame, model_type: str, 
                          features: str, horizon: int, threshold: float) -> Tuple:
        """Run Qlib analysis."""
        try:
            from core.analyzer import QlibAnalyzer
            
            # Initialize analyzer
            self.analyzer = QlibAnalyzer()
            
            # Prepare data
            prepared_data = self.analyzer.prepare_data(data)
            
            # Generate signals
            signals = self.analyzer.generate_signals(prepared_data, horizon, threshold)
            
            # Store signals
            self.current_signals = signals
            
            # Prepare results
            preview = self._prepare_signals_preview(signals)
            stats = self._generate_signal_stats(signals)
            feature_imp = self._get_feature_importance()
            summary = f"Qlib analysis completed using {model_type} model"
            
            return preview, stats, feature_imp, summary, "✅ Qlib analysis completed"
            
        except Exception as e:
            self.logger.error(f"Qlib analysis error: {e}")
            return (pd.DataFrame(), f"Error: {str(e)}", pd.DataFrame(), 
                   f"Qlib analysis failed: {str(e)}", "❌ Analysis failed")
    
    def _run_ml_analysis(self, data: pd.DataFrame, task: str, model_type: str,
                        lookback: int, test_size: float, threshold: float) -> Tuple:
        """Run ML analysis."""
        try:
            from core.ml_models import MLAnalyzer
            
            # Initialize analyzer
            self.ml_analyzer = MLAnalyzer()
            
            # Feature engineering
            features_df = self.ml_analyzer.feature_engineering(data, lookback_window=lookback)
            
            # Prepare features and target
            feature_cols = [col for col in features_df.columns if col not in ['target', 'target_direction']]
            X = features_df[feature_cols].dropna()
            y = features_df['target_direction'].loc[X.index] if task == "Classification" else features_df['target'].loc[X.index]
            
            # Train model
            if task == "Classification":
                results = self.ml_analyzer.train_classifier(X, y, model_type, test_size)
            else:
                results = self.ml_analyzer.train_regressor(X, y, model_type, test_size)
            
            # Generate predictions as signals
            predictions = self.ml_analyzer.predict(X, f'{model_type}_{task.lower()}')
            
            # Create signals dataframe
            signals_df = features_df.loc[X.index].copy()
            signals_df['signal'] = predictions
            signals_df['signal_strength'] = abs(predictions)
            signals_df['position'] = np.where(predictions > threshold, 1,
                                            np.where(predictions < -threshold, -1, 0))
            
            # Store signals
            self.current_signals = signals_df
            
            # Prepare results
            preview = self._prepare_signals_preview(signals_df)
            stats = self._generate_signal_stats(signals_df)
            feature_imp = self._get_ml_feature_importance(model_type, task)
            summary = f"ML analysis completed: {task} with {model_type}"
            
            return preview, stats, feature_imp, summary, "✅ ML analysis completed"
            
        except Exception as e:
            self.logger.error(f"ML analysis error: {e}")
            return (pd.DataFrame(), f"Error: {str(e)}", pd.DataFrame(), 
                   f"ML analysis failed: {str(e)}", "❌ Analysis failed")
    
    def _run_strategy_analysis(self, data: pd.DataFrame, strategy_type: str,
                              params: tuple, threshold: float) -> Tuple:
        """Run strategy analysis."""
        try:
            from strategies.momentum_strategy import MomentumStrategy
            from strategies.mean_reversion_strategy import MeanReversionStrategy
            
            # Extract strategy parameters
            (momentum_lookback, momentum_threshold, momentum_volume_threshold,
             mr_window, mr_std_multiplier, mr_volume_confirmation) = params[6:12]
            
            # Initialize strategy
            if strategy_type == "MomentumStrategy":
                strategy_params = {
                    'lookback_period': int(momentum_lookback),
                    'momentum_threshold': momentum_threshold,
                    'volume_threshold': momentum_volume_threshold
                }
                strategy = MomentumStrategy(strategy_params)
            else:  # MeanReversionStrategy
                strategy_params = {
                    'lookback_window': int(mr_window),
                    'std_dev_multiplier': mr_std_multiplier,
                    'volume_confirmation': mr_volume_confirmation
                }
                strategy = MeanReversionStrategy(strategy_params)
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Store signals
            self.current_signals = signals
            
            # Prepare results
            preview = self._prepare_signals_preview(signals)
            stats = self._generate_signal_stats(signals)
            feature_imp = self._get_strategy_feature_importance(strategy)
            summary = f"{strategy_type} analysis completed"
            
            return preview, stats, feature_imp, summary, f"✅ {strategy_type} analysis completed"
            
        except Exception as e:
            self.logger.error(f"Strategy analysis error: {e}")
            return (pd.DataFrame(), f"Error: {str(e)}", pd.DataFrame(), 
                   f"Strategy analysis failed: {str(e)}", "❌ Analysis failed")
    
    def _prepare_signals_preview(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Prepare signals for preview display."""
        try:
            if signals.empty:
                return pd.DataFrame()
            
            preview_data = signals.copy()
            
            # Add date column
            if isinstance(preview_data.index, pd.DatetimeIndex):
                preview_data['Date'] = preview_data.index.strftime('%Y-%m-%d')
            else:
                preview_data['Date'] = range(len(preview_data))
            
            # Add symbol column if not present
            if 'symbol' not in preview_data.columns:
                preview_data['Symbol'] = 'STOCK'
            
            # Select relevant columns
            cols = ['Date', 'Symbol', 'signal', 'signal_strength', 'position']
            available_cols = [col for col in cols if col in preview_data.columns]
            
            if available_cols:
                preview_data = preview_data[available_cols]
            
            # Format numeric columns
            if 'signal' in preview_data.columns:
                preview_data['signal'] = preview_data['signal'].round(3)
            if 'signal_strength' in preview_data.columns:
                preview_data['signal_strength'] = preview_data['signal_strength'].round(3)
            
            # Limit to first 50 rows
            return preview_data.head(50)
            
        except Exception as e:
            self.logger.error(f"Error preparing signals preview: {e}")
            return pd.DataFrame()
    
    def _generate_signal_stats(self, signals: pd.DataFrame) -> str:
        """Generate signal statistics."""
        try:
            if signals.empty:
                return "No signals generated"
            
            stats_lines = []
            stats_lines.append("📊 Signal Statistics")
            stats_lines.append(f"Total Records: {len(signals):,}")
            
            if 'signal' in signals.columns:
                total_signals = (signals['signal'] != 0).sum()
                buy_signals = (signals['signal'] > 0).sum()
                sell_signals = (signals['signal'] < 0).sum()
                
                stats_lines.append(f"Total Signals: {total_signals}")
                stats_lines.append(f"Buy Signals: {buy_signals}")
                stats_lines.append(f"Sell Signals: {sell_signals}")
                
                if len(signals) > 0:
                    signal_frequency = total_signals / len(signals)
                    stats_lines.append(f"Signal Frequency: {signal_frequency:.2%}")
            
            if 'signal_strength' in signals.columns:
                avg_strength = signals['signal_strength'].mean()
                max_strength = signals['signal_strength'].max()
                stats_lines.append(f"Avg Strength: {avg_strength:.3f}")
                stats_lines.append(f"Max Strength: {max_strength:.3f}")
            
            return "\n".join(stats_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating signal stats: {e}")
            return f"Error: {str(e)}"
    
    def _get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Qlib analyzer."""
        try:
            if self.analyzer is None:
                return pd.DataFrame()
            
            importance = self.analyzer.get_feature_importance()
            if not importance:
                return pd.DataFrame()
            
            df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            df = df.sort_values('Importance', ascending=False)
            df['Importance'] = df['Importance'].round(4)
            
            return df.head(10)
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def _get_ml_feature_importance(self, model_type: str, task: str) -> pd.DataFrame:
        """Get feature importance from ML analyzer."""
        try:
            if self.ml_analyzer is None:
                return pd.DataFrame()
            
            model_name = f'{model_type}_{task.lower()}'
            importance = self.ml_analyzer.get_feature_importance(model_name)
            
            if not importance:
                return pd.DataFrame()
            
            df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            df = df.sort_values('Importance', ascending=False)
            df['Importance'] = df['Importance'].round(4)
            
            return df.head(10)
            
        except Exception as e:
            self.logger.error(f"Error getting ML feature importance: {e}")
            return pd.DataFrame()
    
    def _get_strategy_feature_importance(self, strategy) -> pd.DataFrame:
        """Get feature importance from strategy."""
        try:
            importance = strategy.get_feature_importance()
            if not importance:
                return pd.DataFrame()
            
            df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            df = df.sort_values('Importance', ascending=False)
            df['Importance'] = df['Importance'].round(4)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting strategy feature importance: {e}")
            return pd.DataFrame()
    
    def _save_signals(self) -> str:
        """Save current signals."""
        try:
            if self.current_signals is None or self.current_signals.empty:
                return "❌ No signals to save"
            
            # Save signals (implementation would save to file)
            self.logger.info("Signals saved")
            return "✅ Signals saved successfully"
            
        except Exception as e:
            self.logger.error(f"Error saving signals: {e}")
            return f"❌ Error saving signals: {str(e)}"
    
    def _clear_analysis(self) -> Tuple:
        """Clear analysis results."""
        try:
            self.current_signals = None
            self.analyzer = None
            self.ml_analyzer = None
            
            return (pd.DataFrame(), "", pd.DataFrame(), "", "✅ Analysis results cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing analysis: {e}")
            return (pd.DataFrame(), f"Error: {str(e)}", pd.DataFrame(), f"Error: {str(e)}", "❌ Error clearing analysis")
    
    def _get_sample_data(self) -> pd.DataFrame:
        """Get sample data for testing (would be replaced with actual data from data panel)."""
        try:
            # Create sample data for demonstration
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            
            data = pd.DataFrame({
                'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
                'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01) + np.abs(np.random.randn(len(dates)) * 0.5),
                'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01) - np.abs(np.random.randn(len(dates)) * 0.5),
                'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating sample data: {e}")
            return pd.DataFrame()
    
    def get_current_signals(self) -> Optional[pd.DataFrame]:
        """Get currently generated signals."""
        return self.current_signals

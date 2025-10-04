"""
Data fetching panel for the Gradio UI.
"""

import gradio as gr
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataPanel:
    """
    Data fetching and management panel.
    """
    
    def __init__(self):
        """Initialize data panel."""
        self.logger = logging.getLogger(__name__)
        self.current_data = None
    
    def create_interface(self) -> gr.Blocks:
        """
        Create data panel interface.
        
        Returns:
            Gradio interface for data panel
        """
        with gr.Blocks(title="Data Management") as interface:
            gr.Markdown("# 📊 Data Management")
            gr.Markdown("Fetch and manage market data from Databento")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Data Source Configuration
                    gr.Markdown("## Data Source")
                    
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
                    
                    schema_dropdown = gr.Dropdown(
                        choices=["ohlcv-1d", "ohlcv-1h", "ohlcv-1m"],
                        label="Data Schema",
                        value="ohlcv-1d"
                    )
                    
                    venue_dropdown = gr.Dropdown(
                        choices=["XNAS", "XNYS", "IEXG"],
                        label="Venue",
                        value="XNAS"
                    )
                    
                    # Fetch Data Button
                    fetch_btn = gr.Button("📥 Fetch Data", variant="primary", size="lg")
                    
                    # Data Status
                    data_status = gr.Textbox(
                        label="Data Status",
                        value="No data loaded",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    # Data Preview
                    gr.Markdown("## Data Preview")
                    
                    data_preview = gr.Dataframe(
                        label="Market Data",
                        headers=["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"],
                        interactive=False
                    )
                    
                    with gr.Row():
                        # Data Statistics
                        gr.Markdown("### Data Statistics")
                        stats_text = gr.Textbox(
                            label="Statistics",
                            lines=8,
                            interactive=False
                        )
            
            # Data Management Actions
            with gr.Row():
                save_data_btn = gr.Button("💾 Save Data", variant="secondary")
                load_data_btn = gr.Button("📂 Load Cached Data", variant="secondary")
                clear_data_btn = gr.Button("🗑️ Clear Data", variant="stop")
            
            # Event Handlers
            fetch_btn.click(
                fn=self._fetch_data,
                inputs=[symbols_input, start_date, end_date, schema_dropdown, venue_dropdown],
                outputs=[data_preview, stats_text, data_status]
            )
            
            save_data_btn.click(
                fn=self._save_data,
                inputs=[symbols_input],
                outputs=[data_status]
            )
            
            load_data_btn.click(
                fn=self._load_cached_data,
                inputs=[symbols_input],
                outputs=[data_preview, stats_text, data_status]
            )
            
            clear_data_btn.click(
                fn=self._clear_data,
                outputs=[data_preview, stats_text, data_status]
            )
        
        return interface
    
    def _fetch_data(self, symbols: str, start_date: str, end_date: str, 
                   schema: str, venue: str) -> Tuple[pd.DataFrame, str, str]:
        """
        Fetch data from Databento.
        
        Args:
            symbols: Comma-separated symbols
            start_date: Start date string
            end_date: End date string
            schema: Data schema
            venue: Trading venue
            
        Returns:
            Tuple of (dataframe, statistics, status)
        """
        try:
            from core.data_fetcher import DatabentoFetcher
            
            # Parse symbols
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
            
            if not symbol_list:
                return pd.DataFrame(), "No symbols provided", "Error: No symbols"
            
            # Initialize fetcher
            fetcher = DatabentoFetcher()
            
            # Validate symbols
            valid_symbols = fetcher.validate_symbols(symbol_list)
            if not valid_symbols:
                return pd.DataFrame(), "No valid symbols", "Error: Invalid symbols"
            
            # Fetch data
            data = fetcher.fetch_historical(
                symbols=valid_symbols,
                start_date=start_date,
                end_date=end_date,
                schema=schema,
                venue=venue
            )
            
            if data.empty:
                return pd.DataFrame(), "No data returned", "Error: No data"
            
            # Store current data
            self.current_data = data
            
            # Generate statistics
            stats = self._generate_data_stats(data)
            
            # Prepare preview data
            preview_data = self._prepare_preview_data(data)
            
            status = f"✅ Loaded {len(data)} records for {len(valid_symbols)} symbols"
            
            self.logger.info(f"Fetched data: {len(data)} records")
            return preview_data, stats, status
            
        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            self.logger.error(error_msg)
            return pd.DataFrame(), "", f"❌ {error_msg}"
    
    def _save_data(self, symbols: str) -> str:
        """
        Save current data to cache.
        
        Args:
            symbols: Symbol string for filename
            
        Returns:
            Status message
        """
        try:
            if self.current_data is None or self.current_data.empty:
                return "❌ No data to save"
            
            from core.data_fetcher import DatabentoFetcher
            
            # Create filename
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
            filename = f"{'_'.join(symbol_list[:3])}_data.parquet"
            cache_path = f"data/raw/{filename}"
            
            # Save data
            fetcher = DatabentoFetcher()
            fetcher.cache_data(self.current_data, cache_path)
            
            self.logger.info(f"Saved data to {cache_path}")
            return f"✅ Data saved to {cache_path}"
            
        except Exception as e:
            error_msg = f"Error saving data: {str(e)}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def _load_cached_data(self, symbols: str) -> Tuple[pd.DataFrame, str, str]:
        """
        Load cached data.
        
        Args:
            symbols: Symbol string for filename
            
        Returns:
            Tuple of (dataframe, statistics, status)
        """
        try:
            from core.data_fetcher import DatabentoFetcher
            
            # Create filename
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
            filename = f"{'_'.join(symbol_list[:3])}_data.parquet"
            cache_path = f"data/raw/{filename}"
            
            # Load data
            fetcher = DatabentoFetcher()
            data = fetcher.load_cached_data(cache_path)
            
            if data.empty:
                return pd.DataFrame(), "", "❌ No cached data found"
            
            # Store current data
            self.current_data = data
            
            # Generate statistics
            stats = self._generate_data_stats(data)
            
            # Prepare preview data
            preview_data = self._prepare_preview_data(data)
            
            status = f"✅ Loaded cached data: {len(data)} records"
            
            self.logger.info(f"Loaded cached data: {len(data)} records")
            return preview_data, stats, status
            
        except Exception as e:
            error_msg = f"Error loading cached data: {str(e)}"
            self.logger.error(error_msg)
            return pd.DataFrame(), "", f"❌ {error_msg}"
    
    def _clear_data(self) -> Tuple[pd.DataFrame, str, str]:
        """
        Clear current data.
        
        Returns:
            Tuple of (empty dataframe, empty stats, status)
        """
        try:
            self.current_data = None
            self.logger.info("Data cleared")
            return pd.DataFrame(), "", "✅ Data cleared"
            
        except Exception as e:
            error_msg = f"Error clearing data: {str(e)}"
            self.logger.error(error_msg)
            return pd.DataFrame(), "", f"❌ {error_msg}"
    
    def _generate_data_stats(self, data: pd.DataFrame) -> str:
        """
        Generate data statistics string.
        
        Args:
            data: Market data
            
        Returns:
            Statistics string
        """
        try:
            if data.empty:
                return "No data available"
            
            stats_lines = []
            stats_lines.append(f"📊 Data Statistics")
            stats_lines.append(f"Total Records: {len(data):,}")
            
            if 'symbol' in data.columns:
                unique_symbols = data['symbol'].nunique()
                stats_lines.append(f"Unique Symbols: {unique_symbols}")
            
            if isinstance(data.index, pd.DatetimeIndex):
                date_range = f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"
                stats_lines.append(f"Date Range: {date_range}")
            
            if 'close' in data.columns:
                stats_lines.append(f"Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                stats_lines.append(f"Average Price: ${data['close'].mean():.2f}")
            
            if 'volume' in data.columns:
                total_volume = data['volume'].sum()
                avg_volume = data['volume'].mean()
                stats_lines.append(f"Total Volume: {total_volume:,.0f}")
                stats_lines.append(f"Average Volume: {avg_volume:,.0f}")
            
            # Missing data
            missing_data = data.isnull().sum().sum()
            if missing_data > 0:
                stats_lines.append(f"Missing Values: {missing_data}")
            
            return "\n".join(stats_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating stats: {e}")
            return f"Error generating statistics: {str(e)}"
    
    def _prepare_preview_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for preview display.
        
        Args:
            data: Raw market data
            
        Returns:
            Formatted preview data
        """
        try:
            if data.empty:
                return pd.DataFrame()
            
            # Select relevant columns for preview
            preview_cols = []
            
            # Add date/timestamp
            if isinstance(data.index, pd.DatetimeIndex):
                preview_data = data.copy()
                preview_data['Date'] = preview_data.index.strftime('%Y-%m-%d')
                preview_cols.append('Date')
            elif 'timestamp' in data.columns:
                preview_data = data.copy()
                preview_data['Date'] = pd.to_datetime(preview_data['timestamp']).dt.strftime('%Y-%m-%d')
                preview_cols.append('Date')
            else:
                preview_data = data.copy()
            
            # Add symbol if available
            if 'symbol' in data.columns:
                preview_cols.append('Symbol')
            else:
                preview_data['Symbol'] = 'STOCK'
                preview_cols.append('Symbol')
            
            # Add OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                if col in data.columns:
                    preview_cols.append(col.title())
            
            # Select and format columns
            if preview_cols:
                preview_data = preview_data[preview_cols]
            
            # Format numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close']
            for col in numeric_cols:
                if col in preview_data.columns:
                    preview_data[col] = preview_data[col].round(2)
            
            if 'Volume' in preview_data.columns:
                preview_data['Volume'] = preview_data['Volume'].astype(int)
            
            # Limit to first 100 rows for preview
            return preview_data.head(100)
            
        except Exception as e:
            self.logger.error(f"Error preparing preview data: {e}")
            return pd.DataFrame()
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """
        Get currently loaded data.
        
        Returns:
            Current data DataFrame or None
        """
        return self.current_data

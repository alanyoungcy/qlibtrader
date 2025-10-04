"""
Databento data fetcher for historical and live market data.
"""

import os
import pandas as pd
from typing import List, Dict, Optional
import databento as db
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class DatabentoFetcher:
    """
    Handles all Databento API interactions for fetching market data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Databento client.
        
        Args:
            api_key: Databento API key. If None, will try to get from environment.
        """
        if api_key is None:
            api_key = os.getenv('DATABENTO_API_KEY')
            if api_key is None:
                raise ValueError("API key must be provided or set in DATABENTO_API_KEY environment variable")
        
        self.client = db.Historical(api_key)
        self.logger = logging.getLogger(__name__)
        
    def fetch_historical(self, 
                        symbols: List[str],
                        start_date: str,
                        end_date: str,
                        schema: str = 'ohlcv-1d',
                        venue: str = 'XNAS') -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            schema: Data schema (e.g., 'ohlcv-1d', 'ohlcv-1h', 'ohlcv-1m')
            venue: Exchange venue (default: XNAS for NASDAQ)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching historical data for {symbols} from {start_date} to {end_date}")
            
            # Convert date strings to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Fetch data from Databento
            data = self.client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=symbols,
                schema=schema,
                start=start_dt,
                end=end_dt,
                venue=venue
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            # Standardize column names
            if not df.empty:
                df = self._standardize_columns(df)
                df = df.sort_index()
            
            self.logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise
    
    def fetch_live(self, symbols: List[str], schema: str = 'ohlcv-1d') -> pd.DataFrame:
        """
        Fetch live data stream (placeholder for real-time implementation).
        
        Args:
            symbols: List of symbols to fetch
            schema: Data schema
            
        Returns:
            DataFrame with live data
        """
        # Note: This would require a live streaming setup
        # For now, return empty DataFrame as placeholder
        self.logger.warning("Live data fetching not yet implemented")
        return pd.DataFrame()
    
    def cache_data(self, data: pd.DataFrame, path: str) -> None:
        """
        Cache fetched data locally.
        
        Args:
            data: DataFrame to cache
            path: File path to save data
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save as parquet for efficiency
            if path.endswith('.csv'):
                data.to_csv(path)
            else:
                # Default to parquet
                data.to_parquet(path)
            
            self.logger.info(f"Data cached to {path}")
            
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")
            raise
    
    def load_cached_data(self, path: str) -> pd.DataFrame:
        """
        Load previously cached data.
        
        Args:
            path: File path to load data from
            
        Returns:
            DataFrame with cached data
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cached data not found at {path}")
            
            if path.endswith('.csv'):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                # Default to parquet
                df = pd.read_parquet(path)
            
            self.logger.info(f"Loaded cached data from {path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for consistency.
        
        Args:
            df: Raw DataFrame from Databento
            
        Returns:
            DataFrame with standardized columns
        """
        # Map common Databento column names to standard names
        column_mapping = {
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'symbol': 'symbol',
            'ts_event': 'timestamp'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)
        
        # Set timestamp as index if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def get_available_symbols(self, venue: str = 'XNAS') -> List[str]:
        """
        Get list of available symbols for a venue.
        
        Args:
            venue: Exchange venue
            
        Returns:
            List of available symbols
        """
        try:
            # This would typically query Databento's symbol metadata
            # For now, return common symbols as placeholder
            common_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'META', 'NVDA', 'NFLX', 'AMD', 'INTC'
            ]
            self.logger.info(f"Returning {len(common_symbols)} common symbols")
            return common_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate that symbols are available for trading.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of valid symbols
        """
        try:
            available_symbols = self.get_available_symbols()
            valid_symbols = [s for s in symbols if s in available_symbols]
            
            invalid_symbols = [s for s in symbols if s not in available_symbols]
            if invalid_symbols:
                self.logger.warning(f"Invalid symbols found: {invalid_symbols}")
            
            return valid_symbols
            
        except Exception as e:
            self.logger.error(f"Error validating symbols: {e}")
            return symbols  # Return original list if validation fails

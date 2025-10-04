"""
Configuration management utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration files and environment variables.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.logger = logging.getLogger(__name__)
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_name: str, 
                   config_type: str = "yaml") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of configuration file (without extension)
            config_type: Type of config file ('yaml' or 'json')
            
        Returns:
            Configuration dictionary
        """
        try:
            config_path = self.config_dir / f"{config_name}.{config_type}"
            
            if not config_path.exists():
                self.logger.warning(f"Config file {config_path} not found")
                return {}
            
            with open(config_path, 'r') as file:
                if config_type == 'yaml':
                    config = yaml.safe_load(file)
                else:
                    import json
                    config = json.load(file)
            
            # Resolve environment variables
            config = self._resolve_env_vars(config)
            
            self.configs[config_name] = config
            self.logger.info(f"Loaded config: {config_name}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config {config_name}: {e}")
            return {}
    
    def save_config(self, config_name: str, 
                   config_data: Dict[str, Any],
                   config_type: str = "yaml") -> bool:
        """
        Save configuration to file.
        
        Args:
            config_name: Name of configuration file
            config_data: Configuration data to save
            config_type: Type of config file ('yaml' or 'json')
            
        Returns:
            True if successful
        """
        try:
            config_path = self.config_dir / f"{config_name}.{config_type}"
            
            with open(config_path, 'w') as file:
                if config_type == 'yaml':
                    yaml.dump(config_data, file, default_flow_style=False, indent=2)
                else:
                    import json
                    json.dump(config_data, file, indent=2)
            
            self.configs[config_name] = config_data
            self.logger.info(f"Saved config: {config_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving config {config_name}: {e}")
            return False
    
    def get_config(self, config_name: str, 
                  key: Optional[str] = None,
                  default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            config_name: Name of configuration
            key: Specific key to retrieve (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            # Load config if not already loaded
            if config_name not in self.configs:
                self.load_config(config_name)
            
            config = self.configs.get(config_name, {})
            
            if key is None:
                return config
            
            # Navigate nested keys using dot notation
            keys = key.split('.')
            value = config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting config {config_name}.{key}: {e}")
            return default
    
    def update_config(self, config_name: str, 
                     key: str, value: Any,
                     save: bool = True) -> bool:
        """
        Update configuration value.
        
        Args:
            config_name: Name of configuration
            key: Key to update (dot notation supported)
            value: New value
            save: Whether to save to file
            
        Returns:
            True if successful
        """
        try:
            # Load config if not already loaded
            if config_name not in self.configs:
                self.load_config(config_name)
            
            config = self.configs.get(config_name, {})
            
            # Navigate nested keys
            keys = key.split('.')
            current = config
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            
            # Update stored config
            self.configs[config_name] = config
            
            # Save to file if requested
            if save:
                self.save_config(config_name, config)
            
            self.logger.info(f"Updated config {config_name}.{key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating config {config_name}.{key}: {e}")
            return False
    
    def _resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment variables in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with resolved environment variables
        """
        def resolve_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            else:
                return value
        
        return resolve_value(config)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get_config('data_config', default={
            'databento': {
                'api_key': '${DATABENTO_API_KEY}',
                'default_schema': 'ohlcv-1d',
                'cache_dir': './data/raw'
            },
            'storage': {
                'raw_data': './data/raw',
                'processed_data': './data/processed',
                'results': './data/results'
            }
        })
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return self.get_config('strategy_config', default={
            'strategies': {
                'momentum': {
                    'lookback': 20,
                    'threshold': 0.02,
                    'volume_threshold': 1.5
                },
                'mean_reversion': {
                    'window': 50,
                    'std_dev': 2.0,
                    'volume_confirmation': True
                }
            },
            'backtesting': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005
            }
        })
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get_config('model_config', default={
            'qlib': {
                'model_type': 'lightgbm',
                'features': ['close', 'volume', 'rsi', 'macd']
            },
            'sklearn': {
                'classifier': 'RandomForestClassifier',
                'regressor': 'GradientBoostingRegressor',
                'cv_folds': 5
            }
        })
    
    def create_default_configs(self) -> None:
        """Create default configuration files."""
        try:
            # Data config
            data_config = {
                'databento': {
                    'api_key': '${DATABENTO_API_KEY}',
                    'default_schema': 'ohlcv-1d',
                    'cache_dir': './data/raw'
                },
                'storage': {
                    'raw_data': './data/raw',
                    'processed_data': './data/processed',
                    'results': './data/results'
                }
            }
            self.save_config('data_config', data_config)
            
            # Strategy config
            strategy_config = {
                'strategies': {
                    'momentum': {
                        'lookback': 20,
                        'threshold': 0.02,
                        'volume_threshold': 1.5,
                        'rsi_oversold': 30,
                        'rsi_overbought': 70
                    },
                    'mean_reversion': {
                        'window': 50,
                        'std_dev': 2.0,
                        'volume_confirmation': True,
                        'rsi_oversold': 30,
                        'rsi_overbought': 70
                    }
                },
                'backtesting': {
                    'initial_capital': 100000,
                    'commission': 0.001,
                    'slippage': 0.0005,
                    'max_position_size': 1.0
                }
            }
            self.save_config('strategy_config', strategy_config)
            
            # Model config
            model_config = {
                'qlib': {
                    'model_type': 'lightgbm',
                    'features': ['close', 'volume', 'rsi', 'macd', 'returns'],
                    'train_period': '2020-01-01',
                    'test_period': '2023-01-01'
                },
                'sklearn': {
                    'classifier': 'RandomForestClassifier',
                    'regressor': 'GradientBoostingRegressor',
                    'cv_folds': 5,
                    'test_size': 0.2
                }
            }
            self.save_config('model_config', model_config)
            
            self.logger.info("Created default configuration files")
            
        except Exception as e:
            self.logger.error(f"Error creating default configs: {e}")
    
    def list_configs(self) -> list:
        """List available configuration files."""
        try:
            config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
            return [f.stem for f in config_files]
        except Exception as e:
            self.logger.error(f"Error listing configs: {e}")
            return []
    
    def delete_config(self, config_name: str) -> bool:
        """
        Delete configuration file.
        
        Args:
            config_name: Name of configuration to delete
            
        Returns:
            True if successful
        """
        try:
            config_path = self.config_dir / f"{config_name}.yaml"
            
            if config_path.exists():
                config_path.unlink()
                self.logger.info(f"Deleted config: {config_name}")
            
            # Remove from memory
            if config_name in self.configs:
                del self.configs[config_name]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting config {config_name}: {e}")
            return False

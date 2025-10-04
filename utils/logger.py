"""
Logging utilities for the trading system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class TradingSystemFormatter(logging.Formatter):
    """
    Custom formatter for trading system logs.
    """
    
    def __init__(self, include_context: bool = True):
        """
        Initialize formatter.
        
        Args:
            include_context: Whether to include additional context in logs
        """
        self.include_context = include_context
        
        # Define format strings
        self.simple_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.detailed_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        )
        
        super().__init__(self.detailed_format if include_context else self.simple_format)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log string
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def setup_logger(name: str = "trading_system",
                level: str = "INFO",
                log_file: Optional[str] = None,
                log_dir: str = "logs",
                format_type: str = "standard",
                max_file_size: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5,
                console_output: bool = True) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Log file name (optional)
        log_dir: Directory for log files
        format_type: Format type ('standard', 'detailed', 'json')
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Set up formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        include_context = format_type == "detailed"
        formatter = TradingSystemFormatter(include_context=include_context)
    
    # File handler
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        file_path = log_path / log_file
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "trading_system") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_performance(func):
    """
    Decorator to log function performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    return wrapper


class TradingSystemLogger:
    """
    Centralized logger for trading system components.
    """
    
    def __init__(self, component_name: str):
        """
        Initialize component logger.
        
        Args:
            component_name: Name of the component
        """
        self.component_name = component_name
        self.logger = get_logger(f"trading_system.{component_name}")
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(f"[{self.component_name}] {message}", extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(f"[{self.component_name}] {message}", extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(f"[{self.component_name}] {message}", extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(f"[{self.component_name}] {message}", extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(f"[{self.component_name}] {message}", extra=kwargs)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade information."""
        self.info("Trade executed", trade=trade_data)
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log signal generation."""
        self.info("Signal generated", signal=signal_data)
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """Log performance metrics."""
        self.info("Performance metrics", performance=performance_data)
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log error with context."""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        self.error("Error occurred", error=error_data, exc_info=True)


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on configuration.
    
    Args:
        config: Logging configuration
    """
    try:
        # Set up main logger
        main_logger = setup_logger(
            name=config.get('name', 'trading_system'),
            level=config.get('level', 'INFO'),
            log_file=config.get('log_file'),
            log_dir=config.get('log_dir', 'logs'),
            format_type=config.get('format_type', 'standard'),
            max_file_size=config.get('max_file_size', 10 * 1024 * 1024),
            backup_count=config.get('backup_count', 5),
            console_output=config.get('console_output', True)
        )
        
        # Set up component loggers
        components = config.get('components', {})
        for component_name, component_config in components.items():
            setup_logger(
                name=f"trading_system.{component_name}",
                level=component_config.get('level', 'INFO'),
                log_file=component_config.get('log_file'),
                log_dir=component_config.get('log_dir', 'logs'),
                format_type=component_config.get('format_type', 'standard'),
                console_output=component_config.get('console_output', False)
            )
        
        main_logger.info("Logging configuration completed")
        
    except Exception as e:
        print(f"Error configuring logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)


# Initialize default logger
default_logger = setup_logger()

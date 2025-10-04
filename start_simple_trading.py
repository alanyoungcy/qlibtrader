#!/usr/bin/env python3
"""
Startup script for the simplified trading system.
"""

import os
import sys
import logging
import argparse
import signal
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger, configure_logging


def setup_logging_config():
    """Setup logging configuration."""
    try:
        # Default logging configuration
        logging_config = {
            "name": "simple_trading_launcher",
            "level": "INFO",
            "log_file": "trading_system.log",
            "log_dir": "logs",
            "format_type": "standard",
            "console_output": True,
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5
        }
        
        configure_logging(logging_config)
        logger = setup_logger("simple_launcher")
        logger.info("Logging configuration completed")
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return None


def setup_environment():
    """Set up environment variables and directories."""
    try:
        # Create necessary directories
        directories = [
            "data/raw",
            "data/processed", 
            "data/results",
            "logs",
            "configs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Load environment variables from .env file if it exists
        env_file = Path(".env")
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"Loaded environment variables from {env_file}")
        
        print("Environment setup completed")
        return True
        
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "numpy",
        "pandas", 
        "sklearn",  # scikit-learn is imported as sklearn
        "gradio",
        "plotly",
        "yaml"      # pyyaml is imported as yaml
    ]
    
    missing_required = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    if missing_required:
        print("❌ Missing required packages:")
        for package in missing_required:
            print(f"   - {package}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True


def print_welcome_message():
    """Print welcome message and system information."""
    print("\n" + "="*60)
    print("🚀 AI-Powered Trading System - Simplified")
    print("="*60)
    print("Built with Databento, Qlib, Scikit-learn, and Gradio")
    print("\nFeatures:")
    print("  📊 Data fetching and management")
    print("  🔍 Analysis and signal generation") 
    print("  📈 Backtesting engine")
    print("  🎯 Trading strategies")
    print("  📱 Simplified web interface")
    print("\n" + "="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simplified Trading System Launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    try:
        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        
        # Print welcome message
        print_welcome_message()
        
        # Setup logging
        logger = setup_logging_config()
        if logger is None:
            return 1
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Check for API key
        if not os.getenv('DATABENTO_API_KEY'):
            logger.warning("DATABENTO_API_KEY not found in environment variables")
            logger.warning("Some features may not work without a valid API key")
        
        # Import and start the simplified UI
        from ui.simple_gradio_app import SimpleTradingSystemUI
        
        logger.info("Starting simplified trading system interface...")
        
        # Initialize UI
        ui = SimpleTradingSystemUI()
        
        print(f"\n🌐 Starting Simplified Trading System Interface")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        print(f"\n📱 Open your browser and navigate to:")
        print(f"   http://{args.host}:{args.port}")
        print(f"\n💡 Press Ctrl+C to stop the server")
        
        # Launch interface
        ui.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Trading System stopped by user")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error starting Trading System: {e}")
        if logger:
            logger.error(f"Error starting application: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

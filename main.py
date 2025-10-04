"""
Main entry point for the Trading System application.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ui.gradio_app import TradingSystemUI
from utils.logger import setup_logger, configure_logging
from utils.config import ConfigManager


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
        
    except Exception as e:
        print(f"Error setting up environment: {e}")
        sys.exit(1)


def setup_logging_config():
    """Set up logging configuration."""
    try:
        config_manager = ConfigManager()
        
        # Default logging configuration
        logging_config = {
            "name": "trading_system",
            "level": "INFO",
            "log_file": "trading_system.log",
            "log_dir": "logs",
            "format_type": "standard",
            "console_output": True,
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "components": {
                "data_fetcher": {"level": "INFO", "log_file": "data_fetcher.log"},
                "analyzer": {"level": "INFO", "log_file": "analyzer.log"},
                "backtester": {"level": "INFO", "log_file": "backtester.log"},
                "ui": {"level": "INFO", "log_file": "ui.log"}
            }
        }
        
        configure_logging(logging_config)
        
        logger = setup_logger("main")
        logger.info("Logging configuration completed")
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Trading System with Databento, Qlib, and Machine Learning"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the interface"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing configuration files (default: configs)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with debug features"
    )
    
    parser.add_argument(
        "--server",
        type=str,
        choices=["gradio", "uvicorn"],
        default="gradio",
        help="Server type to use (default: gradio)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (uvicorn only)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (uvicorn only)"
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only run setup and exit"
    )
    
    return parser.parse_args()


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
    
    optional_packages = [
        "databento",
        "qlib",
        "backtrader",
        "lightgbm"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print("❌ Missing required packages:")
        for package in missing_required:
            print(f"   - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print("⚠️  Missing optional packages (some features may not work):")
        for package in missing_optional:
            print(f"   - {package}")
        print("\nInstall with: pip install -r requirements.txt")
    
    print("✅ All required dependencies are installed")
    return True


def print_welcome_message():
    """Print welcome message and system information."""
    print("\n" + "="*60)
    print("🚀 AI-Powered Trading System")
    print("="*60)
    print("Built with Databento, Qlib, Scikit-learn, and Gradio")
    print("\nFeatures:")
    print("  📊 Data fetching from Databento")
    print("  🔍 Qlib analysis and ML predictions") 
    print("  📈 Advanced backtesting engine")
    print("  🎯 Multiple trading strategies")
    print("  📱 Modern web interface")
    print("\n" + "="*60)


def main():
    """Main application entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print welcome message
        print_welcome_message()
        
        # Setup environment
        setup_environment()
        
        # Setup logging
        logger = setup_logging_config()
        
        if args.setup_only:
            print("✅ Setup completed. Exiting as requested.")
            return 0
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Check for API key
        if not os.getenv('DATABENTO_API_KEY'):
            print("⚠️  Warning: DATABENTO_API_KEY not found in environment variables")
            print("   Some features may not work without a valid API key")
            print("   Set it in your .env file or environment")
        
        # Initialize and launch UI
        if logger:
            logger.info(f"Starting Trading System with {args.server} server")
        
        if args.server == "uvicorn":
            # Launch with Uvicorn server
            from server import run_server
            
            print(f"\n🚀 Launching Trading System with Uvicorn...")
            print(f"   Host: {args.host}")
            print(f"   Port: {args.port}")
            print(f"   Workers: {args.workers}")
            print(f"   Reload: {args.reload}")
            print(f"\n📱 Open your browser and navigate to:")
            print(f"   http://{args.host}:{args.port}")
            print(f"\n💡 Press Ctrl+C to stop the server")
            
            run_server(
                host=args.host,
                port=args.port,
                workers=args.workers,
                reload=args.reload,
                log_level=args.log_level.lower()
            )
        else:
            # Launch with Gradio's built-in server
            ui = TradingSystemUI(config_path=args.config_dir)
            
            print(f"\n🌐 Launching Trading System with Gradio...")
            print(f"   Host: {args.host}")
            print(f"   Port: {args.port}")
            print(f"   Share: {args.share}")
            print(f"\n📱 Open your browser and navigate to:")
            print(f"   http://{args.host}:{args.port}")
            print(f"\n💡 Press Ctrl+C to stop the server")
            
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

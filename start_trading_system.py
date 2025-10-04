#!/usr/bin/env python3
"""
Unified startup script for the Trading System.
This script can start either the Gradio interface directly or the FastAPI server with Gradio.
"""

import os
import sys
import logging
import argparse
import signal
import time
from pathlib import Path
from typing import Optional

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger, configure_logging
from utils.config import ConfigManager


class TradingSystemLauncher:
    """Main launcher for the Trading System."""
    
    def __init__(self):
        """Initialize the launcher."""
        self.logger = None
        self.server_process = None
        self.gradio_process = None
        self.running = False
        
    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration."""
        try:
            # Default logging configuration
            logging_config = {
                "name": "trading_system_launcher",
                "level": log_level,
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
            self.logger = setup_logger("launcher")
            self.logger.info("Logging configuration completed")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            return False
        
        return True
    
    def setup_environment(self):
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
                self.logger.info(f"Loaded environment variables from {env_file}")
            
            self.logger.info("Environment setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up environment: {e}")
            return False
    
    def check_dependencies(self):
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
            self.logger.error("Missing required packages:")
            for package in missing_required:
                self.logger.error(f"   - {package}")
            self.logger.error("Install with: pip install -r requirements.txt")
            return False
        
        if missing_optional:
            self.logger.warning("Missing optional packages (some features may not work):")
            for package in missing_optional:
                self.logger.warning(f"   - {package}")
            self.logger.warning("Install with: pip install -r requirements.txt")
        
        self.logger.info("All required dependencies are installed")
        return True
    
    def start_gradio_only(self, host: str = "0.0.0.0", port: int = 7860, share: bool = False):
        """Start only the Gradio interface."""
        try:
            self.logger.info("Starting Gradio interface only...")
            
            from ui.gradio_app import TradingSystemUI
            
            # Initialize UI
            ui = TradingSystemUI()
            
            self.logger.info(f"Launching Gradio interface on {host}:{port}")
            print(f"\n🌐 Trading System Interface")
            print(f"   Host: {host}")
            print(f"   Port: {port}")
            print(f"   Share: {share}")
            print(f"\n📱 Open your browser and navigate to:")
            print(f"   http://{host}:{port}")
            print(f"\n💡 Press Ctrl+C to stop the server")
            
            # Launch Gradio
            ui.launch(
                share=share,
                server_name=host,
                server_port=port,
                show_error=True,
                quiet=False
            )
            
        except Exception as e:
            self.logger.error(f"Error starting Gradio interface: {e}")
            raise
    
    def start_fastapi_server(self, host: str = "0.0.0.0", port: int = 8000, 
                           workers: int = 1, reload: bool = False):
        """Start the FastAPI server with Gradio."""
        try:
            self.logger.info("Starting FastAPI server with Gradio...")
            
            from server import run_server
            
            print(f"\n🚀 Trading System Server")
            print(f"   Host: {host}")
            print(f"   Port: {port}")
            print(f"   Workers: {workers}")
            print(f"   Reload: {reload}")
            print(f"\n📱 Open your browser and navigate to:")
            print(f"   http://{host}:{port}")
            print(f"\n💡 Press Ctrl+C to stop the server")
            
            # Run server
            run_server(
                host=host,
                port=port,
                workers=workers,
                reload=reload,
                log_level="info"
            )
            
        except Exception as e:
            self.logger.error(f"Error starting FastAPI server: {e}")
            raise
    
    def print_welcome_message(self):
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
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        sys.exit(0)
    
    def run(self, args):
        """Main run method."""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Print welcome message
            self.print_welcome_message()
            
            # Setup logging
            if not self.setup_logging(args.log_level):
                return 1
            
            # Setup environment
            if not self.setup_environment():
                return 1
            
            # Check dependencies
            if not self.check_dependencies():
                return 1
            
            # Check for API key
            if not os.getenv('DATABENTO_API_KEY'):
                self.logger.warning("DATABENTO_API_KEY not found in environment variables")
                self.logger.warning("Some features may not work without a valid API key")
                self.logger.warning("Set it in your .env file or environment")
            
            self.running = True
            
            # Start the appropriate service
            if args.mode == "gradio":
                self.start_gradio_only(
                    host=args.host,
                    port=args.port,
                    share=args.share
                )
            elif args.mode == "server":
                self.start_fastapi_server(
                    host=args.host,
                    port=args.port,
                    workers=args.workers,
                    reload=args.reload
                )
            else:
                self.logger.error(f"Unknown mode: {args.mode}")
                return 1
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Trading System stopped by user")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error running Trading System: {e}")
            return 1


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Trading System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start Gradio interface only (recommended for development)
  python start_trading_system.py --mode gradio
  
  # Start FastAPI server with Gradio (recommended for production)
  python start_trading_system.py --mode server
  
  # Start with custom host and port
  python start_trading_system.py --mode gradio --host 127.0.0.1 --port 8080
  
  # Start with public sharing (Gradio only)
  python start_trading_system.py --mode gradio --share
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["gradio", "server"],
        default="gradio",
        help="Launch mode: 'gradio' for direct Gradio interface, 'server' for FastAPI server"
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
        help="Port to bind the server to (default: 7860 for gradio, 8000 for server)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the interface (gradio mode only)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (server mode only)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (server mode only)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only run setup and exit"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Adjust default port based on mode
    if args.mode == "server" and args.port == 7860:
        args.port = 8000
    
    launcher = TradingSystemLauncher()
    
    if args.setup_only:
        # Just run setup
        launcher.setup_logging(args.log_level)
        launcher.setup_environment()
        success = launcher.check_dependencies()
        print("✅ Setup completed" if success else "❌ Setup failed")
        return 0 if success else 1
    
    return launcher.run(args)


if __name__ == "__main__":
    sys.exit(main())

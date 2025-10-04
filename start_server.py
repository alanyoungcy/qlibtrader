#!/usr/bin/env python3
"""
Simple startup script for the Trading System with Uvicorn.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from server import run_server


def main():
    """Main function for starting the server."""
    parser = argparse.ArgumentParser(description="Start Trading System with Uvicorn")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Log level (default: info)"
    )
    
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run in production mode (optimized settings)"
    )
    
    args = parser.parse_args()
    
    # Production mode settings
    if args.production:
        args.workers = max(1, os.cpu_count() or 1)
        args.log_level = "warning"
        args.reload = False
        print(f"🚀 Production mode: {args.workers} workers, log level: {args.log_level}")
    
    # Development mode settings
    if args.reload:
        args.workers = 1  # Reload doesn't work with multiple workers
        print("🔧 Development mode: auto-reload enabled")
    
    print("\n" + "="*60)
    print("🚀 Trading System Server")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print(f"Auto-reload: {args.reload}")
    print("="*60)
    print(f"\n📱 Open your browser and navigate to:")
    print(f"   http://{args.host}:{args.port}")
    print(f"\n💡 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Start the server
    try:
        run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

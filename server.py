"""
FastAPI server wrapper for the Trading System Gradio application.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from utils.logger import setup_logger, configure_logging
from utils.config import ConfigManager


# Global variables for the app and UI
trading_ui = None
gradio_app = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    global trading_ui, gradio_app
    
    # Startup
    logger = setup_logger("server")
    logger.info("Starting Trading System Server")
    
    try:
        # Import Gradio and UI components here to avoid import issues
        import gradio as gr
        from ui.gradio_app import TradingSystemUI
        
        # Initialize trading system UI
        trading_ui = TradingSystemUI()
        
        # Create Gradio app
        gradio_app = trading_ui.create_interface()
        
        # Mount Gradio app
        app = gr.mount_gradio_app(app, gradio_app, path="/")
        
        logger.info("Trading System Server started successfully")
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Trading System Server")


# Create FastAPI app
app = FastAPI(
    title="Trading System API",
    description="AI-Powered Trading System with Databento, Qlib, and Machine Learning",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint that redirects to the Gradio interface."""
    return HTMLResponse("""
    <html>
        <head>
            <title>Trading System</title>
            <meta http-equiv="refresh" content="0; url=/gradio/">
        </head>
        <body>
            <p>Redirecting to Trading System...</p>
            <p><a href="/gradio/">Click here if not redirected automatically</a></p>
        </body>
    </html>
    """)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "trading-system",
        "version": "0.1.0"
    }


@app.get("/api/status")
async def api_status():
    """API status endpoint."""
    return {
        "api": "running",
        "gradio_available": gradio_app is not None,
        "ui_initialized": trading_ui is not None
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    if trading_ui is None:
        return {"error": "Trading system not initialized"}
    
    try:
        config_manager = ConfigManager()
        return {
            "data_config": config_manager.get_data_config(),
            "strategy_config": config_manager.get_strategy_config(),
            "model_config": config_manager.get_model_config()
        }
    except Exception as e:
        return {"error": str(e)}


# Mount static files if they exist
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info"
):
    """
    Run the server using Uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        reload: Enable auto-reload for development
        log_level: Log level
    """
    logger = setup_logger("server")
    
    logger.info(f"Starting Trading System Server")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Log Level: {log_level}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Reload doesn't work with multiple workers
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading System Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )

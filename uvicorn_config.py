"""
Uvicorn configuration for production deployment.
"""

import os
import multiprocessing
from pathlib import Path


# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", multiprocessing.cpu_count()))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()

# SSL configuration (for production)
SSL_KEYFILE = os.getenv("SSL_KEYFILE")
SSL_CERTFILE = os.getenv("SSL_CERTFILE")

# Application configuration
APP_MODULE = "server:app"
APP_DIR = Path(__file__).parent

# Logging configuration
ACCESS_LOG = os.getenv("ACCESS_LOG", "true").lower() == "true"
USE_COLORS = os.getenv("USE_COLORS", "false").lower() == "true"

# Development settings
RELOAD = os.getenv("RELOAD", "false").lower() == "true"
RELOAD_DIRS = [str(APP_DIR)] if RELOAD else None

# Production optimizations
KEEP_ALIVE_TIMEOUT = int(os.getenv("KEEP_ALIVE_TIMEOUT", 5))
TIMEOUT_GRACEFUL_SHUTDOWN = int(os.getenv("TIMEOUT_GRACEFUL_SHUTDOWN", 30))
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 1000))
MAX_REQUESTS_JITTER = int(os.getenv("MAX_REQUESTS_JITTER", 100))

# Security headers
SECURE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}


def get_uvicorn_config():
    """
    Get Uvicorn configuration dictionary.
    
    Returns:
        Dictionary with Uvicorn configuration
    """
    config = {
        "app": APP_MODULE,
        "host": HOST,
        "port": PORT,
        "log_level": LOG_LEVEL,
        "access_log": ACCESS_LOG,
        "use_colors": USE_COLORS,
        "reload": RELOAD,
        "keep_alive_timeout": KEEP_ALIVE_TIMEOUT,
        "timeout_graceful_shutdown": TIMEOUT_GRACEFUL_SHUTDOWN,
    }
    
    # Add reload directories if in development mode
    if RELOAD and RELOAD_DIRS:
        config["reload_dirs"] = RELOAD_DIRS
    
    # Add SSL configuration if provided
    if SSL_KEYFILE and SSL_CERTFILE:
        config["ssl_keyfile"] = SSL_KEYFILE
        config["ssl_certfile"] = SSL_CERTFILE
    
    # Production optimizations
    if not RELOAD:  # Only apply in production
        config.update({
            "workers": WORKERS,
            "max_requests": MAX_REQUESTS,
            "max_requests_jitter": MAX_REQUESTS_JITTER,
        })
    
    return config


def get_gunicorn_config():
    """
    Get Gunicorn configuration for production deployment.
    
    Returns:
        Dictionary with Gunicorn configuration
    """
    return {
        "bind": f"{HOST}:{PORT}",
        "workers": WORKERS,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "worker_connections": 1000,
        "max_requests": MAX_REQUESTS,
        "max_requests_jitter": MAX_REQUESTS_JITTER,
        "keepalive": KEEP_ALIVE_TIMEOUT,
        "timeout": 30,
        "graceful_timeout": TIMEOUT_GRACEFUL_SHUTDOWN,
        "preload_app": True,
        "accesslog": "-" if ACCESS_LOG else None,
        "errorlog": "-",
        "loglevel": LOG_LEVEL,
        "access_log_format": '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
    }


if __name__ == "__main__":
    # Print configuration for debugging
    import json
    
    print("Uvicorn Configuration:")
    print(json.dumps(get_uvicorn_config(), indent=2))
    
    print("\nGunicorn Configuration:")
    print(json.dumps(get_gunicorn_config(), indent=2))

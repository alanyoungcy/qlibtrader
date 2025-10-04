# Trading System Startup Guide

## Quick Start

The trading system now has a unified startup script that handles both Gradio and FastAPI server modes.

### Option 1: Gradio Interface Only (Recommended for Development)

```bash
# Start the Gradio interface directly
python start_trading_system.py --mode gradio

# Start with custom host and port
python start_trading_system.py --mode gradio --host 127.0.0.1 --port 8080

# Start with public sharing (creates a public URL)
python start_trading_system.py --mode gradio --share
```

### Option 2: FastAPI Server with Gradio (Recommended for Production)

```bash
# Start the FastAPI server with Gradio
python start_trading_system.py --mode server

# Start with custom settings
python start_trading_system.py --mode server --host 0.0.0.0 --port 8000 --workers 2
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Launch mode: `gradio` or `server` | `gradio` |
| `--host` | Host to bind to | `0.0.0.0` |
| `--port` | Port to bind to | `7860` (gradio) or `8000` (server) |
| `--share` | Create public link (gradio only) | `False` |
| `--workers` | Number of workers (server only) | `1` |
| `--reload` | Enable auto-reload (server only) | `False` |
| `--log-level` | Logging level | `INFO` |
| `--setup-only` | Run setup and exit | `False` |

## Examples

### Development Mode
```bash
# Start Gradio interface for development
python start_trading_system.py --mode gradio --host 127.0.0.1 --port 7860
```

### Production Mode
```bash
# Start FastAPI server for production
python start_trading_system.py --mode server --host 0.0.0.0 --port 8000 --workers 4
```

### Public Sharing
```bash
# Start with public URL (useful for demos)
python start_trading_system.py --mode gradio --share
```

### Setup Only
```bash
# Just run setup and check dependencies
python start_trading_system.py --setup-only
```

## Troubleshooting

### If you get dependency errors:
```bash
# Install missing packages
pip install -r requirements.txt

# Or install specific packages
pip install plotly scikit-learn pyyaml gradio
```

### If you get connection errors:
1. Try using `--mode gradio` instead of the server mode
2. Check if the port is already in use
3. Try a different port with `--port 8080`

### If you get import errors:
```bash
# Make sure you're in the project directory
cd /path/to/qlibtrader

# Check Python path
python -c "import sys; print(sys.path)"
```

## Accessing the Interface

- **Gradio Mode**: Open your browser to `http://localhost:7860` (or your custom host:port)
- **Server Mode**: Open your browser to `http://localhost:8000` (or your custom host:port)

## Stopping the System

Press `Ctrl+C` in the terminal to stop the system gracefully.

## Environment Variables

Make sure to set your API keys in a `.env` file:

```bash
# Create .env file
echo "DATABENTO_API_KEY=your_api_key_here" > .env
```

## Logs

Logs are stored in the `logs/` directory:
- `trading_system.log` - Main application logs
- `ui.log` - UI-specific logs
- `data_fetcher.log` - Data fetching logs
- `analyzer.log` - Analysis logs
- `backtester.log` - Backtesting logs

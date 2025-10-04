# 🚀 AI-Powered Trading System

A comprehensive trading system that integrates Databento market data, Qlib analysis, machine learning models, and advanced backtesting capabilities through a modern web interface.

## ✨ Features

- **📊 Data Management**: Fetch and manage market data from Databento
- **🔍 Advanced Analysis**: Qlib-powered analysis and ML predictions
- **📈 Backtesting Engine**: Comprehensive backtesting with performance metrics
- **🎯 Trading Strategies**: Momentum and mean reversion strategies
- **📱 Modern UI**: Intuitive Gradio-based web interface
- **⚙️ Configurable**: YAML-based configuration system
- **🧪 Testable**: Comprehensive test suite
- **🚀 Multiple Startup Options**: Choose the best interface for your needs

## 🏗️ Architecture

```
trading_system/
├── core/                    # Core trading components
│   ├── data_fetcher.py     # Databento integration
│   ├── analyzer.py         # Qlib analysis wrapper
│   ├── backtester.py       # Backtesting engine
│   └── ml_models.py        # Sklearn models
├── strategies/             # Trading strategies
│   ├── base_strategy.py    # Abstract base strategy
│   ├── momentum_strategy.py
│   └── mean_reversion_strategy.py
├── ui/                     # Web interface
│   ├── gradio_app.py       # Main Gradio app
│   └── components/         # UI panels
├── utils/                  # Utilities
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging utilities
│   └── validators.py      # Data validation
├── configs/               # Configuration files
├── tests/                 # Test suite
└── data/                  # Data storage
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Databento API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd qlibtrader
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env_template .env
   # Edit .env and add your DATABENTO_API_KEY
   ```

4. **Run the application**

   **🚀 Recommended: Working Trading System (Stable)**
   ```bash
   python working_trading_app.py
   ```
   - Access at: `http://127.0.0.1:7860`
   - Full functionality with stable interface

   **🔧 Alternative: Simplified Trading System**
   ```bash
   python start_simple_trading.py
   ```
   - Access at: `http://0.0.0.0:7860`
   - Simplified interface for development

   **🧪 Testing: Minimal Interface**
   ```bash
   python minimal_working_app.py
   ```
   - Access at: `http://127.0.0.1:7860`
   - Basic functionality test

   **⚙️ Original System (Advanced)**
   ```bash
   python start_trading_system.py --mode gradio
   ```
   - Access at: `http://0.0.0.0:7860`
   - Full original interface (may have stability issues)

   **🏭 Production: FastAPI Server**
   ```bash
   python start_trading_system.py --mode server
   ```
   - Access at: `http://0.0.0.0:8000`
   - Production-ready FastAPI server

5. **Open your browser**
   - Working/Simplified: `http://127.0.0.1:7860`
   - Original: `http://0.0.0.0:7860`
   - Production: `http://0.0.0.0:8000`

## 📖 Usage

### 1. Data Management
- Enter symbols (e.g., AAPL, MSFT, GOOGL)
- Set date range and data schema
- Fetch and cache market data

### 2. Analysis & Signals
- Choose analysis type (Qlib, ML, or Strategy)
- Configure parameters
- Generate trading signals

### 3. Backtesting
- Set backtest parameters (capital, commission, etc.)
- Run backtests on historical data
- Analyze performance metrics

### 4. Results & Reports
- View comprehensive reports
- Export results in various formats
- Monitor performance metrics

## ⚙️ Configuration

The system uses YAML configuration files:

- `configs/data_config.yaml` - Data source settings
- `configs/strategy_config.yaml` - Strategy parameters
- `configs/model_config.yaml` - ML model settings

### Environment Variables

Create a `.env` file with:
```bash
DATABENTO_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_data_fetcher.py
pytest tests/test_analyzer.py
pytest tests/test_backtester.py
```

Run Gradio diagnostic tool:
```bash
python diagnose_gradio.py
```

## 🔧 Troubleshooting

### Common Issues

**1. Gradio Interface Disconnects**
- Use the working trading app: `python working_trading_app.py`
- Check Gradio version: `pip show gradio`
- Run diagnostics: `python diagnose_gradio.py`

**2. Missing Dependencies**
```bash
pip install -r requirements.txt
# Or install specific packages:
pip install plotly scikit-learn pyyaml gradio
```

**3. Port Already in Use**
```bash
# Use a different port:
python working_trading_app.py --port 8080
# Or kill existing processes:
pkill -f "python.*trading"
```

**4. API Key Issues**
- Create `.env` file: `cp env_template .env`
- Add your Databento API key to `.env`
- Restart the application

### Startup Options

| Script | Purpose | Stability | Features |
|--------|---------|-----------|----------|
| `working_trading_app.py` | **Recommended** | ✅ High | Full functionality |
| `start_simple_trading.py` | Development | ✅ High | Simplified UI |
| `minimal_working_app.py` | Testing | ✅ High | Basic features |
| `start_trading_system.py` | Advanced | ⚠️ Medium | Full original UI |

## 📊 Strategies

### Momentum Strategy
- Identifies trending price movements
- Uses RSI, MACD, and volume confirmation
- Configurable lookback periods and thresholds

### Mean Reversion Strategy
- Detects overbought/oversold conditions
- Uses Bollinger Bands and Z-score
- Volume confirmation for signal validation

## 🚀 Deployment Options

### Docker Deployment

**Build and run with Docker:**
```bash
# Build the image
docker build -t trading-system .

# Run with environment variables
docker run -p 8000:8000 \
  -e DATABENTO_API_KEY=your_key_here \
  trading-system
```

**Using Docker Compose:**
```bash
# Create .env file with your API key
echo "DATABENTO_API_KEY=your_key_here" > .env

# Start the application
docker-compose up -d

# With reverse proxy
docker-compose --profile proxy up -d
```

### Production Deployment

**Using Gunicorn with Uvicorn workers:**
```bash
pip install gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Environment Variables for Production:**
```bash
export HOST=0.0.0.0
export PORT=8000
export WORKERS=4
export LOG_LEVEL=warning
export ACCESS_LOG=true
```

### Development vs Production

**Development Mode:**
```bash
python start_server.py --reload --log-level debug
```

**Production Mode:**
```bash
python start_server.py --production --workers 4
```

## 🔧 Development

### Adding New Strategies

1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement `generate_signals()` method
3. Add strategy to the UI configuration

### Adding New Analysis Methods

1. Extend the `QlibAnalyzer` or `MLAnalyzer` classes
2. Implement analysis methods
3. Update the UI to include new options

### Customizing the UI

1. Modify panel components in `ui/components/`
2. Add new tabs or features to `ui/gradio_app.py`
3. Update styling in the CSS sections

## 📈 Performance Metrics

The backtesting engine calculates:
- Total Return and Annualized Return
- Sharpe Ratio and Sortino Ratio
- Maximum Drawdown
- Win Rate and Trade Statistics
- Risk-adjusted metrics

## 🔒 Security

- API keys stored in environment variables
- Input validation for all user inputs
- Secure data caching and storage
- Error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Databento](https://databento.com/) for market data
- [Qlib](https://github.com/microsoft/qlib) for quantitative analysis
- [Gradio](https://gradio.app/) for the web interface
- [Backtrader](https://www.backtrader.com/) for backtesting

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the test examples

## 🔄 Changelog

### v0.1.0
- Initial release
- Basic data fetching and analysis
- Momentum and mean reversion strategies
- Gradio web interface
- Comprehensive backtesting engine

#!/bin/bash

# QlibTrader Setup Script
# This script sets up the trading system for new users

echo "🚀 Setting up QlibTrader - AI-Powered Trading System"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔑 Creating .env file..."
    cp env_template .env
    echo "⚠️  Please edit .env and add your DATABENTO_API_KEY"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed data/results logs

# Set permissions for executable scripts
echo "🔧 Setting permissions..."
chmod +x *.py
chmod +x ui/*.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your DATABENTO_API_KEY"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the trading system: python working_trading_app.py"
echo "4. Open your browser to: http://127.0.0.1:7860"
echo ""
echo "For more options, see README.md"

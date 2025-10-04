"""
Setup script for the Trading System package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AI-Powered Trading System with Databento, Qlib, and Machine Learning"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="qlibtrader",
    version="0.1.0",
    author="Trading System Team",
    author_email="team@tradingsystem.com",
    description="AI-Powered Trading System with Databento, Qlib, and Machine Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qlibtrader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "technical": [
            "ta-lib>=0.4.24",
        ],
        "crypto": [
            "ccxt>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qlibtrader=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "data/*", "*.md"],
    },
    keywords=[
        "trading", "finance", "quantitative", "machine-learning", "ai",
        "databento", "qlib", "backtesting", "strategy", "investment"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/qlibtrader/issues",
        "Source": "https://github.com/yourusername/qlibtrader",
        "Documentation": "https://github.com/yourusername/qlibtrader/wiki",
    },
)

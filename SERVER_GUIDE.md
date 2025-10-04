# 🚀 Trading System Server Guide

This guide explains the different ways to start the Trading System server using Uvicorn.

## Quick Start Options

### 1. Using the Startup Script (Recommended)
```bash
# Basic startup
python start_server.py

# Development mode with auto-reload
python start_server.py --reload --log-level debug

# Production mode with multiple workers
python start_server.py --production --workers 4
```

### 2. Using Main Script
```bash
# Start with Uvicorn
python main.py --server uvicorn

# With custom port and workers
python main.py --server uvicorn --port 8000 --workers 2
```

### 3. Direct Uvicorn Command
```bash
# Basic Uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000

# With reload for development
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# With multiple workers (production)
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Using Gunicorn (Production)
```bash
# Install Gunicorn first
pip install gunicorn

# Run with Uvicorn workers
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Command Line Options

### Startup Script Options
```bash
python start_server.py [OPTIONS]

Options:
  --host HOST           Host to bind to (default: 0.0.0.0)
  --port PORT           Port to bind to (default: 8000)
  --workers WORKERS     Number of worker processes (default: 1)
  --reload              Enable auto-reload for development
  --log-level LEVEL     Log level: critical, error, warning, info, debug, trace
  --production          Run in production mode (optimized settings)
```

### Main Script Options
```bash
python main.py --server uvicorn [OPTIONS]

Options:
  --host HOST           Host to bind to (default: 0.0.0.0)
  --port PORT           Port to bind to (default: 8000)
  --workers WORKERS     Number of worker processes (default: 1)
  --reload              Enable auto-reload (uvicorn only)
  --log-level LEVEL     Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Environment Variables

You can configure the server using environment variables:

```bash
# Server configuration
export HOST=0.0.0.0
export PORT=8000
export WORKERS=4
export LOG_LEVEL=info
export ACCESS_LOG=true

# SSL configuration (optional)
export SSL_KEYFILE=/path/to/key.pem
export SSL_CERTFILE=/path/to/cert.pem

# Development settings
export RELOAD=true
export USE_COLORS=true
```

## Development vs Production

### Development Mode
- Single worker process
- Auto-reload enabled
- Debug logging
- Colored output

```bash
python start_server.py --reload --log-level debug
```

### Production Mode
- Multiple worker processes (based on CPU cores)
- No auto-reload
- Warning/error logging only
- Optimized settings

```bash
python start_server.py --production
```

## Docker Deployment

### Using Docker Compose
```bash
# Create environment file
echo "DATABENTO_API_KEY=your_key_here" > .env

# Start the application
docker-compose up -d

# With reverse proxy
docker-compose --profile proxy up -d
```

### Using Docker directly
```bash
# Build the image
docker build -t trading-system .

# Run with environment variables
docker run -p 8000:8000 \
  -e DATABENTO_API_KEY=your_key_here \
  -e WORKERS=4 \
  trading-system
```

## Health Checks

The server provides health check endpoints:

- **Health Check**: `GET /health`
- **API Status**: `GET /api/status`
- **Configuration**: `GET /api/config`

Example:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/status
```

## Performance Tuning

### Worker Configuration
- **CPU-bound tasks**: Set workers = CPU cores
- **I/O-bound tasks**: Set workers = 2 * CPU cores + 1
- **Mixed workload**: Start with CPU cores, adjust based on monitoring

### Memory Considerations
- Each worker process uses additional memory
- Monitor memory usage with multiple workers
- Consider using fewer workers on memory-constrained systems

### Example Configurations

**Small server (2 CPU cores, 4GB RAM):**
```bash
python start_server.py --workers 2 --log-level warning
```

**Medium server (4 CPU cores, 8GB RAM):**
```bash
python start_server.py --workers 4 --log-level info
```

**Large server (8 CPU cores, 16GB RAM):**
```bash
python start_server.py --workers 8 --log-level warning
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Use a different port
   python start_server.py --port 8001
   ```

2. **Permission denied:**
   ```bash
   # Make sure the script is executable
   chmod +x start_server.py
   ```

3. **Import errors:**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

4. **Memory issues with multiple workers:**
   ```bash
   # Reduce number of workers
   python start_server.py --workers 1
   ```

### Logs and Monitoring

- Logs are written to the `logs/` directory
- Use `--log-level debug` for detailed logging
- Monitor system resources (CPU, memory, disk)
- Check application logs for errors

## Security Considerations

### Production Security
- Use HTTPS in production
- Configure firewall rules
- Set up rate limiting
- Use environment variables for secrets
- Regular security updates

### SSL/TLS Configuration
```bash
# With SSL certificates
uvicorn server:app --host 0.0.0.0 --port 443 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

## Monitoring and Maintenance

### Health Monitoring
```bash
# Check server health
curl http://localhost:8000/health

# Check API status
curl http://localhost:8000/api/status
```

### Log Rotation
- Configure log rotation for production
- Monitor log file sizes
- Archive old logs

### Updates
- Keep dependencies updated
- Test updates in development first
- Use version pinning for production

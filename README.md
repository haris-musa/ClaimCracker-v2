---
title: ClaimCracker
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_file: web/main.py
app_port: 10000
pinned: false
license: mit
python_version: "3.11.11"
env:
  - PYTHONUNBUFFERED=1
  - MAX_THREADS=2
  - PYTORCH_NO_CUDA=1
  - OMP_NUM_THREADS=1
  - MKL_NUM_THREADS=1
  - ENVIRONMENT=production
  - LOG_LEVEL=INFO
  - HF_HOME=/tmp/.cache/huggingface
  - LOG_DIR=/tmp/logs
---

# ClaimCracker v2

Fake news detection system combining ML with a modern web API.

## Features

- **ML Pipeline**

  - DistilBERT-based classification (96.03% accuracy)
  - Fast inference (~13.94ms/text)
  - Efficient text preprocessing
  - Production-optimized model loading

- **Web API**
  - FastAPI with async support
  - Request caching and rate limiting
  - Prometheus monitoring
  - Comprehensive error handling
  - OpenAPI documentation

## Environment Setup

### Requirements

- Python 3.11.11
- Conda (recommended) or virtualenv

### Using Conda

```bash
# Create conda environment
conda create -n claimcracker2 python=3.11.11

# Activate environment
conda activate claimcracker2

# Install dependencies
pip install -r requirements.txt
```

### Using virtualenv

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate
# OR (Unix/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Environment Setup**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run API**

```bash
cd web
uvicorn main:app --host 0.0.0.0 --port 10000
```

API will be available at:

- API: http://localhost:10000
- Docs: http://localhost:10000/docs
- Metrics: http://localhost:10000/metrics

## API Endpoints

### Core Endpoints

- `GET /` - Welcome and status
- `GET /health` - Health check
- `POST /predict` - Fake news detection

### Management

- `GET /metrics` - Prometheus metrics
- `GET /cache/stats` - Cache statistics
- `POST /cache/clear` - Clear prediction cache

### Rate Limits

- `/predict`: 30 requests/minute
- `/health`: 60 requests/minute
- `/cache/*`: 10 requests/minute

## Performance

### Model Metrics

- Validation Accuracy: 96.03%
- F1 Score: 0.9603
- Inference Time: 13.94ms/text
- Model Size: <500MB

### API Performance

- Response Time: <1s
- Cache Hit Ratio: ~70%
- Memory Usage: <512MB

## Development

### Prerequisites

- Python 3.11
- FastAPI
- PyTorch
- Transformers

### Testing

```bash
pytest tests/
```

## Project Structure

```
ClaimCracker-v2/
├── Dataset/           # Training data
├── src/              # ML pipeline
│   └── ml/
│       ├── config/   # Configuration
│       ├── data/     # Data processing
│       ├── models/   # Model architecture
│       └── training/ # Training logic
├── web/              # FastAPI application
├── tests/            # Test suite
└── models/           # Model artifacts
```

## Documentation

- API Reference: [docs/api_reference.md](docs/api_reference.md)
- Model Details: [models/final_model/README.md](models/final_model/README.md)

# Use Python 3.11.11 slim image
FROM python:3.11.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create cache directories with correct permissions
RUN mkdir -p /tmp/.cache/huggingface && \
    mkdir -p /tmp/logs && \
    chmod 777 /tmp/.cache/huggingface && \
    chmod 777 /tmp/logs

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies with security flags
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Copy application files
COPY web/ web/
COPY src/ src/
COPY models/final_model/config.json models/final_model/

# Create model directory and download model files with verification
RUN mkdir -p models/final_model && \
    echo "Downloading model file..." && \
    curl -L --retry 3 --retry-delay 2 https://huggingface.co/harismusa/claimcracker-model/resolve/main/model.pt -o models/final_model/model.pt && \
    if [ ! -f models/final_model/model.pt ]; then \
        echo "Failed to download model file" && exit 1; \
    fi && \
    if [ ! -s models/final_model/model.pt ]; then \
        echo "Downloaded model file is empty" && exit 1; \
    fi && \
    echo "Model files in directory:" && \
    ls -la models/final_model/

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    MAX_THREADS=4 \
    PYTORCH_NO_CUDA=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONPATH=/app \
    HF_HOME=/tmp/.cache/huggingface \
    LOG_DIR=/tmp/logs \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    TIMEOUT=120

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/.cache/huggingface /tmp/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 10000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Start command with production settings
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "4", "--timeout-keep-alive", "120"] 
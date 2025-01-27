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
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with correct permissions
RUN mkdir -p /tmp/.cache/huggingface && \
    mkdir -p /tmp/logs && \
    chmod 777 /tmp/.cache/huggingface && \
    chmod 777 /tmp/logs

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY web/ web/
COPY src/ src/
COPY models/final_model/config.json models/final_model/

# Create model directory and download model files with verification
RUN mkdir -p models/final_model && \
    echo "Downloading model file..." && \
    curl -L https://huggingface.co/harismusa/claimcracker-model/resolve/main/model.pt -o models/final_model/model.pt && \
    if [ ! -f models/final_model/model.pt ]; then \
        echo "Failed to download model file" && exit 1; \
    fi && \
    if [ ! -s models/final_model/model.pt ]; then \
        echo "Downloaded model file is empty" && exit 1; \
    fi && \
    echo "Model files in directory:" && \
    ls -la models/final_model/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MAX_THREADS=2
ENV PYTORCH_NO_CUDA=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/.cache/huggingface
ENV LOG_DIR=/tmp/logs

# Expose port
EXPOSE 10000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Start command
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "10000"] 
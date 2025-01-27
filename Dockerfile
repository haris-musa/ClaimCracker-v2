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

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY web/ web/
COPY src/ src/

# Create model directory and download model files
RUN mkdir -p models/final_model && \
    curl -L https://huggingface.co/harismusa/claimcracker-model/resolve/main/model.pt -o models/final_model/model.pt

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

# Start command
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "10000"] 
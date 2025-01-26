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

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY web/ web/
COPY src/ src/

# Create model directory and download model at runtime
RUN mkdir -p models/final_model

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MAX_THREADS=4

# Expose port
EXPOSE 8000

# Create entrypoint script
RUN echo '#!/bin/sh\n\
if [ -n "$MODEL_FILE_URL" ]; then\n\
    echo "Downloading model from $MODEL_FILE_URL..."\n\
    curl -L "$MODEL_FILE_URL" -o models/final_model/model.pt\n\
fi\n\
cd /app/web\n\
exec uvicorn main:app --host 0.0.0.0 --port $PORT\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"] 
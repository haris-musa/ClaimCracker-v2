# Use Python 3.11.11 slim image
FROM python:3.11.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY web/ web/
COPY models/final_model/ models/final_model/
COPY src/ src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MAX_THREADS=4

# Expose port
EXPOSE 8000

# Change to web directory and start the server
WORKDIR /app/web
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
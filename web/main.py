"""FastAPI application for news classification."""

import os
# Set TensorFlow options before any imports that might trigger TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging

import time
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn
from typing import Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

try:
    from web.model_service import model_service
    from web.logging_config import logger
except ImportError:
    from model_service import model_service
    from logging_config import logger

# Initialize metrics
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time spent processing model predictions",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Total number of cache hits"
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total number of cache misses"
)

MEMORY_USAGE = Gauge(
    "memory_usage_bytes",
    "Memory usage in bytes"
)

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="ClaimCracker",
    description="Fake News Detection API",
    version="2.0.0"
)

# Set up Prometheus instrumentation first
instrumentator = Instrumentator()

def track_prediction_time(metrics_dict: Any):
    if hasattr(metrics_dict, 'request') and hasattr(metrics_dict.request, 'url'):
        if metrics_dict.request.url.path == "/predict":
            PREDICTION_LATENCY.observe(metrics_dict.modified_duration)

instrumentator.add(track_prediction_time)
instrumentator.instrument(app).expose(app)

# Add rate limit exceeded error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
origins = os.getenv("CORS_ORIGINS", "*").split(",")
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# Add resource monitoring middleware
@app.middleware("http")
async def monitor_resources(request: Request, call_next):
    """Monitor system resources and cache metrics."""
    # Update resource metrics
    MEMORY_USAGE.set(psutil.Process().memory_info().rss)
    CPU_USAGE.set(psutil.Process().cpu_percent())
    
    # Update cache metrics from previous stats
    stats = model_service.get_cache_stats()
    CACHE_HITS.inc(stats["cache_hits"])
    CACHE_MISSES.inc(stats["cache_misses"])
    
    response = await call_next(request)
    return response

class PredictionRequest(BaseModel):
    text: str

    @validator('text')
    def validate_text(cls, v):
        # Check minimum length
        if len(v.strip()) < 10:
            raise ValueError("Text must be at least 10 characters long")
        
        # Check maximum length (100,000 chars ≈ 20,000 words)
        if len(v) > 100_000:
            raise ValueError("Text is too long. Maximum length is 100,000 characters")
        
        return v.strip()

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

class CacheStats(BaseModel):
    cache_hits: int
    cache_misses: int
    cache_size: int
    max_size: int
    uptime: float

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their responses."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        extra={"extra_data": {
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else None,
            "headers": dict(request.headers)
        }}
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log response
        logger.info(
            "Request completed",
            extra={"extra_data": {
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "processing_time": time.time() - start_time
            }}
        )
        
        return response
        
    except Exception as e:
        # Log error
        logger.error(
            "Request failed",
            exc_info=True,
            extra={"extra_data": {
                "method": request.method,
                "url": str(request.url),
                "error": str(e),
                "processing_time": time.time() - start_time
            }}
        )
        raise

@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    """Welcome endpoint with API status."""
    logger.info("Welcome endpoint called")
    return {
        "message": "Welcome to ClaimCracker API",
        "version": "2.0.0",
        "status": "active"
    }

@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    """Health check endpoint."""
    try:
        # Try to load model
        model_service.load_model()
        logger.info("Health check passed")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(
            "Health check failed",
            exc_info=True,
            extra={"extra_data": {"error": str(e)}}
        )
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict(request: Request, prediction_request: PredictionRequest):
    """Predict if a news article is real or fake.
    
    Args:
        request: FastAPI request object
        prediction_request: News article text
        
    Returns:
        Prediction results including confidence scores
    """
    try:
        # Log input length for monitoring
        logger.info(
            "Prediction request received",
            extra={"extra_data": {"text_length": len(prediction_request.text)}}
        )
        
        result = model_service.predict(prediction_request.text)
        
        # Log prediction result
        logger.info(
            "Prediction completed",
            extra={"extra_data": {
                "prediction": result["prediction"],
                "confidence": result["confidence"]
            }}
        )
        
        return result
    except Exception as e:
        logger.error(
            "Prediction endpoint failed",
            exc_info=True,
            extra={"extra_data": {
                "error": str(e),
                "text_length": len(prediction_request.text)
            }}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/cache/stats", response_model=CacheStats)
@limiter.limit("10/minute")
async def cache_stats(request: Request):
    """Get cache statistics."""
    return model_service.get_cache_stats()

@app.post("/cache/clear")
@limiter.limit("5/minute")
async def clear_cache(request: Request):
    """Clear the prediction cache."""
    model_service.clear_cache()
    return {"status": "Cache cleared"}

@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    """Add rate limit information to response headers."""
    response = await call_next(request)
    
    # Get rate limit data from request state
    rate_limit_data = getattr(request.state, 'view_rate_limit', None)
    if rate_limit_data and isinstance(rate_limit_data, tuple) and len(rate_limit_data) == 2:
        # Rate limit data is a tuple of (remaining, limit)
        remaining, limit = rate_limit_data
        
        # Add headers
        response.headers['X-RateLimit-Limit'] = str(limit)
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        # Calculate reset time (1 minute from now since we use per-minute limits)
        reset_time = int(time.time() + 60)
        response.headers['X-RateLimit-Reset'] = str(reset_time)
    
    return response

if __name__ == "__main__":
    logger.info("Starting API server")
    port = int(os.getenv("PORT", 10000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Server will listen on {host}:{port}")
    uvicorn.run("web.main:app", host=host, port=port, reload=False)  # Use correct module path 
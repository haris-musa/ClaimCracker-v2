"""FastAPI application for news classification."""

import os
# Set TensorFlow options before any imports that might trigger TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging

import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any

from model_service import model_service
from logging_config import logger

app = FastAPI(
    title="ClaimCracker",
    description="Fake News Detection API",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

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
async def root():
    """Welcome endpoint."""
    logger.info("Welcome endpoint called")
    return {
        "message": "Welcome to ClaimCracker API",
        "version": "2.0.0",
        "status": "active"
    }

@app.get("/health")
async def health():
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
async def predict(request: PredictionRequest):
    """Predict if a news article is real or fake.
    
    Args:
        request: News article text
        
    Returns:
        Prediction results including confidence scores
    """
    try:
        result = model_service.predict(request.text)
        return result
    except Exception as e:
        logger.error(
            "Prediction endpoint failed",
            exc_info=True,
            extra={"extra_data": {
                "error": str(e),
                "text_length": len(request.text)
            }}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Get cache statistics."""
    return model_service.get_cache_stats()

@app.post("/cache/clear")
async def clear_cache():
    """Clear the prediction cache."""
    model_service.clear_cache()
    return {"status": "Cache cleared"}

if __name__ == "__main__":
    logger.info("Starting API server")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
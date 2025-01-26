"""FastAPI application for news classification."""

import os
# Set TensorFlow options before any imports that might trigger TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any

from model_service import model_service

app = FastAPI(
    title="ClaimCracker",
    description="Fake News Detection API",
    version="2.0.0"
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

@app.get("/")
async def root():
    """Welcome endpoint."""
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
        return {"status": "healthy"}
    except Exception as e:
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
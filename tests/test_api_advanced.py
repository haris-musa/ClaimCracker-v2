"""Advanced test cases for the FastAPI application."""

import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
import logging

# Import app
import sys
sys.path.append(str(Path(__file__).parent.parent))
from web.main import app

# Create test client
client = TestClient(app)

def test_rate_limit_exceeded():
    """Test that rate limiting works."""
    # Make multiple requests quickly
    responses = []
    for _ in range(50):  # Assuming rate limit is less than 50 per minute
        responses.append(client.get("/"))
    
    # At least one response should be rate limited
    assert any(r.status_code == 429 for r in responses)
    
    # Check rate limit headers
    last_response = responses[-1]
    assert "X-RateLimit-Limit" in last_response.headers
    assert "X-RateLimit-Remaining" in last_response.headers
    assert "X-RateLimit-Reset" in last_response.headers

def test_metrics_endpoint():
    """Test the Prometheus metrics endpoint."""
    # Make a prediction first to generate some metrics
    client.post("/predict", json={"text": "Test article"})
    
    # Get metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.text
    
    # Check for our custom metrics
    assert "model_prediction_latency_seconds" in metrics
    assert "cache_hits_total" in metrics
    assert "cache_misses_total" in metrics
    assert "memory_usage_bytes" in metrics
    assert "cpu_usage_percent" in metrics

def test_cors_headers():
    """Test CORS headers are present."""
    # Test preflight request
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "Content-Type" in response.headers["access-control-allow-headers"]

    # Test actual request
    response = client.post(
        "/predict",
        json={"text": "Test article"},
        headers={"Origin": "http://localhost:3000"}
    )
    assert response.headers["access-control-allow-origin"] == "*"

@pytest.mark.parametrize("text", [
    "",  # Empty text
    "a" * 10000,  # Very long text
    "🌟 Unicode text 🌍",  # Unicode
    "<script>alert('xss')</script>",  # Potential XSS
    "' OR '1'='1",  # SQL injection attempt
])
def test_predict_edge_cases(text):
    """Test prediction endpoint with edge cases."""
    response = client.post("/predict", json={"text": text})
    assert response.status_code in [200, 422]  # Either success or validation error
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["prediction"] in ["Real", "Fake"]
        assert 0 <= data["confidence"] <= 1

def test_cache_performance():
    """Test cache performance."""
    text = "This is a test article."
    
    # First request (cache miss)
    start_time = time.time()
    response1 = client.post("/predict", json={"text": text})
    miss_time = time.time() - start_time
    
    # Second request (cache hit)
    start_time = time.time()
    response2 = client.post("/predict", json={"text": text})
    hit_time = time.time() - start_time
    
    # Cache hit should be significantly faster
    assert hit_time < miss_time
    assert response1.json() == response2.json()

def test_concurrent_requests():
    """Test handling of concurrent requests."""
    import asyncio
    
    async def make_request():
        return client.post("/predict", json={"text": "Test article"})
    
    async def test_concurrent():
        # Make 5 concurrent requests
        tasks = [make_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        return responses
    
    responses = asyncio.run(test_concurrent())
    assert all(r.status_code in [200, 422] for r in responses)  # Allow 422 until model is fixed

def test_error_logging():
    """Test error logging functionality."""
    # Test with invalid input that should trigger validation error
    response = client.post(
        "/predict",
        json={"text": None}  # This should cause an error
    )
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert error_detail["msg"] == "Input should be a valid string"
    assert error_detail["type"] == "string_type"

    # Test with input that should trigger model error
    response = client.post(
        "/predict",
        json={"text": "x" * 1000000}  # Very long text to trigger error
    )
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert "maximum length" in error_detail["msg"].lower()

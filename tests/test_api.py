import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

# Import app and model_service
import sys
sys.path.append(str(Path(__file__).parent.parent))
from web.main import app
from web.model_service import ModelService

# Create test client
client = TestClient(app)

# Mock prediction result
MOCK_PREDICTION = {
    "prediction": "Real",
    "confidence": 0.9493,
    "probabilities": {
        "Real": 0.9493,
        "Fake": 0.0507
    }
}

# Mock cache stats
MOCK_CACHE_STATS = {
    "cache_hits": 10,
    "cache_misses": 5,
    "cache_size": 15,
    "max_size": 1000,
    "uptime": 3600.0
}

@pytest.fixture
def mock_model_service():
    """Fixture to mock the model service."""
    with patch("web.main.model_service") as mock_service:
        # Configure mock methods
        mock_service.predict.return_value = MOCK_PREDICTION
        mock_service.get_cache_stats.return_value = MOCK_CACHE_STATS
        yield mock_service

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to ClaimCracker API"
    assert data["version"] == "2.0.0"
    assert data["status"] == "active"

def test_health_check(mock_model_service):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    mock_model_service.load_model.assert_called_once()

def test_predict_endpoint(mock_model_service):
    """Test the prediction endpoint."""
    test_text = "This is a test news article."
    response = client.post(
        "/predict",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data == MOCK_PREDICTION
    mock_model_service.predict.assert_called_once_with(test_text)

def test_predict_endpoint_invalid_input(mock_model_service):
    """Test the prediction endpoint with invalid input."""
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_model_error(mock_model_service):
    """Test the prediction endpoint when model fails."""
    mock_model_service.predict.side_effect = Exception("Model error")
    response = client.post(
        "/predict",
        json={"text": "test"}
    )
    assert response.status_code == 500
    assert "Model error" in response.json()["detail"]

def test_cache_stats_endpoint(mock_model_service):
    """Test the cache stats endpoint."""
    response = client.get("/cache/stats")
    assert response.status_code == 200
    assert response.json() == MOCK_CACHE_STATS

def test_clear_cache_endpoint(mock_model_service):
    """Test the cache clear endpoint."""
    response = client.post("/cache/clear")
    assert response.status_code == 200
    assert response.json() == {"status": "Cache cleared"}
    mock_model_service.clear_cache.assert_called_once()

def test_rate_limit_headers():
    """Test that rate limit headers are present."""
    response = client.get("/")
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers 

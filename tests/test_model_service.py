import pytest
from unittest.mock import patch, MagicMock
import torch
from pathlib import Path

# Import model service
import sys
sys.path.append(str(Path(__file__).parent.parent))
from web.model_service import ModelService

# Test data
TEST_TEXT = "This is a test news article about technology."
MOCK_LOGITS = torch.tensor([[0.8, 0.2]])
MOCK_PROBS = torch.tensor([[0.9493, 0.0507]])

@pytest.fixture
def mock_newsclassifier():
    """Fixture to mock the NewsClassifier."""
    with patch("web.model_service.NewsClassifier") as mock_cls:
        mock_model = MagicMock()
        mock_model.return_value = {"logits": MOCK_LOGITS}
        mock_cls.from_pretrained.return_value = mock_model
        yield mock_cls

@pytest.fixture
def mock_tokenizer():
    """Fixture to mock the AutoTokenizer."""
    with patch("web.model_service.AutoTokenizer") as mock_tok:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones((1, 10)),
            "attention_mask": torch.ones((1, 10))
        }
        mock_tok.from_pretrained.return_value = mock_tokenizer
        yield mock_tok

@pytest.fixture
def model_service(mock_newsclassifier, mock_tokenizer):
    """Fixture to create a model service instance with mocked components."""
    service = ModelService("test_models")
    return service

def test_model_initialization(model_service):
    """Test model service initialization."""
    assert model_service.model_dir.name == "test_models"
    assert model_service.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_service.labels == ["Real", "Fake"]
    assert model_service.max_cache_size == 1000

def test_load_model(model_service, mock_newsclassifier, mock_tokenizer):
    """Test model loading."""
    model_service.load_model()
    mock_newsclassifier.from_pretrained.assert_called_once_with(model_service.model_dir)
    mock_tokenizer.from_pretrained.assert_called_once_with("distilbert-base-uncased")
    assert model_service.model is not None
    assert model_service.tokenizer is not None

def test_predict(model_service):
    """Test model prediction."""
    result = model_service.predict(TEST_TEXT)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Real", "Fake"]
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 2

def test_cache_functionality(model_service):
    """Test prediction caching."""
    # First prediction
    result1 = model_service.predict(TEST_TEXT)
    stats1 = model_service.get_cache_stats()
    
    # Second prediction (should hit cache)
    result2 = model_service.predict(TEST_TEXT)
    stats2 = model_service.get_cache_stats()
    
    # Check cache hit
    assert result1 == result2
    assert stats2["cache_hits"] > stats1["cache_hits"]

def test_clear_cache(model_service):
    """Test cache clearing."""
    # Make a prediction
    model_service.predict(TEST_TEXT)
    stats1 = model_service.get_cache_stats()
    
    # Clear cache
    model_service.clear_cache()
    stats2 = model_service.get_cache_stats()
    
    assert stats2["cache_hits"] == 0
    assert stats2["cache_misses"] == 0
    assert stats2["cache_size"] == 0

def test_cache_stats(model_service):
    """Test cache statistics."""
    stats = model_service.get_cache_stats()
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    assert "cache_size" in stats
    assert "max_size" in stats
    assert "uptime" in stats
    assert stats["max_size"] == 1000

def test_model_error_handling(model_service, mock_newsclassifier):
    """Test error handling during model operations."""
    mock_newsclassifier.from_pretrained.side_effect = Exception("Model loading error")
    
    with pytest.raises(Exception) as exc_info:
        model_service.load_model()
    assert "Model loading error" in str(exc_info.value) 

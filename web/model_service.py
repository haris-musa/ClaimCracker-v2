"""Model service for news classification."""

import os
import sys
import time
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ml.models.classifier import NewsClassifier
try:
    from web.logging_config import logger
except ImportError:
    from logging_config import logger

# Default paths
DEFAULT_MODEL_DIR = os.path.join(project_root, "models/final_model")
TEST_MODEL_DIR = os.path.join(project_root, "models/test_model")

class ModelService:
    """Service for news classification model."""
    
    def __init__(self, model_dir: str = None):
        """Initialize the model service.
        
        Args:
            model_dir: Directory containing the model files. If None:
                     - Uses TEST_MODEL_DIR if TESTING env var is set
                     - Uses DEFAULT_MODEL_DIR otherwise
        """
        if model_dir is None:
            # Use test model if in testing environment
            if os.getenv("TESTING"):
                model_dir = TEST_MODEL_DIR
            else:
                model_dir = DEFAULT_MODEL_DIR
                
        self.model_dir = Path(model_dir).resolve()
        logger.info(
            "Initializing model service",
            extra={"extra_data": {"model_dir": str(self.model_dir)}}
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[NewsClassifier] = None
        self.tokenizer = None
        self.labels = ["Real", "Fake"]
        
        # Cache settings
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        self.max_cache_size = 1000  # Store up to 1000 predictions
        
        # Initialize cached predict
        self._cached_predict = lru_cache(maxsize=self.max_cache_size)(self._make_prediction)
        
        logger.info(
            "Model service initialized",
            extra={"extra_data": {
                "device": str(self.device),
                "cache_size": self.max_cache_size
            }}
        )
        
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is None:
            logger.info(
                "Loading model",
                extra={"extra_data": {"device": str(self.device)}}
            )
            try:
                self.model = NewsClassifier.from_pretrained(self.model_dir)
                self.model.to(self.device)
                self.model.eval()
                
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(
                    "Failed to load model",
                    exc_info=True,
                    extra={"extra_data": {"error": str(e)}}
                )
                raise
    
    def _make_prediction(self, text: str) -> Dict[str, Any]:
        """Make a prediction without caching."""
        # Validate text length
        if len(text) > 100000:  # 100KB limit
            raise ValueError("Text too long. Maximum length is 100,000 characters.")
        
        if self.model is None:
            self.load_model()
            
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            
        # Get prediction and confidence
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        result = {
            "prediction": self.labels[pred_idx],
            "confidence": confidence,
            "probabilities": {
                label: prob.item()
                for label, prob in zip(self.labels, probs[0])
            }
        }
        
        logger.info(
            "Made prediction",
            extra={"extra_data": {
                "prediction": result["prediction"],
                "confidence": confidence,
                "inference_time": inference_time,
                "text_length": len(text)
            }}
        )
        
        return result

    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction for the given text.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary containing prediction, confidence and probabilities
        """
        # Try to get from cache
        try:
            result = self._cached_predict(text)
            self.cache_hits += 1
            logger.info(
                "Cache hit",
                extra={"extra_data": {
                    "total_hits": self.cache_hits,
                    "cache_size": self._cached_predict.cache_info().currsize
                }}
            )
        except Exception as e:
            self.cache_misses += 1
            logger.info(
                "Cache miss",
                extra={"extra_data": {
                    "total_misses": self.cache_misses,
                    "cache_size": self._cached_predict.cache_info().currsize
                }}
            )
            result = self._make_prediction(text)
            
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        cache_info = self._cached_predict.cache_info()
        stats = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": cache_info.currsize,
            "max_size": self.max_cache_size,
            "uptime": time.time() - self.start_time
        }
        
        logger.info(
            "Cache stats requested",
            extra={"extra_data": stats}
        )
        
        return stats
    
    def clear_cache(self):
        """Clear the prediction cache."""
        logger.info(
            "Clearing cache",
            extra={"extra_data": {
                "cache_size": self._cached_predict.cache_info().currsize,
                "total_hits": self.cache_hits,
                "total_misses": self.cache_misses
            }}
        )
        
        self._cached_predict.cache_clear()
        self.cache_hits = 0
        self.cache_misses = 0

# Create singleton instance
model_service = ModelService() 
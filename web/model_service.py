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

class ModelService:
    """Service for news classification model."""
    
    def __init__(self, model_dir: str = "../models/final_model"):
        """Initialize the model service.
        
        Args:
            model_dir: Directory containing the model files
        """
        self.model_dir = Path(model_dir).resolve()
        print(f"Model directory: {self.model_dir}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[NewsClassifier] = None
        self.tokenizer = None
        self.labels = ["Real", "Fake"]
        
        # Cache settings
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        self.max_cache_size = 1000  # Store up to 1000 predictions
        
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is None:
            print(f"Loading model on device: {self.device}")
            self.model = NewsClassifier.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    @lru_cache(maxsize=1000)
    def _cached_predict(self, text: str) -> Dict[str, Any]:
        """Make a cached prediction.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing prediction results
        """
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
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            
        # Get prediction and confidence
        pred_idx = torch.argmax(probs).item()
        prediction = self.labels[pred_idx]
        confidence = probs[0][pred_idx].item()
        
        # Format probabilities
        prob_dict = {
            label: prob.item()
            for label, prob in zip(self.labels, probs[0])
        }
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction with caching.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing prediction results
        """
        # Ensure model is loaded
        self.load_model()
        
        # Track cache stats
        cache_info = self._cached_predict.cache_info()
        if cache_info.hits > self.cache_hits:
            self.cache_hits = cache_info.hits
        if cache_info.misses > self.cache_misses:
            self.cache_misses = cache_info.misses
            
        # Return cached or new prediction
        return self._cached_predict(text)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        cache_info = self._cached_predict.cache_info()
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": cache_info.currsize,
            "max_size": self.max_cache_size,
            "uptime": time.time() - self.start_time
        }
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._cached_predict.cache_clear()
        self.cache_hits = 0
        self.cache_misses = 0

# Create singleton instance
model_service = ModelService() 
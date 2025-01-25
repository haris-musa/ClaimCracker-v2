from typing import List, Dict, Optional
import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Text preprocessing transformer for news articles."""
    
    def __init__(self,
                 remove_urls: bool = True,
                 remove_numbers: bool = False,
                 lowercase: bool = True,
                 min_word_length: int = 2):
        """Initialize preprocessor.
        
        Args:
            remove_urls: Whether to remove URLs from text
            remove_numbers: Whether to remove numbers
            lowercase: Whether to convert text to lowercase
            min_word_length: Minimum word length to keep
        """
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        
        # URL regex pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
    
    def clean_text(self, text: str) -> str:
        """Clean a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs if specified
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove words shorter than min_length
        words = text.split()
        words = [w for w in words if len(w) >= self.min_word_length]
        
        return ' '.join(words)
    
    def fit(self, X: List[str], y=None):
        """Fit preprocessor (does nothing).
        
        Args:
            X: List of text strings
            y: Labels (not used)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X: List[str]) -> List[str]:
        """Transform a list of text strings.
        
        Args:
            X: List of text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in X] 
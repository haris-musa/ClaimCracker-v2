"""News classification model using transformers."""

from typing import Dict, Any, Optional
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

class NewsClassifier(nn.Module):
    """Transformer-based classifier for fake news detection."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 2,
        dropout: float = 0.1,
        model_name: str = "distilbert-base-uncased"
    ):
        """Initialize the classifier.
        
        Args:
            hidden_size: Size of transformer hidden states
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
            model_name: Name of the pretrained transformer model
        """
        super().__init__()
        
        # Save hyperparameters
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained model and tokenizer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for padding
            labels: Optional ground truth labels
            
        Returns:
            Dictionary containing model outputs and loss
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Prepare outputs
        result = {"logits": logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            result["loss"] = self.criterion(logits, labels)
        
        return result
    
    def prepare_input(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        """Prepare model inputs from raw texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary of model inputs
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return inputs
    
    def save_pretrained(self, save_dir: str) -> None:
        """Save model and tokenizer.
        
        Args:
            save_dir: Directory to save the model
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state with pickle_protocol=5 for better compatibility
        torch.save(
            self.state_dict(),
            save_path / "model.pt",
            pickle_protocol=5
        )
        
        # Save config as a simple dict
        config = {
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "model_name": self.model_name,
            "dropout": self.dropout.p
        }
        torch.save(
            config,
            save_path / "config.pt",
            pickle_protocol=5
        )
    
    @classmethod
    def from_pretrained(cls, load_dir: str) -> "NewsClassifier":
        """Load model from directory.
        
        Args:
            load_dir: Directory containing saved model
            
        Returns:
            Loaded model instance
        """
        load_path = Path(load_dir)
        
        # Load config without weights_only since it contains Python objects
        config = torch.load(
            load_path / "config.pt",
            map_location='cpu'
        )
        
        # Create model instance
        model = cls(**config)
        
        # Load state dict with weights_only since it only contains tensors
        state_dict = torch.load(
            load_path / "model.pt",
            map_location='cpu',
            weights_only=True
        )
        model.load_state_dict(state_dict)
        
        return model 
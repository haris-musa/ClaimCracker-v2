"""Configuration for model training."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model configuration
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 2
    max_length: int = 512
    dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data configuration
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_seed: int = 42
    
    # Paths
    data_dir: Path = Path("Dataset")
    model_dir: Path = Path("models")
    output_dir: Path = Path("outputs")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def device(self) -> str:
        """Get device to use for training."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def save(self, path: str) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        import json
        
        # Convert paths to strings
        config_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from file.
        
        Args:
            path: Path to load configuration from
            
        Returns:
            Loaded configuration
        """
        import json
        
        with open(path) as f:
            config_dict = json.load(f)
        
        # Convert string paths back to Path objects
        for k, v in config_dict.items():
            if k.endswith("_dir"):
                config_dict[k] = Path(v)
        
        return cls(**config_dict) 
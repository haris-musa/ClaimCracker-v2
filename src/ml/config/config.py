from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class MLConfig:
    """Configuration for ML pipeline."""
    
    # Data paths
    data_dir: str = "Dataset"
    models_dir: str = "models"
    
    # Data processing
    text_preprocessing: Dict[str, Any] = None
    max_sequence_length: int = 512
    validation_split: float = 0.1
    test_split: float = 0.2
    random_seed: int = 42
    
    # Training
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 5
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.text_preprocessing is None:
            self.text_preprocessing = {
                "remove_urls": True,
                "remove_numbers": False,
                "lowercase": True,
                "min_word_length": 2
            }
        
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

# Default configuration
default_config = MLConfig() 
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NewsDataset:
    """Dataset handler for news classification."""
    
    def __init__(self, data_dir: str = "Dataset"):
        """Initialize dataset handler.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = Path(data_dir)
        self.dataset_path = self.data_dir / "Dataset.csv"
        
        # Will be loaded on demand
        self._data: Optional[pd.DataFrame] = None
        self._stats: Optional[Dict] = None
    
    def load_data(self) -> pd.DataFrame:
        """Load and cache the combined dataset.
        
        Returns:
            DataFrame containing the news articles
        """
        if self._data is None:
            self._data = pd.read_csv(self.dataset_path)
            # Verify expected columns
            required_columns = [
                "ID", "News_Title", "News_Text", "Label",
                "Published_ Date", "Source", "Source_URL",
                "Author", "Country", "Language", "News_Type"
            ]
            missing_cols = [col for col in required_columns if col not in self._data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        return self._data
    
    def get_stats(self) -> Dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self._stats is None:
            df = self.load_data()
            self._stats = {
                "total_samples": len(df),
                "real_samples": len(df[df["Label"] == "Real"]),
                "fake_samples": len(df[df["Label"] == "Fake"]),
                "features": list(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
                "class_distribution": df["Label"].value_counts(normalize=True).to_dict()
            }
        return self._stats
    
    def prepare_splits(self, 
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      random_state: int = 42
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train, validation and test splits.
        
        Args:
            test_size: Proportion of dataset to include in the test split
            val_size: Proportion of dataset to include in the validation split
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = self.load_data()
        
        # First split: train + val, test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["Label"]
        )
        
        # Second split: train, val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df["Label"]
        )
        
        return train_df, val_df, test_df 
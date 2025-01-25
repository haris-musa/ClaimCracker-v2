"""Dataset analysis script for ClaimCracker v2."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from data.dataset import NewsDataset
from data.preprocessor import TextPreprocessor
from config.config import MLConfig

def analyze_dataset(config: MLConfig) -> Dict[str, Any]:
    """Analyze the news dataset and return statistics.
    
    Args:
        config: ML pipeline configuration
        
    Returns:
        Dictionary containing dataset statistics
    """
    # Initialize dataset
    print("Loading dataset...")
    dataset = NewsDataset(data_dir=config.data_dir)
    
    # Get basic statistics
    stats = dataset.get_stats()
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Real news: {stats['real_samples']}")
    print(f"Fake news: {stats['fake_samples']}")
    print(f"\nFeatures: {', '.join(stats['features'])}")
    
    # Load data for detailed analysis
    df = dataset.load_data()
    
    # Text length analysis
    df['title_length'] = df['News_Title'].str.len()
    df['text_length'] = df['News_Text'].str.len()
    
    print("\nText Length Statistics:")
    print("\nTitle Length:")
    print(df['title_length'].describe())
    print("\nText Length:")
    print(df['text_length'].describe())
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    preprocessor = TextPreprocessor(**config.text_preprocessing)
    sample_text = df['News_Text'].iloc[0]
    cleaned_text = preprocessor.clean_text(sample_text)
    print(f"\nOriginal length: {len(sample_text)}")
    print(f"Cleaned length: {len(cleaned_text)}")
    
    # Test train/val/test split
    print("\nTesting data splits...")
    train_df, val_df, test_df = dataset.prepare_splits(
        test_size=config.test_split,
        val_size=config.validation_split
    )
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return {
        "basic_stats": stats,
        "length_stats": {
            "title": df['title_length'].describe().to_dict(),
            "text": df['text_length'].describe().to_dict()
        },
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        }
    }

def main():
    """Main function to run analysis."""
    config = MLConfig()
    results = analyze_dataset(config)
    
    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    import json
    with open(output_dir / "dataset_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/dataset_analysis.json")

if __name__ == "__main__":
    main() 
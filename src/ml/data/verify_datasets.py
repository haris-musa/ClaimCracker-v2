"""Script to verify relationships between dataset files."""

from pathlib import Path
import pandas as pd

def verify_datasets(data_dir: str = "Dataset") -> None:
    """Verify relationships between dataset files.
    
    Args:
        data_dir: Directory containing the dataset files
    """
    # Load all datasets
    print("Loading datasets...")
    combined_df = pd.read_csv(Path(data_dir) / "Dataset.csv")
    real_df = pd.read_csv(Path(data_dir) / "Dataset - Real.csv")
    fake_df = pd.read_csv(Path(data_dir) / "Dataset - Fake.csv")
    
    # Basic counts
    print("\nDataset sizes:")
    print(f"Combined dataset: {len(combined_df)} rows")
    print(f"Real news dataset: {len(real_df)} rows")
    print(f"Fake news dataset: {len(fake_df)} rows")
    print(f"Sum of Real + Fake: {len(real_df) + len(fake_df)} rows")
    
    # Check if Real + Fake = Combined
    is_sum_equal = len(combined_df) == (len(real_df) + len(fake_df))
    print(f"\nReal + Fake equals Combined size: {is_sum_equal}")
    
    # Verify content overlap
    real_ids = set(real_df['ID'])
    fake_ids = set(fake_df['ID'])
    combined_ids = set(combined_df['ID'])
    
    print("\nID Verification:")
    print(f"IDs in Real dataset: {len(real_ids)}")
    print(f"IDs in Fake dataset: {len(fake_ids)}")
    print(f"IDs in Combined dataset: {len(combined_ids)}")
    
    # Check for duplicates
    print("\nChecking for duplicates...")
    print(f"Duplicate IDs in Real: {len(real_df) - len(real_ids)}")
    print(f"Duplicate IDs in Fake: {len(fake_df) - len(fake_ids)}")
    print(f"Duplicate IDs in Combined: {len(combined_df) - len(combined_ids)}")
    
    # Verify if all Real and Fake IDs are in Combined
    real_in_combined = real_ids.issubset(combined_ids)
    fake_in_combined = fake_ids.issubset(combined_ids)
    
    print("\nContent verification:")
    print(f"All Real news IDs exist in Combined: {real_in_combined}")
    print(f"All Fake news IDs exist in Combined: {fake_in_combined}")
    
    # Check for ID overlap between Real and Fake
    overlap = real_ids.intersection(fake_ids)
    print(f"\nOverlapping IDs between Real and Fake: {len(overlap)}")
    
    if len(overlap) > 0:
        print("Warning: Found overlapping IDs between Real and Fake datasets!")

if __name__ == "__main__":
    verify_datasets() 
# ClaimCracker v2 - Training Notebooks

This directory contains the Jupyter notebook used for model training and experimentation.

## Notebook

`train_model.ipynb` - Main training notebook

- Designed for Google Colab
- Uses GPU acceleration
- Includes visualization and analysis
- Achieves 96.03% validation accuracy
- ~13.94ms inference time per text

## Setup Instructions

1. Upload dataset to Google Drive:

   - Create folder: `ClaimCracker/data`
   - Upload `Dataset.csv`

2. Open notebook in Colab:

   - File > Open notebook > GitHub
   - Enter repository URL
   - Select `notebooks/train_model.ipynb`

3. Connect to GPU runtime:

   - Runtime > Change runtime type
   - Select GPU

4. Run all cells:
   - The notebook will:
     - Mount Google Drive
     - Install dependencies
     - Train model
     - Save results

## Model Outputs

The trained model and configuration are saved to:
`/content/drive/MyDrive/ClaimCracker/final_model/`

Files saved:

- `model.pt` - Model weights
- `config.pt` - Model configuration
- `model_architecture.py` - Model class definition
- `test_model.py` - Example usage and testing
- `README.md` - Documentation

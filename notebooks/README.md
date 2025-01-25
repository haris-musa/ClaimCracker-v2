# ClaimCracker v2 - Training Notebooks

This directory contains Jupyter notebooks for model training and experimentation.

## Notebooks

1. `train_model.ipynb` - Main training notebook
   - Designed for Google Colab
   - Uses GPU acceleration
   - Includes visualization
   - Saves model to Google Drive

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

The trained model and configuration will be saved to:
`/content/drive/MyDrive/ClaimCracker/models/best_model/`

Files saved:

- `model.pt` - Model state dict
- `config.pt` - Model configuration
- `tokenizer/` - Tokenizer files
- `training_config.json` - Training settings

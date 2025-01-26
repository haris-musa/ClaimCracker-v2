# Training Pipeline

This directory contains the training pipeline for ClaimCracker's fake news detection model.

## Notebooks

### `train_model.ipynb`

Production training notebook optimized for Google Colab.

**Features**:

- GPU-accelerated training
- Automatic mixed precision
- Early stopping with F1 metric
- Performance visualization
- Model export and testing

**Performance**:

- Validation Accuracy: 96.03%
- F1 Score: 0.9603
- Inference Time: 13.94ms/text
- Model Size: <500MB

## Setup

1. **Data Preparation**

```bash
# In Google Drive
ClaimCracker/
└── data/
    └── Dataset.csv  # Combined dataset
```

2. **Colab Setup**

- Open: File > Open notebook > GitHub
- URL: `[repository-url]`
- Select: `notebooks/train_model.ipynb`
- Runtime: GPU (T4/P100)

3. **Training**

- Mount Google Drive
- Install dependencies
- Run training pipeline
- Export model artifacts

## Outputs

Training produces the following in `/content/drive/MyDrive/ClaimCracker/final_model/`:

```
final_model/
├── model.pt              # Model weights
├── config.pt            # Configuration
├── model_architecture.py # Model definition
├── test_model.py        # Usage examples
└── README.md            # Documentation
```

## Performance Optimization

- Batch size: 16 (free-tier compatible)
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup steps: 500
- Early stopping: F1-based

## Resource Usage

- GPU Memory: <8GB
- Training Time: ~1 hour
- Disk Space: <1GB
- RAM: <16GB

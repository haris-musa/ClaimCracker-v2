
# ClaimCracker v2 - Fake News Detection Model

## Model Details
- Architecture: DistilBERT-based classifier
- Performance: 96.03% validation accuracy
- Inference speed: ~13.94ms per text
- Size: Check model.pt and config.pt

## Usage
1. Load the model:
```python
from test_model import load_model, test_prediction

# Simple prediction
result = test_prediction("Your news text here")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Files
- model.pt: Model weights
- config.pt: Model configuration
- model_architecture.py: Model class definition
- test_model.py: Example usage and testing

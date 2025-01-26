import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer
from src.ml.models.classifier import NewsClassifier

@pytest.fixture
def test_text():
    return "This is a test article about technology and science."

def load_model(model_dir):
    # Load config
    config = torch.load(Path(model_dir) / "config.pt", map_location=torch.device('cpu'), weights_only=True)
    
    # Create model
    model = NewsClassifier(**config)
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    return model

def test_prediction(test_text, model_dir="final_model"):
    # Use correct path for test
    current_dir = Path(__file__).parent
    model = load_model(current_dir)
    
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(
        [test_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
    
    # Check output format
    assert len(pred.shape) == 1
    assert pred.item() in [0, 1]  # Binary classification
    assert probs.shape == (1, 2)  # Two class probabilities
    assert torch.allclose(torch.sum(probs, dim=1), torch.tensor([1.0]))

if __name__ == "__main__":
    # Test examples
    texts = [
        "Reuters reports global markets show steady growth.",
        "SHOCKING: Scientists hide the truth about water!"
    ]
    
    for text in texts:
        result = test_prediction(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")

"""
import json
import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer
from src.ml.models.classifier import NewsClassifier

@pytest.fixture
def test_text():
    return "This is a test article about technology and science."

def load_model(model_dir):
    with open(Path(model_dir) / "config.json") as f:
        config = json.load(f)
    
    model = NewsClassifier(**config)
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=torch.device('cpu')))
    model.eval()
    
    return model

def test_prediction(test_text, model_dir="final_model"):
    current_dir = Path(__file__).parent
    model = load_model(current_dir)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(
        [test_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
    
    assert len(pred.shape) == 1
    assert pred.item() in [0, 1]
    assert probs.shape == (1, 2)
    assert torch.allclose(torch.sum(probs, dim=1), torch.tensor([1.0]))

if __name__ == "__main__":
    texts = [
        "Reuters reports global markets show steady growth.",
        "SHOCKING: Scientists hide the truth about water!"
    ]
    
    for text in texts:
        result = test_prediction(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
"""
